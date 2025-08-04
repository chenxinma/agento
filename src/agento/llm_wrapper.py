from contextlib import AbstractAsyncContextManager
import logging
import os
import time
from typing import AsyncGenerator, Callable, Optional, Self
import uuid

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from agento.extra import qwen_cli_tools

from .chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
    Choice,
    StreamChoice,
    Usage,
)

from .agent import HandleContext

def nvl(val: Optional[int], default: int = 0) -> int:
    return val if val is not None else default

class LLMWrapper:
    def __init__(self, 
                agent:Agent[AgentDepsT, str], 
                deps: AgentDepsT = None,
                agent_name:str="pydantic-ai-agent"):
        self.logger = logging.getLogger('LLMWrapper')
        self.agent = agent
        self.deps = deps
        self.agent_name = agent_name
        self._app = FastAPI(title="Pydantic-AI OpenAI API Wrapper", version="1.0.0")

        # 检测请求头
        # @self._app.middleware("http")
        # async def add_process_time_header(
        #     request: Request, call_next: Callable[[Request], Awaitable[Response]]
        # ) -> Response:
        #     start_time = time.time()
        #     # print(await request.json())
        #     response = await call_next(request)
        #     process_time = time.time() - start_time
        #     response.headers["X-Process-Time"] = str(process_time)
        #     return response
        
        self._app.add_api_route("/", self._root, methods=["GET"])
        self._app.add_api_route("/v1/models", self._list_models, methods=["GET"])
        self._app.add_api_route("/v1/chat/completions", self._create_chat_completion, methods=["POST"])

        self.before_completion = None
        self.after_completion = None
        self.expected_api_keys = []
        self._using_qwen_cli_tools = False

    def handle_before_completion(self, func: Callable[[str, str, HandleContext[AgentDepsT]], list[ModelMessage]]) -> Self:
        self.before_completion = func
        return self
    
    def handle_after_completion(self, func: Callable[[str, HandleContext[AgentDepsT]], None]) -> Self:
        self.after_completion = func
        return self

    def set_expected_api_key(self, api_key:str) -> Self:
        self.expected_api_keys.append(api_key)
        return self
    
    def use_qwen_cli_tools(self, is_use:bool = True) -> Self:
        self._using_qwen_cli_tools = is_use
        return self
    
    @property
    def app(self):
        return self._app
    
    def start(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        import uvicorn
        uvicorn.run(self._app, host=host, port=port, reload=reload)
    
    def start_https(self, ssl_keyfile:str, ssl_certfile:str, host: str = "0.0.0.0", port: int = 8443, reload: bool = False):
        import uvicorn
        uvicorn.run(self._app, host=host, port=port, reload=reload, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)

    async def _root(self):
        return {"message": "Pydantic-AI OpenAI API Wrapper is running"}

    def _auth(self, authorization: str | None):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        
        provided_api_key = authorization.split(" ")[1]
        if not provided_api_key in self.expected_api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    def _gen_session_id(self, authorization: str | None) -> str:
        if not authorization:
            return str(uuid.uuid4())
        _api_key = authorization.split(" ")[1]
        # 生成盐值
        salt = bytes(os.getenv("AGENTO_SALT", "abc"), 'utf-8')
        # 定义密钥派生函数
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        # 对 API key 进行加密
        _api_key = kdf.derive(_api_key.encode())
        _api_key = _api_key.hex()  # 转换为 32 字节长度的十六进制字符串
        return _api_key
    
    async def _list_models(self): 
        return {
            "object": "list",
            "data": [
                {
                    "id": self.agent_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "agento(not OpenAI)"
                }
            ]
        }
    
    async def _create_chat_completion(self, request: ChatCompletionRequest, authorization: str = Header(None)):
        self.logger.debug(f"Received request: {request}")
        try:
            # Validate API key
            self._auth(authorization)
            session_id = self._gen_session_id(authorization)

            
            # Convert messages to pydantic-ai format
            conversation = "\n".join([f"{msg.role}: {str(msg.content)}" for msg in request.messages])
            
            if request.stream:
                return StreamingResponse(
                    self._stream_chat_response(request, conversation, session_id),
                    media_type="text/event-stream"
                )
            else:
                return await self._generate_chat_response(request, conversation, session_id)
        except Exception as e:
            # Handle validation errors specifically
            if hasattr(e, 'errors'):
                # Pydantic validation error
                error_details = []
                for error in e.errors(): # pyright: ignore[reportAttributeAccessIssue]
                    loc = ".".join(str(loc) for loc in error['loc'])
                    error_details.append(f"{loc}: {error['msg']}")
                raise HTTPException(
                    status_code=422,
                    detail={
                        "type": "validation_error",
                        "errors": error_details
                    }
                )
            else:
                # General error
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _generate_chat_response(self, request: ChatCompletionRequest, conversation: str, session_id:str) -> ChatCompletionResponse:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        message_history = None
        _toolsets = None
        
        # Call session start callback
        if self.before_completion:
            message_history = self.before_completion(session_id, conversation, HandleContext(deps=self.deps))
        if self.use_qwen_cli_tools:
            _toolsets = [FunctionToolset(tools=qwen_cli_tools)]
        # Run the agent
        result = await self.agent.run(conversation, 
                                      deps=self.deps, 
                                      message_history=message_history,
                                      toolsets=_toolsets)

        # Create response
        created_time = int(time.time())
        
        # Estimate token counts (simplified)
        prompt_tokens = len(conversation.split())
        completion_tokens = len(str(result.output).split())
        total_tokens = prompt_tokens + completion_tokens
        
        choice = Choice(
            index=0,
            message=ChatMessage(role="assistant", content=str(result.output)),
            finish_reason="stop"
        )
        
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )

        if self.after_completion:
            self.after_completion(session_id, HandleContext(deps=self.deps, run_result=result))

        
        return ChatCompletionResponse(
            id=response_id,
            created=created_time,
            model=request.model,
            choices=[choice],
            usage=usage
        )

    async def _stream_chat_response(self, request: ChatCompletionRequest, conversation: str, session_id:str) -> AsyncGenerator[str, None]:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())
        _message_history = None
        _toolsets = None

        # Start with empty delta
        start_chunk = ChatCompletionStreamResponse(
            id=response_id,
            created=created_time,
            model=request.model,
            choices=[StreamChoice(index=0, delta={"role": "assistant"})]
        )
        yield f"data: {start_chunk.model_dump_json()}\n\n"

        # Call session start callback
        if self.before_completion:
            _message_history = self.before_completion(session_id, conversation, HandleContext(deps=self.deps))
        
        if self.use_qwen_cli_tools:
            _toolsets = [FunctionToolset(tools=qwen_cli_tools)]
        
        agent_usage = None
        # Run agent and stream response
        async with self.agent.run_stream(conversation, 
                                         deps=self.deps,
                                         message_history=_message_history,
                                         toolsets=_toolsets) as result:
            async for chunk in result.stream_text(delta=True):
                stream_chunk = ChatCompletionStreamResponse(
                    id=response_id,
                    created=int(time.mktime(result.timestamp().timetuple())),
                    model=request.model,
                    choices=[StreamChoice(index=0, delta={"content": chunk})]
                )
                yield f"data: {stream_chunk.model_dump_json()}\n\n"
                agent_usage = result.usage()

        # Final chunk
        final_chunk = ChatCompletionStreamResponse(
            id=response_id,
            created=int(time.time()),
            model=request.model,
            choices=[StreamChoice(index=0, delta={}, finish_reason="stop")]
        )
        if request.stream_options and "include_usage" in request.stream_options and agent_usage:
            final_chunk.usage = Usage(
                prompt_tokens=nvl(agent_usage.request_tokens),
                completion_tokens=nvl(agent_usage.response_tokens),
                total_tokens=nvl(agent_usage.total_tokens)
            )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

        if self.after_completion:
            self.after_completion(session_id, HandleContext(deps=self.deps, stream_result=result))
