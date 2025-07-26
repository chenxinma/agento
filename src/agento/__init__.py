import dataclasses
import os
import time
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Self,
    Union,
)
import uuid

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.tools import AgentDepsT

# OpenAI API compatible models
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., description="A list of messages comprising the conversation")
    max_tokens: Optional[int] = Field(None, description="The maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Sampling temperature")
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress")
    top_p: Optional[float] = Field(1.0, ge=0, le=1, description="Nucleus sampling parameter")
    n: Optional[int] = Field(1, ge=1, le=10, description="Number of completions to generate")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(0.0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2, le=2)

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class StreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]

@dataclasses.dataclass(repr=False)
class HandleContext(Generic[AgentDepsT]):
    """Information about the current call."""

    deps: AgentDepsT
    run_result: Optional[AgentRunResult[str]] = None
    stream_result: Optional[StreamedRunResult[AgentDepsT, str]] = None

class LLMWrapper:
    def __init__(self, 
                agent:Agent[AgentDepsT, str], 
                deps: AgentDepsT = None):
        self.agent = agent
        self.deps = deps
        self._app = FastAPI(title="Pydantic-AI OpenAI API Wrapper", version="1.0.0")

        self._app.add_api_route("/", self._root, methods=["GET"])
        self._app.add_api_route("/v1/models", self._list_models, methods=["GET"])
        self._app.add_api_route("/v1/chat/completions", self._create_chat_completion, methods=["POST"])

        self.before_completion = None
        self.after_completion = None
        self.expected_api_keys = []

    def handle_before_completion(self, func: Callable[[str, str, HandleContext[AgentDepsT]], list[ModelMessage]]) -> Self:
        self.before_completion = func
        return self
    
    def handle_after_completion(self, func: Callable[[str, HandleContext[AgentDepsT]], None]) -> Self:
        self.after_completion = func
        return self

    def set_expected_api_key(self, api_key:str) -> Self:
        self.expected_api_keys.append(api_key)
        return self
    
    @property
    def app(self):
        return self._app
    
    def start(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        import uvicorn
        uvicorn.run(self._app, host=host, port=port, reload=reload)

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
    
    async def _list_models(self, authorization: str = Header(None)):
        # Validate API key
        self._auth(authorization)
 
        return {
            "object": "list",
            "data": [
                {
                    "id": "pydantic-ai-agent",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "pydantic-ai"
                }
            ]
        }
    
    async def _create_chat_completion(self, request: ChatCompletionRequest, authorization: str = Header(None)):
        try:
            # Validate API key
            self._auth(authorization)
            session_id = self._gen_session_id(authorization)

            
            # Convert messages to pydantic-ai format
            conversation = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
            
            if request.stream:
                return StreamingResponse(
                    self._stream_chat_response(request, conversation, session_id),
                    media_type="text/event-stream"
                )
            else:
                return await self._generate_chat_response(request, conversation, session_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _generate_chat_response(self, request: ChatCompletionRequest, conversation: str, session_id:str) -> ChatCompletionResponse:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        message_history = None
        
        # Call session start callback
        if self.before_completion:
            message_history = self.before_completion(session_id, conversation, HandleContext(deps=self.deps))
                
        # Run the agent
        result = await self.agent.run(conversation, 
                                      deps=self.deps, 
                                      message_history=message_history)
        
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
        message_history = None
        
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
            message_history = self.before_completion(session_id, conversation, HandleContext(deps=self.deps))
        
        # Run agent and stream response
        async with self.agent.run_stream(conversation, 
                                         deps=self.deps,
                                         message_history=message_history) as result:
            async for chunk in result.stream_text(delta=True):
                stream_chunk = ChatCompletionStreamResponse(
                    id=response_id,
                    created=int(time.mktime(result.timestamp().timetuple())),
                    model=request.model,
                    choices=[StreamChoice(index=0, delta={"content": chunk})]
                )
                yield f"data: {stream_chunk.model_dump_json()}\n\n"
        
        # Final chunk
        final_chunk = ChatCompletionStreamResponse(
            id=response_id,
            created=int(time.time()),
            model=request.model,
            choices=[StreamChoice(index=0, delta={}, finish_reason="stop")]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

        if self.after_completion:
            self.after_completion(session_id, HandleContext(deps=self.deps, stream_result=result))
