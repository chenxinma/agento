from pydantic_ai import Agent
import time
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.output import OutputDataT
from pydantic_ai.providers.openai import OpenAIProvider
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

app = FastAPI(title="Pydantic-AI OpenAI API Wrapper", version="1.0.0")

class LLMWrapper:
    def __init__(self, 
                agent:Agent[AgentDepsT, OutputDataT], 
                deps: AgentDepsT = None, 
                history_messages: list[ModelMessage] | None = None):
        self.agent = agent
        self.deps = deps
        self.history_messages = history_messages

        app.add_api_route("/", self.root, methods=["GET"])
        app.add_api_route("/v1/models", self.list_models, methods=["GET"])
        app.add_api_route("/v1/chat/completions", self.create_chat_completion, methods=["POST"])
    
    @property
    def app(self):
        return app

    async def root(self):
        return {"message": "Pydantic-AI OpenAI API Wrapper is running"}
    
    async def list_models(self):
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
    
    async def create_chat_completion(self, request: ChatCompletionRequest):
        try:
            # Convert messages to pydantic-ai format
            conversation = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
            
            if request.stream:
                return StreamingResponse(
                    self.stream_chat_response(request, conversation),
                    media_type="text/event-stream"
                )
            else:
                return await self.generate_chat_response(request, conversation)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def generate_chat_response(self, request: ChatCompletionRequest, conversation: str) -> ChatCompletionResponse:
        # Run the agent
        result = await self.agent.run(conversation, 
                                        deps=self.deps, 
                                        message_history=self.history_messages)
        
        # Create response
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
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
        
        return ChatCompletionResponse(
            id=response_id,
            created=created_time,
            model=request.model,
            choices=[choice],
            usage=usage
        )
    
    async def stream_chat_response(self, request: ChatCompletionRequest, conversation: str) -> AsyncGenerator[str, None]:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())
        
        # Start with empty delta
        start_chunk = ChatCompletionStreamResponse(
            id=response_id,
            created=created_time,
            model=request.model,
            choices=[StreamChoice(index=0, delta={"role": "assistant"})]
        )
        yield f"data: {start_chunk.model_dump_json()}\n\n"
        
        # Run agent and stream response
        async with self.agent.run_stream(conversation, 
                                        deps=self.deps, 
                                        message_history=self.history_messages) as result:
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

