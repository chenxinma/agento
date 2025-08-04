from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# OpenAI API compatible models
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author")
    content: Union[str, List[Dict[str, str]]] = Field(..., description="The content of the message")

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
    stream_options: Optional[Dict[str, Any]] = Field(None, description="Stream options")
    # 添加工具调用相关字段
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="A list of tools the model may call")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field("auto", description="Controls which tool is called")

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
    usage: Optional[Usage] = None