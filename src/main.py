
from datetime import datetime
import os
from typing import List

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelResponsePart, TextPart
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from agento import HandleContext, LLMWrapper

# Initialize pydantic-ai agent
load_dotenv()
base_url = os.getenv('BASE_URL')
api_key = os.getenv('API_KEY')
model_name = os.getenv('MODEL_NAME', "qwen-max")
model = OpenAIModel(
    model_name,
    provider=OpenAIProvider(
        base_url=base_url,
        api_key=api_key,
    ),
)
agent = Agent(
    model,
    instructions='You are a helpful assistant.',
)

@agent.tool_plain
def get_current_time() -> datetime:
    return datetime.now()

def before_completion(prompt: str, session_id: str, ctx: HandleContext) -> List[ModelMessage]:
    print(prompt, session_id)
    return [ModelResponse(parts=[TextPart(content="小红书用户名:我饿了a")])]

def after_completion(session_id: str, ctx: HandleContext) -> None:
    if ctx.stream_result:
        print(session_id, ctx.stream_result.new_messages)

if __name__ == "__main__":

    LLMWrapper(agent) \
        .set_expected_api_key("sss") \
        .handle_before_completion(before_completion) \
        .handle_after_completion(after_completion) \
        .start()
