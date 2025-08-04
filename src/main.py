"""
This module implements a simple agent using the Pydantic-AI library.
The agent is a simple echo agent that returns the input message.
"""
from calendar import c
from datetime import datetime
import os
from typing import List

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from agento import HandleContext, LLMWrapper
import logging

if not os.path.exists("logs"):
    os.mkdir("logs")
# 配置日志输出到文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/agento.log',
    filemode='a'
)

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
    instructions="如果使用了",
)

@agent.tool_plain
def get_current_time() -> datetime:
    return datetime.now()

def before_completion(prompt: str, session_id: str, ctx: HandleContext) -> List[ModelMessage]:
    # print(prompt, session_id)
    return [ModelResponse(parts=[TextPart(content="小红书用户名:我饿了a")])]

def after_completion(session_id: str, ctx: HandleContext) -> None:
    if ctx.stream_result:
        # print(session_id, ctx.stream_result.new_messages)
        pass

if __name__ == "__main__":
    # ca_key = os.getenv('CA_KEY')
    # ca_cert = os.getenv('CA_CERT')
    # if not ca_key or not ca_cert:
    #     raise ValueError("CA_KEY and CA_CERT must be set")
    # if not os.path.exists(ca_key) or not os.path.exists(ca_cert):
    #     raise ValueError("CA_KEY and CA_CERT files must not exist")

    LLMWrapper(agent) \
        .set_expected_api_key("sss") \
        .use_qwen_cli_tools(True) \
        .start()
        # .handle_before_completion(before_completion) \
        # .handle_after_completion(after_completion) \
        
