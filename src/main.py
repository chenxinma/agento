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
)

@agent.tool_plain
def get_current_time() -> datetime:
    return datetime.now()

@agent.tool_plain
def list_directory(path: str) -> str:
    """
    列出目录下的文件和文件夹的本地执行指令
    Args:
        path: 目录路径
    Returns:
        xml执行指令
    """
    return f"""<tool_call>
<function=list_directory>
<parameter=path>
{path}
</parameter>
</function>
</tool_call>"""

@agent.tool_plain
def run_shell_command(command: str) -> str:
    """
    执行shell命令的本地执行指令
    Args:
        command: shell命令
    Returns:
        xml执行指令
    """
    return f"""<tool_call>
<function=run_shell_command>
<parameter=command>
{command}
</parameter>
</function>
</tool_call>"""

@agent.tool_plain
def glob(pattern: str) -> str:
    """
    匹配文件路径名的本地执行指令
    Args:
        pattern: 匹配模式
    Returns:
        xml执行指令
    """
    return f"""<tool_call>
<function=glob>
<parameter=path>
{pattern}
</parameter>
</function>
</tool_call>"""

@agent.tool_plain
def read_file(path: str) -> str:
    """
    读取文件内容的本地执行指令
    Args:
        path: 文件路径
    Returns:
        xml执行指令
    """
    return f"""<tool_call>
<function=read_file>
<parameter=path>
{path}
</parameter>
</function>
</tool_call>"""

@agent.tool_plain
def replace(path:str, old_content:str, new_content:str) -> str:
    """
    替换文件内容的本地执行指令
    Args:
        path: 文件路径
        old_content: 旧内容
        new_content: 新内容
    Returns:
        xml执行指令
    """
    return f"""<tool_call>
<function=replace>
<parameter=path>
{path}
</parameter>
<parameter=old_content>
({old_content})
</parameter>
<parameter=new_content>
({new_content})
</parameter>
</function>
</tool_call>"""

@agent.tool_plain
def write_file(path:str) -> str:
    """
    写入文件内容的本地执行指令
    Args:
        path: 文件路径
        content: 文件内容
    Returns:
        xml执行指令
    """
    return f"""<tool_call>
<function=write_file>
<parameter=path>
{path}
</parameter>
</function>
</tool_call>"""



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
        .start()
        # .handle_before_completion(before_completion) \
        # .handle_after_completion(after_completion) \
        
