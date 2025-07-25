
from datetime import datetime
import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from agento import LLMWrapper

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

if __name__ == "__main__":
    llm = LLMWrapper(agent)
    import uvicorn
    uvicorn.run(llm.app, host="0.0.0.0", port=8000)
