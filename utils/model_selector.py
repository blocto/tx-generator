from enum import Enum
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI


class ModelType(Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
    GEMINI_1_0_PRO = "gemini-1.0-pro"
    GEMINI_1_5_PRO = "gemini-1.5-pro"


def get_chat_model(model: ModelType) -> BaseChatModel:
    if model in [ModelType.GPT_4O_MINI, ModelType.GPT_4O]:
        return ChatOpenAI(model=model.value, temperature=0)
    elif model == ModelType.CLAUDE_3_5_SONNET:
        return ChatAnthropic(model=model.value, temperature=0)
    elif model in [ModelType.GEMINI_1_0_PRO, ModelType.GEMINI_1_5_PRO]:
        return ChatVertexAI(model=model.value, temperature=0)
    else:
        raise ValueError(f"Unknown model: {model}")
