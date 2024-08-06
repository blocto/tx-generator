from enum import Enum
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI


def get_chat_model(provider: str = "openai", temperature: float = 0.7) -> BaseChatModel:
    normalized_provider = provider.strip().lower()
    chat_models = {
        "openai": ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
        ),
        "anthropic": ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=temperature,
        ),
        "google": ChatVertexAI(
            model="gemini-1.5-pro",
            temperature=temperature,
        ),
    }
    try:
        return chat_models[normalized_provider]
    except KeyError:
        raise ValueError(f"Provider must be one of: {', '.join(chat_models.keys())}")
