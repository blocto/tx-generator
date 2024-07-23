from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI


def get_chat_model(name: str):
    if name in ["gpt-4o-mini", "gpt-4o"]:
        return ChatOpenAI(model=name, temperature=0)
    elif name in ["claude-3-5-sonnet-20240620"]:
        return ChatAnthropic(model=name, temperature=0)
    elif name in ["gemini-1.5-flash-001", "gemini-1.5-pro-001"]:
        return ChatVertexAI(model=name, temperature=0)
    else:
        raise ValueError(f"Unknown model name: {name}")
