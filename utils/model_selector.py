from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


def get_chat_model(name: str):
    if name in ["gpt-4o-mini", "gpt-4o"]:
        return ChatOpenAI(model=name, temperature=0)
    elif name in ["claude-3-5-sonnet-20240620"]:
        return ChatAnthropic(model=name, temperature=0)
    else:
        raise ValueError(f"Unknown model name: {name}")
