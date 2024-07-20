from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent

from case.code_retriever import get_retriever


def _create_retrieval_tool():
    return create_retriever_tool(
        retriever=get_retriever(),
        name="case_code_retriever",
        description="Retrieves code snippets from the batch case codebase.",
    )


tools = [_create_retrieval_tool()]


def get_generator() -> Runnable:
    agent = create_react_agent(
        model=ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            streaming=True,
        ),
        tools=tools,
    )
    return agent
