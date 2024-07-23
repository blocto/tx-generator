from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain.tools.retriever import create_retriever_tool

import os
import json
import numpy as np
from tqdm import tqdm


from case.code_retriever import get_retriever
from case.code_loader import CaseCodeLoader
from utils.model_selector import get_chat_model


def _create_retrieval_tool():
    return create_retriever_tool(
        retriever=get_retriever(),
        name="case_code_retriever",
        description="Retrieves code snippets from the batch case codebase.",
    )


class Tx(BaseModel):
    description: str = Field(description="A summarized description of the transaction.")
    to: str = Field(description="The receiving address of the transaction.")
    value: str = Field(description="The amount of native token to transfer.")
    function_name: str = Field(description="Function name of the transaction.")
    input_args: list[str] = Field(description="Input arguments of the function.")


class Case(BaseModel):
    case_id: str = Field(description="Unique identifier for the case.")
    description: str = Field(description="A brief description of the case.")
    steps: list[Tx] = Field(description="Each step represents a transaction.")


class CaseOutput(BaseModel):
    cases: list[Case] = Field(
        description="List of cases. This could be empty if no cases are found."
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = get_retriever()


def generate(model_name: str = "gpt-4o"):
    model = get_chat_model(model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are a coding assistant with expertise in creating structured batch transactions using TypeScript. 
                    A structured batch is a collection of transactions that can be executed sequentially. 
                    For example, a transaction involving a token in a protocol typically includes an approve function.
                    
                    Answer the user's query using the provided TypeScript code snippets:
                    ---
                    {context}
                    ---
                    
                    Here's the user question:""",
            ),
            ("human", "{input}"),
        ]
    )

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | model.with_structured_output(Case)
    )

    return chain


def transform_case(model_name: str = "gpt-4o"):
    model = get_chat_model(model_name)
    loader = CaseCodeLoader()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a blockchain expert. Create a structured output from the given code snippets.",
            ),
            (
                "human",
                "Transform the provided code below to a structured format. \n\n{code}.",
            ),
        ]
    )
    chain = (
        {"code": RunnablePassthrough()} | prompt | model.with_structured_output(Case)
    )

    output_path = f"data/case_outputs_{model_name}.jsonl"
    # Check if the file exists, and delete it if it does
    if os.path.exists(output_path):
        os.remove(output_path)

    for doc in tqdm(
        loader.lazy_load(), desc="Processing Documents", unit="doc", total=116
    ):
        metadata = doc.metadata
        # Skip documents that do not match the criteria
        if (
            "PartialBatchCase" not in doc.page_content
            or metadata["case"] in ["prebuilt-tx"]
            # or metadata["file"] in ["gauntlet-weth-prime.ts"]
        ):
            continue

        # print(f"#{index}: {metadata['case']}/{metadata['file']}")
        # print("-------------------")
        try:
            result = chain.invoke(doc.page_content)
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Append JSON data to the file
            with open(output_path, "a") as f:
                f.write(json.dumps(result.dict()))
                f.write("\n")
        except Exception as e:
            tqdm.write(f"Error processing {metadata['case']}/{metadata['file']}")


def get_cases():
    with open("raw_data/meta.json", "r") as f:
        cases = json.load(f)

    transformed_data = {
        case["id"]: {
            "chain_id": case["chain_id"],
            "preview_txn_count": case["preview_txn_count"],
        }
        for case in cases
    }
    return transformed_data
