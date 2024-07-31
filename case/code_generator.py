import os
import json
import time
from tqdm.asyncio import tqdm_asyncio
from langchain_core.document_loaders import BaseLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool

from case.code_retriever import get_retriever
from utils.model_selector import ModelType, get_chat_model


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
    total_steps: int = Field(description="The total transaction count of the case.")
    steps: list[Tx] = Field(description="Each step represents a transaction.")


class CaseOutput(BaseModel):
    cases: list[Case] = Field(
        description="List of cases. This could be empty if no cases are found."
    )


class SkippedFile(BaseModel):
    file_name: str = Field(description="Name of the file that was skipped.")
    error: str = Field(description="Error message that caused the file to be skipped.")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = get_retriever()


def generate(model_type: ModelType):
    model = get_chat_model(model_type)

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


async def transform_case(total_files: int, loader: BaseLoader, model_type: ModelType):
    model = get_chat_model(model_type)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a blockchain expert with extensive knowledge in TypeScript and blockchain transactions. 
                Your task is to assist the user by following these specific guidelines.\n\n
                RULES:\n
                1. Use the user-provided code snippets to create a structured output.\n
                2. The steps are typically detailed in the `previewTx` field.\n
                3. Ensure that the number of steps in the `previewTx` matches the number specified in the `txn_count`.\n
                \n\n
                Follow these rules to provide accurate responses.
                """,
            ),
            ("human", "{code}"),
        ]
    )
    chain = (
        {"code": RunnablePassthrough()} | prompt | model.with_structured_output(Case)
    )

    output_path = f"case/data/case_outputs_{model_type.value}.jsonl"

    # Check if the file exists, and delete it if it does
    if os.path.exists(output_path):
        os.remove(output_path)

    skipped_files: list[SkippedFile] = []
    duration_list = []

    async for doc in tqdm_asyncio(
        loader.alazy_load(),
        desc="Processing Documents",
        unit="doc",
        total=total_files,
    ):
        metadata = doc.metadata
        meta_str = f"{metadata['case']}/{metadata['file']}"
        # Skip documents that do not match the criteria
        if "PartialBatchCase" not in doc.page_content:
            skipped_files.append(
                SkippedFile(file_name=meta_str, error="No case found.")
            )
            continue

        try:
            start_time = time.time()

            result = await chain.ainvoke(doc.page_content)

            end_time = time.time()

            duration_list.append(end_time - start_time)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Append JSON data to the file
            with open(output_path, "a") as f:
                f.write(json.dumps(result.dict()))
                f.write("\n")
        except Exception as e:
            skipped_files.append(
                SkippedFile(file_name=meta_str, error="Unable to process document.")
            )

    return skipped_files


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
