import os
import json
from enum import Enum
from typing import List, Dict
from io import TextIOWrapper
from tqdm.asyncio import tqdm_asyncio
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from utils.model_selector import get_chat_model


class Tx(BaseModel):
    description: str = Field(description="A summarized description of the transaction.")
    to: str = Field(description="The receiving address of the transaction.")
    value: str = Field(description="The amount of native token to transfer.")
    function_name: str = Field(description="Function name of the transaction.")
    input_args: list[str] = Field(
        description="Input arguments of the function. If the value is not determined, use `{user_input}`."
    )


class Case(BaseModel):
    case_id: str = Field(description="Unique identifier for the case.")
    description: str = Field(description="A brief description of the case.")
    total_steps: int = Field(description="The total transaction count of the case.")
    steps: list[Tx] = Field(description="Each step represents a transaction.")


class CaseOutput(BaseModel):
    cases: list[Case] = Field(
        description="List of cases. This could be empty if no cases are found."
    )


class TransformError(Enum):
    NotFoundError = "Case Not Found"
    ParseError = "Parse Error"


async def process_document(
    doc: Document,
    chain: Runnable,
    output_path: str,
    skipped_files: Dict[TransformError, List[str]],
):
    metadata = doc.metadata
    meta_str = f"{metadata['case']}/{metadata['file']}"
    # Skip documents that do not match the criteria
    if "PartialBatchCase" not in doc.page_content:
        skipped_files[TransformError.NotFoundError].append(meta_str)
        return
    try:
        result = await chain.ainvoke(doc.page_content)
        with open(output_path, "a") as f:
            f.write(json.dumps(result.dict()) + "\n")
    except Exception:
        skipped_files[TransformError.ParseError].append(meta_str)


async def transform(
    loader: BaseLoader,
    model: BaseChatModel,
    total: int | None = None,
    output_dir: str = "data",
) -> Dict[TransformError, List[str]]:
    output_path = f"{output_dir}/case_{model.model_name}.jsonl"
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    # Check if the file exists, and delete it if it does
    if os.path.exists(output_path):
        os.remove(output_path)

    skipped_files: Dict[TransformError, List[str]] = {
        TransformError.NotFoundError: [],
        TransformError.ParseError: [],
    }

    async for doc in tqdm_asyncio(loader.alazy_load(), desc="Processing", total=total):
        await process_document(doc, chain, output_path, skipped_files)

    return skipped_files
