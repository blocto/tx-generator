import os
import json
import time
from enum import Enum
from tqdm.asyncio import tqdm_asyncio
from langchain_core.document_loaders import BaseLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from utils.model_selector import get_chat_model


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


class TransformError(Enum):
    NotFoundError = "Case Not Found"
    ParseError = "Parse Error"


async def transform(
    loader: BaseLoader,
    model: BaseChatModel,
    total: int | None = None,
    output_dir: str = "data",
) -> dict[TransformError, list[str]]:
    output_path = f"{output_dir}/case_{model.model_name}.jsonl"

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

    skipped_files: dict[str, list[str]] = {}  # skipped_reason : [file_name]
    duration_list = []

    async for doc in tqdm_asyncio(
        loader.alazy_load(), desc="Processing Documents", unit="doc", total=total
    ):
        metadata = doc.metadata
        meta_str = f"{metadata['case']}/{metadata['file']}"
        # Skip documents that do not match the criteria
        if "PartialBatchCase" not in doc.page_content:
            skipped_files[TransformError.NotFoundError] = meta_str
            continue

        try:
            start_time = time.time()

            result = await chain.ainvoke(doc.page_content)

            end_time = time.time()

            duration_list.append(end_time - start_time)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "a") as f:
                f.write(json.dumps(result.dict()))
                f.write("\n")
        except Exception as e:
            skipped_files[TransformError.ParseError] = meta_str
    return skipped_files
