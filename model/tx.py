from langchain_core.pydantic_v1 import BaseModel, Field


class Tx(BaseModel):
    description: str = Field(description="A summarized description of the transaction.")
    to: str = Field(description="The receiving address of the transaction.")
    value: str = Field(description="The amount of native token to transfer.")
    function_name: str = Field(description="Function name of the transaction.")
    input_args: list[str] = Field(
        description="Input arguments of the function. If the value is not determined, use `{user_input}`."
    )
