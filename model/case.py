from langchain_core.pydantic_v1 import BaseModel, Field
from .tx import Tx


class Case(BaseModel):
    case_id: str = Field(description="Unique identifier for the case.")
    description: str = Field(description="A brief description of the case.")
    total_steps: int = Field(description="The total transaction count of the case.")
    steps: list[Tx] = Field(description="Each step represents a transaction.")


class CaseOutput(BaseModel):
    cases: list[Case] = Field(
        description="List of cases. This could be empty if no cases are found."
    )
