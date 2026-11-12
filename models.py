from pydantic import BaseModel
from typing import Any


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]


class FunctionCalling(BaseModel):
    prompt: str
