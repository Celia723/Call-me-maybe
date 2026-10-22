from pydantic import BaseModel


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict


class FunctionCalling(BaseModel):
    prompt: str