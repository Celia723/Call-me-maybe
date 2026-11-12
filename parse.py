import models
import json


def load_functions(path: str) -> list:

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Recorremos la lista de diccionarios y creamos los objetos Pydantic
    functions_list = [models.FunctionDefinition(**item) for item in data]

    return functions_list


def load_natural_text(path: str) -> list:

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    calling_list = [models.FunctionCalling(**item) for item in data]

    return calling_list
