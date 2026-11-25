
import json
import re
from typing import Any
import numpy as np

from tqdm import tqdm

from models import FunctionCalling, FunctionCallResult
from logit_masking import (
    initialize_vocab,
    mask_logits,
    get_allowed_tokens
)


def encode_text(sdk: Any, text: str) -> list[int]:
    """Convierte un texto en una lista de token IDs."""

    return sdk.encode(text).tolist()[0]


def get_logits(sdk: Any, input_ids: list[int]):
    """Obtiene los logits del siguiente token."""

    logits = sdk.get_logits_from_input_ids(input_ids)

    return logits


def get_function_name(text: str) -> str | None:
    """Extrae el nombre de la función cuando ya se ha generado."""

    match = re.search(r'"name"\s*:\s*"([^"]*)"', text)

    if match:
        return match.group(1)

    return None


def get_current_parameter_type(
    text: str,
    function: dict[str, Any]
) -> str | None:
    """
    Detecta qué parámetro se está generando y devuelve su tipo.
    """

    parameters_marker = '"parameters"'

    if parameters_marker not in text:
        return None

    parameters_text = text.split(parameters_marker, 1)[1]

    # Buscamos el último nombre de parámetro seguido de dos puntos.
    matches = list(
        re.finditer(r'"([^"]+)"\s*:\s*', parameters_text)
    )

    if not matches:
        return None

    last_match = matches[-1]
    parameter_name = last_match.group(1)

    # Comprobamos que el valor de ese parámetro todavía no ha terminado.
    value_text = parameters_text[last_match.end():]

    if "," in value_text or "}" in value_text:
        return None

    parameter_definition = function["parameters"].get(parameter_name)

    if parameter_definition is None:
        return None

    return parameter_definition["type"]


def is_valid_json(text: str) -> bool:
    """Comprueba si el texto contiene un objeto JSON completo."""

    try:
        result = json.loads(text)
        return isinstance(result, dict)

    except json.JSONDecodeError:
        return False


def generate_json(
    prompt: str,
    functions_str: str,
    sdk: Any,
    vocab: dict[str, Any],
    max_new_tokens: int = 200
) -> dict[str, Any]:
    """
    Genera un objeto JSON token a token aplicando
    restricciones según el estado de generación.
    """

    full_prompt = (
        "You must generate exactly one JSON object.\n"
        "Do not include explanations or Markdown.\n"
        "Use this exact structure:\n"
        '{"name": "function_name", "parameters": {}}\n\n'
        "Choose one of the available functions and provide "
        "its required parameters.\n\n"
        f"Available functions:\n{functions_str}\n\n"
        f"User request: {prompt}\n\n"
        "JSON:"
    )

    input_ids = encode_text(sdk, full_prompt)
    generated_ids = []

    # Relacionamos las definiciones de las funciones con sus nombres.
    function_definitions = json.loads(functions_str)

    definitions_by_name = {
        item["name"]: item
        for item in function_definitions
    }

    for _ in range(max_new_tokens):

        # 1. Obtenemos los logits del siguiente token.
        current_ids = input_ids + generated_ids
        logits = get_logits(sdk, current_ids)

        # 2. Decodificamos todo lo generado hasta ahora.
        generated_text = sdk.decode(generated_ids)

        # 3. Detectamos la función seleccionada.
        function_name = get_function_name(generated_text)

        function = definitions_by_name.get(function_name)

        # 4. Detectamos el tipo del parámetro actual.
        parameter_type = None

        if function is not None:
            parameter_type = get_current_parameter_type(
                generated_text,
                function
            )

        # 5. Seleccionamos la categoría de tokens permitidos.
        allowed_ids = get_allowed_tokens(
            generated_text,
            parameter_type,
            vocab
        )

        if len(generated_ids) < 20:
            print("\n--- DEBUG ---")
            print("Texto generado:", repr(generated_text))
            print("Función detectada:", function_name)
            print("Tipo de parámetro:", parameter_type)
            print("Tokens permitidos:", len(allowed_ids))

        # 6. Aplicamos la máscara a los logits.
        restricted_logits = mask_logits(logits, allowed_ids)

        # 7. Seleccionamos el token permitido con mayor logit.
        next_token = int(np.argmax(restricted_logits))

        generated_ids.append(next_token)

        # 8. Comprobamos si ya tenemos un JSON completo.
        generated_text = sdk.decode(generated_ids)

        if is_valid_json(generated_text):
            return json.loads(generated_text)

    raise RuntimeError(
        "The model did not generate a complete JSON object "
        "within the maximum number of tokens."
    )


def validate_result(
    result: dict[str, Any],
    function_definitions: dict[str, dict[str, Any]]
) -> None:
    """Valida el nombre de función y sus parámetros."""

    if set(result.keys()) != {"name", "parameters"}:
        raise ValueError("The generated JSON has an invalid structure.")

    function_name = result["name"]

    if function_name not in function_definitions:
        raise ValueError(
            f"Unknown function: {function_name}"
        )

    function = function_definitions[function_name]
    parameters = result["parameters"]
    schema = function["parameters"]

    if not isinstance(parameters, dict):
        raise ValueError("Parameters must be a JSON object.")

    if set(parameters.keys()) != set(schema.keys()):
        raise ValueError(
            f"Parameter names do not match function '{function_name}'."
        )

    for name, definition in schema.items():

        expected_type = definition["type"]
        value = parameters[name]

        if expected_type == "string":
            valid = isinstance(value, str)

        elif expected_type in ("integer", "int"):
            valid = isinstance(value, int) and not isinstance(value, bool)

        elif expected_type in ("float", "number"):
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )

        elif expected_type == "boolean":
            valid = isinstance(value, bool)

        else:
            raise ValueError(
                f"Unsupported parameter type: {expected_type}"
            )

        if not valid:
            raise ValueError(
                f"Invalid type for parameter '{name}'."
            )


def make_constrained_generation(
    natural_text: list[FunctionCalling],
    functions_str: str,
    nombres: list[str],
    sdk: Any
) -> list[dict[str, Any]]:
    """Genera y valida las llamadas a funciones."""

    function_definitions_list = json.loads(functions_str)

    definitions_by_name = {
        item["name"]: item
        for item in function_definitions_list
    }

    if set(nombres) != set(definitions_by_name):
        raise ValueError(
            "Function names do not match their definitions."
        )

    # Inicializamos el vocabulario una sola vez.
    first_prompt = "Initialize vocabulary"
    initial_ids = encode_text(sdk, first_prompt)
    initial_logits = get_logits(sdk, initial_ids)
    vocab_size = len(initial_logits)

    vocab = initialize_vocab(sdk, vocab_size)

    results = []

    for item in tqdm(natural_text, desc="Generating function calls"):

        result = generate_json(
            item.prompt,
            functions_str,
            sdk,
            vocab
        )

        validate_result(result, definitions_by_name)

        function_result = FunctionCallResult(
            prompt=item.prompt,
            name=result["name"],
            parameters=result["parameters"]
        )

        results.append(function_result.model_dump())

    return results
