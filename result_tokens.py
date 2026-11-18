
import json
import re
from typing import Any

from tqdm import tqdm

from models import FunctionCalling, FunctionCallResult
from logit_masking import mask_logits, get_allowed_next_tokens


def encode_text(sdk: Any, text: str) -> list[int]:
    """Encode text and return a list of token IDs."""

    return sdk.encode(text).tolist()[0]


def get_stop_tokens(sdk: Any) -> set[int]:
    """Get the model's end-of-generation token IDs."""

    tokenizer = getattr(sdk, "_tokenizer", None)

    if tokenizer is None:
        return set()

    stop_tokens = set()

    eos_id = getattr(tokenizer, "eos_token_id", None)

    if isinstance(eos_id, int):
        stop_tokens.add(eos_id)

    return stop_tokens


def select_function(
    input_ids: list[int],
    function_names: list[str],
    sdk: Any
) -> tuple[str, list[int]]:
    """Select a function using token-level constraints."""

    candidates = [
        encode_text(sdk, name)
        for name in function_names
    ]

    if not candidates or any(not candidate for candidate in candidates):
        raise ValueError("Invalid function name candidates.")

    generated_ids = []
    current_ids = input_ids.copy()

    max_length = max(len(candidate) for candidate in candidates)

    for _ in range(max_length):
        allowed_ids = get_allowed_next_tokens(
            candidates,
            generated_ids
        )

        logits = sdk.get_logits_from_input_ids(current_ids)

        masked_logits = mask_logits(logits, allowed_ids)

        next_token = int(masked_logits.argmax())

        generated_ids.append(next_token)
        current_ids.append(next_token)

        for name, candidate in zip(function_names, candidates):
            if generated_ids == candidate:
                return name, current_ids

    raise ValueError("Could not select a valid function.")


def clean_string(value: str) -> str:
    """Remove formatting and surrounding quotation marks."""

    value = value.strip()

    value = re.sub(r"^```(?:text)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)

    if value.startswith("Value:"):
        value = value[len("Value:"):].strip()

    if len(value) >= 2:
        if (
            (value[0] == "'" and value[-1] == "'")
            or (value[0] == '"' and value[-1] == '"')
        ):
            value = value[1:-1]

    return value.strip()


def generate_value(
    prompt: str,
    sdk: Any,
    max_new_tokens: int = 64
) -> str:
    """Generate a parameter value with the language model."""

    input_ids = encode_text(sdk, prompt)
    generated_ids = []
    stop_tokens = get_stop_tokens(sdk)

    for _ in range(max_new_tokens):
        logits = sdk.get_logits_from_input_ids(input_ids)

        next_token = int(
            max(range(len(logits)), key=logits.__getitem__)
        )

        if next_token in stop_tokens:
            break

        input_ids.append(next_token)
        generated_ids.append(next_token)

        generated_text = sdk.decode(generated_ids)

        if "\n" in generated_text:
            break

    return sdk.decode(generated_ids).strip()


def build_parameter_prompt(
    request: str,
    function: dict[str, Any],
    parameter_name: str,
    parameter_type: str
) -> str:
    """Build instructions for extracting one parameter."""

    instructions = (
        "Extract one argument for a function call.\n"
        "Use the user's request as the source of truth.\n"
        "Return only the argument value, without explanations.\n"
        "Do not invent information.\n"
    )

    if parameter_name == "source_string":
        instructions += (
            "Copy the complete original text that the operation "
            "will modify. Remove only quotation marks used as "
            "delimiters around the text.\n"
        )

    elif parameter_name == "regex":
        instructions += (
            "Return a regular expression matching the requested "
            "pattern, not an example of the matched text.\n"
            "For all numbers, use \\d+.\n"
            "For all vowels, use [aeiouAEIOU].\n"
            "For an exact word, return that word as the pattern.\n"
        )

    elif parameter_name == "replacement":
        instructions += (
            "Return the exact replacement text requested by the user.\n"
            "If the user asks for asterisks, return one asterisk: *.\n"
            "Preserve the spelling and capitalization of replacement "
            "words.\n"
        )

    elif parameter_type == "string":
        instructions += (
            "Return the exact requested text. Do not include quotation "
            "marks used only to delimit the text in the request.\n"
        )

    elif parameter_type == "number":
        instructions += (
            "Return only the numeric value. Do not include units "
            "or explanations.\n"
        )

    return (
        instructions
        + f"Function: {function['name']}\n"
        + f"Function description: {function['description']}\n"
        + f"User request: {request}\n"
        + f"Parameter: {parameter_name}\n"
        + f"Required type: {parameter_type}\n"
        + "Value:"
    )


def convert_parameter(
    raw_value: str,
    parameter_type: str,
    parameter_name: str
) -> Any:
    """Convert and validate a generated parameter."""

    if parameter_type == "string":
        return clean_string(raw_value)

    if parameter_type == "number":
        value = raw_value.strip()

        match = re.search(
            r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?",
            value
        )

        if match is None:
            raise ValueError(
                f"Could not extract a number for '{parameter_name}'."
            )

        return json.loads(match.group())

    raise ValueError(
        f"Unsupported parameter type: {parameter_type}"
    )


def generate_parameters(
    request: str,
    function: dict[str, Any],
    sdk: Any
) -> dict[str, Any]:
    """Generate all parameters required by the selected function."""

    parameters = {}

    for name, schema in function["parameters"].items():
        parameter_type = schema["type"]

        prompt = build_parameter_prompt(
            request,
            function,
            name,
            parameter_type
        )

        raw_value = generate_value(prompt, sdk)

        parameters[name] = convert_parameter(
            raw_value,
            parameter_type,
            name
        )

    return parameters


def validate_parameters(
    parameters: dict[str, Any],
    function: dict[str, Any]
) -> None:
    """Validate parameter names and primitive types."""

    schema = function["parameters"]

    if set(parameters) != set(schema):
        raise ValueError(
            f"Parameter names do not match function '{function['name']}'."
        )

    for name, definition in schema.items():
        expected_type = definition["type"]
        value = parameters[name]

        if expected_type == "string":
            valid = isinstance(value, str)

        elif expected_type == "number":
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )

        else:
            raise ValueError(
                f"Unsupported type '{expected_type}' for '{name}'."
            )

        if not valid:
            raise ValueError(
                f"Invalid value type for parameter '{name}'."
            )


def make_constrained_generation(
    natural_text: list[FunctionCalling],
    functions_str: str,
    nombres: list[str],
    sdk: Any
) -> list[dict[str, Any]]:
    """Generate function calls and validate their parameters."""

    function_definitions = json.loads(functions_str)

    definitions_by_name = {
        function["name"]: function
        for function in function_definitions
    }

    if set(nombres) != set(definitions_by_name):
        raise ValueError(
            "Function names do not match the supplied definitions."
        )

    results = []

    for item in tqdm(natural_text, desc="Generating function calls"):
        prompt = (
            "Choose the function that best matches the request.\n"
            "Choose only one of the available function names.\n\n"
            f"Available functions:\n{functions_str}\n\n"
            f"User request: {item.prompt}\n"
            "Function name:"
        )

        input_ids = encode_text(sdk, prompt)

        selected_name, _ = select_function(
            input_ids,
            nombres,
            sdk
        )

        function = definitions_by_name[selected_name]

        parameters = generate_parameters(
            item.prompt,
            function,
            sdk
        )

        validate_parameters(parameters, function)

        result = FunctionCallResult(
            prompt=item.prompt,
            name=selected_name,
            parameters=parameters
        )

        results.append(result.model_dump())

    return results
