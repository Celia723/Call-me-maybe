import numpy as np
from tqdm import tqdm
from typing import Any
from models import FunctionCalling
from logit_masking import apply_mascara, get_structure_ids


def make_all_tokens(
    natural_text: list[FunctionCalling],
    functions_str: str,
    sdk: Any
) -> list[str]:
    """Generate one constrained answer per prompt."""
    result: list[str] = []
    structure_ids = get_structure_ids(sdk)

    for text in tqdm(natural_text, desc="🚀 Creating the solution"):
        prompt_text = text.prompt
        full_prompt = (
            "Pick the function that matches the request and extract its arguments. "
            "Do not execute it.\n"
            f"Available functions:\n{functions_str}\n\n"
            "Example:\n"
            'Request: Greet john\n'
            'Answer: {"name":"fn_greet","parameters":{"name":"john"}}\n\n'
            f"Request: {prompt_text}\n"
            "Answer: "
        )
        input_ids: list[int] = sdk.encode(full_prompt).tolist()[0]
        if isinstance(input_ids, list) and isinstance(input_ids[0], list):
            input_ids = input_ids[0]

        # 2. PRE-FILLING: Forzamos que la respuesta empiece ya con el JSON abierto
        prefix_str = '{"name": "'
        prefix_ids = sdk.encode(prefix_str).tolist()
        if isinstance(prefix_ids, list) and isinstance(prefix_ids[0], list):
            prefix_ids = prefix_ids[0]
        choosen_tokens: list[int] = list(prefix_ids)
        input_ids.extend(prefix_ids)

        for _ in range(45):
            logits = sdk.get_logits_from_input_ids(input_ids)
            translation_tokens = sdk.decode(choosen_tokens)

            if (
                translation_tokens.endswith('"name": "') or
                translation_tokens.endswith('"description": "') or
                translation_tokens.endswith('"parameters": "')
            ):
                next_token_id = int(np.argmax(logits))
            else:
                restricted_logits = apply_mascara(
                    sdk, logits, translation_tokens, structure_ids)
                next_token_id = int(np.argmax(restricted_logits))

            input_ids.append(next_token_id)
            choosen_tokens.append(next_token_id)

            if next_token_id in (151643, 151645):
                break

        result.append(sdk.decode(choosen_tokens))

    return result
