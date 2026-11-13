from logits import apply_mascara, get_allowed_ids
import numpy as np
import tqdm
from typing import Any
from models import FunctionCalling


def make_all_tokens(
    natural_text: list[FunctionCalling],
    functions_str: str,
    sdk: Any
) -> str:
    """Create the result choosing the right tokens for each

    natural language with all the functions in a str format.
    """
    result = ""

    for text in tqdm(natural_text, desc="🚀 Creating the solution"):
        prompt_text = text.prompt

        print(f"\n--- Procesando prompt: {prompt_text} ---")

        # Ahora que lo tenemos se lo pasamos al sdk JUNTO
        full_prompt = (
            f"Funciones:\n{functions_str}\n\nPregunta: {prompt_text}"
        )
        input_tensor = (sdk.encode(full_prompt)).tolist()

        # YA TENEMOS EL PROMPT ENTERO, AHORA CALCULAMOS LOS TOKENS
        choosen_tokens: list[int] = []
        for step in range(200):
            # 1. Sacamos las notas (logits) del paso actual
            logits = sdk.get_logits_from_input_ids(input_tensor)

            # 2. Decodificamos lo generado hasta ahora
            translation_tokens = sdk.decode(choosen_tokens)

            # 3. COMPROBACIÓN DE ESTADO:
            if (
                translation_tokens.endswith('"name": "') or
                translation_tokens.endswith('"description": "') or
                translation_tokens.endswith('"parameters": "')
            ):
                # ESTADO LIBRE: La IA decide sin restricciones
                next_token_id = int(np.argmax(logits))
            else:
                # ESTADO RESTRINGIDO: Forzamos la sintaxis con máscara
                allowed_ids = get_allowed_ids(
                    translation_tokens,
                    sdk.tokenizer
                )
                restricted_logits = apply_mascara(logits, allowed_ids)

                # Seleccionamos el token más alto permitido
                next_token_id = int(np.argmax(restricted_logits))

            # 4. Actualizamos tensor y lista de tokens generados
            input_tensor.append(next_token_id)
            choosen_tokens.append(next_token_id)

            # 5. Condición de parada (EOS token)
            if next_token_id == sdk.tokenizer.eos_token_id:
                result += str(choosen_tokens) + "\n"
                break

    return result
    
