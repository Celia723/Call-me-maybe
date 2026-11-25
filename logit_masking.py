
import numpy as np
from typing import Any


# Caracteres permitidos para cada categoría de tokens.
ALLOWED_CHARS = {
    "structural": set('{}[]":,0123456789.truefalsn- '),
    "number": set('0123456789.-eE,}] '),
    "boolean": set('truefals,}] ')
}


def initialize_vocab(sdk: Any, vocab_size: int) -> dict[str, np.ndarray]:
    """
    Recorre el vocabulario y clasifica los tokens en tres categorías:
    structural, number y boolean.
    """

    structural_tokens = []
    number_tokens = []
    boolean_tokens = []

    for token_id in range(vocab_size):
        text = sdk.decode([token_id])

        if not text:
            continue

        if all(char in ALLOWED_CHARS["structural"] for char in text):
            structural_tokens.append(token_id)

        if all(char in ALLOWED_CHARS["number"] for char in text):
            number_tokens.append(token_id)

        if all(char in ALLOWED_CHARS["boolean"] for char in text):
            boolean_tokens.append(token_id)

    return {
        "all": np.arange(vocab_size, dtype=np.int64),
        "structural": np.array(structural_tokens, dtype=np.int64),
        "number": np.array(number_tokens, dtype=np.int64),
        "boolean": np.array(boolean_tokens, dtype=np.int64)
    }


def mask_logits(
    logits: Any,
    allowed_token_ids: np.ndarray
) -> np.ndarray:
    """
    Conserva los logits de los tokens permitidos
    y asigna -infinito al resto.
    """

    scores = np.asarray(logits, dtype=np.float32).reshape(-1)

    masked_scores = np.full(scores.shape, -np.inf, dtype=np.float32)

    valid_ids = allowed_token_ids[
        (allowed_token_ids >= 0)
        & (allowed_token_ids < len(scores))
    ]

    if len(valid_ids) == 0:
        raise ValueError("No hay tokens permitidos para este estado.")

    masked_scores[valid_ids] = scores[valid_ids]

    return masked_scores


def inside_string(text: str) -> bool:
    """
    Comprueba si estamos dentro de una cadena JSON,
    teniendo en cuenta las comillas escapadas.
    """

    inside = False
    escaped = False

    for char in text:
        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == '"':
            inside = not inside

    return inside


def get_allowed_tokens(
    text: str,
    current_parameter_type: str | None,
    vocab: dict[str, np.ndarray]
) -> np.ndarray:
    """
    Selecciona la categoría de tokens permitidos según
    el contexto actual de generación.
    """

    # Dentro de una cadena se permite todo el vocabulario.
    if inside_string(text):
        return vocab["all"]

    # Si estamos generando un parámetro numérico.
    if current_parameter_type in ("integer", "float", "number"):
        return vocab["number"]

    # Si estamos generando un parámetro booleano.
    if current_parameter_type == "boolean":
        return vocab["boolean"]

    # En el resto de situaciones usamos structural.
    return vocab["structural"]
