import numpy as np
from typing import Any


def get_state(translation_tokens: str) -> list:
    """Determina dinámicamente qué token IDs están permitidos según

    el estado sintáctico del texto generado hasta el momento.
    """
    # 2. Contamos comillas dobles para saber si estamos DENTRO o FUERA
    num_comillas = translation_tokens.count('"')

    # Si ve que no hay comillas abiertas
    if num_comillas % 2 == 0:
        #   Devuelve 0: la mascara con tokens de estructura
        return 0
    else:
        #   Devuelve 1: las mascara con todos los tokens MENOS los de estructura
        return 1


def get_structure_ids(sdk: Any) -> list[int]:
    """Calcula los IDs estructurales una sola vez al inicio."""
    return [
        sdk.encode("{")[0],
        sdk.encode("}")[0],
        sdk.encode("[")[0],
        sdk.encode("]")[0],
        sdk.encode('"')[0],
        sdk.encode(':')[0],
        sdk.encode(',')[0],
        sdk.encode('.')[0],
        sdk.encode('prompt')[0],
        sdk.encode('name')[0],
        sdk.encode('parameter')[0]
    ]


def apply_mascara(
    sdk: Any,
    logits: list[float],
    translation_tokens: list[float],
    structure_ids: list[int]
) -> np.ndarray:
    """Aplica una máscara de -infinito a todos los logits del vocabulario

    EXCEPTO a los índices que están en ids_permitidos, usando NumPy.
    """
    # 1. Convertimos la lista de logits a un array de NumPy
    logits_arr = np.array(logits, dtype=float)


    #   dependiendo del estado creamos una u otra mascara:
    if get_state(translation_tokens):
        # Impares (dentro de string): Blacklist (copiamos y prohibimos estructura)
        logits_mascarados = logits_arr.copy()
        logits_mascarados[structure_ids] = -np.inf
    else:
        # 2. Creamos un array lleno de -infinito del mismo tamaño
        logits_mascarados = np.full_like(logits_arr, -np.inf)
        # solo mnetemos los logits de estructura
        logits_mascarados[structure_ids] = logits_arr[structure_ids]

    return logits_mascarados
