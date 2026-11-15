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


def apply_mascara(
    sdk: Any,
    logits: list[float],
    translation_tokens: list[float],
) -> np.ndarray:
    """Aplica una máscara de -infinito a todos los logits del vocabulario

    EXCEPTO a los índices que están en ids_permitidos, usando NumPy.
    """
    id_open_key = sdk.encode("{")[0]
    id_closed_key = sdk.encode("}")[0]
    id_open_corchete = sdk.encode("[")[0]
    id_closed_corchete = sdk.encode("]")[0]
    id_comilla = sdk.encode('"')[0]
    id_dos_puntos = sdk.encode(':')[0]
    id_coma = sdk.encode(',')[0]
    id_punto = sdk.encode('.')[0]
    id_prompt = sdk.encode('prompt')[0]
    id_name = sdk.encode('name')[0]
    id_parameter = sdk.encode('parameter')[0]

    structure_ids = [
        id_open_key, id_closed_key, id_open_corchete,
        id_closed_corchete, id_comilla, id_coma,
        id_dos_puntos, id_punto, id_prompt,
        id_name, id_parameter
    ]
    
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
