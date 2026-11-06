import numpy as np

def apply_mascara(logits: list[float], ids_permitidos: list[int]) -> np.ndarray:
    """
    Aplica una máscara de -infinito a todos los logits del vocabulario
    EXCEPTO a los índices que están en ids_permitidos, usando NumPy.
    """
    # 1. Convertimos la lista de logits a un array de NumPy
    logits_arr = np.array(logits, dtype=float)
    
    # 2. Creamos un array lleno de -infinito del mismo tamaño
    logits_mascarados = np.full_like(logits_arr, -np.inf)
    
    # 3. Restauramos los logits originales solo en los IDs permitidos
    logits_mascarados[ids_permitidos] = logits_arr[ids_permitidos]
    
    return logits_mascarados


def get_allowed_ids(translation_tokens: str, tokenizer, valid_functions: list) -> list:
    # 1. State: Just opened a quote for a string value (e.g. after '"name": "')
    # Allowed: All alphanumeric tokens + spaces + the closing quote '"'
    
    # 2. State: Just closed a value with a quote (e.g. ends with '"')
    # Allowed: Syntax transition tokens -> comma (','), colon (':'), or closing brace ('}')
    
    # 3. State: Inside parameters object
    # Allowed: Parameter names defined in your Pydantic schemas
