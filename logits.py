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
