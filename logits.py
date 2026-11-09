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


def get_allowed_ids(translation_tokens: str, tokenizer) -> list:

    """
    Determina dinámicamente qué token IDs están permitidos según 
    el estado sintáctico del texto generado hasta el momento.
    """
    id_open_key = tokenizer.encode("{", add_special_tokens=False)[0]
    id_closed_key = tokenizer.encode("}", add_special_tokens=False)[0]
    id_open_corchete = tokenizer.encode("[", add_special_tokens=False)[0]
    id_closed_corchete = tokenizer.encode("]", add_special_tokens=False)[0]
    id_comilla = tokenizer.encode('"', add_special_tokens=False)[0]
    id_dos_puntos = tokenizer.encode(':', add_special_tokens=False)[0]
    id_coma = tokenizer.encode(',', add_special_tokens=False)[0]
    id_punto = tokenizer.encode('.', add_special_tokens=False)[0]
    id_prompt = tokenizer.encode('prompt', add_special_tokens=False)[0]
    id_name = tokenizer.encode('name', add_special_tokens=False)[0]
    id_parameter = tokenizer.encode('parameter', add_special_tokens=False)[0]

    allowed_ids = [
    id_open_key, id_closed_key, id_open_corchete, id_closed_corchete, id_comilla, id_coma, id_dos_puntos, id_punto, id_prompt, id_name, id_parameter]


    # 2. Contamos comillas dobles para saber si estamos DENTRO de un valor o FUERA
    num_comillas = translation_tokens.count('"')

    # si ve q no hay comillas abiertas, 
    if num_comillas % 2 == 0:
        allowed_ids = [id_closed_corchete, id_closed_key, id_coma, id_comilla]
        return allowed_ids
    else:
        #si falta texto por esribir, cogemos el diccionario todo menos los ids de las estructuras
        # del json. 
        
        #CRIBAMOS LAS ESTRUCTRUCTURAS DEL JSON PARA SQUEDARNOS CON SOLO TEXTO
        allowed_ids = []
        
        # recorremos el diccionario y nos quedamos con lo q no es ESTRUCTURA
        for token, token_id in tokenizer.get_vocab().items():
            if not any(c in token for c in ['{', '}', ':', '\n']):
                allowed_ids.append(token_id)
        return allowed_ids
