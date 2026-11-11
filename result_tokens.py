



def make_all_tokens(natural_text: str,functions_str: str, sdk ):
    """
        Create the result chooosing de right tokens for each natural leanguaje with all the functions in a str format
    """
    
    for text in natural_text:
        prompt_text = text.prompt

        print(f"\n--- Procesando prompt: {prompt_text} ---")

        #ahora q lo tenemos se lo pasamos al sdk JUNTO
        full_prompt = f"Funciones:\n{functions_str}\n\nPregunta: {prompt_text}"
        input_tensor = sdk.encode(full_prompt)
        
      #YA TENEMOS EL PROMT ENTERO , AHORA CALCULAMOS TODOS LOS TOKENS SIGUIIENTES
        choosen_tokens = []
        for step in range(200):
            # 1. Sacamos las notas (logits) del paso actual
        logits = sdk.get_logits_from_input_ids(input_tensor)

        # 2. Decodificamos lo generado hasta ahora para analizar el contexto
        # Nota: pasa los tokens acumulados (ej: sdk.decode(choosen_tokens))
        translation_tokens = sdk.decode(choosen_tokens)

        # 3. COMPROBACIÓN DE ESTADO:
        # ¿Estamos a punto de escribir un valor libre? (Atención a las comillas finales: '"name": "')
        if (translation_tokens.endswith('"name": "') or 
            translation_tokens.endswith('"description": "') or 
            translation_tokens.endswith('"parameters": "')):
            
            # ESTADO LIBRE: La IA decide sin restricciones
            next_token_id = np.argmax(logits)

        else:
            # ESTADO RESTRINGIDO: Forzamos la sintaxis con la máscara
            # (Aquí recalculas allowed_ids según el estado exacto si hace falta)
            allowed_ids = get_allowed_ids(translation_tokens, sdk.tokenizer)
            restricted_logits = apply_mascara(logits, allowed_ids)

            # Seleccionamos el token más alto dentro de los permitidos por la máscara
            next_token_id = np.argmax(restricted_logits)

        # 4. Actualizamos el tensor y la lista de tokens generados
        input_tensor = append_token(input_tensor, next_token_id)  # O según acepte tu SDK
        choosen_tokens.append(next_token_id)

        # 5. Condición de parada (EOS token)
        if next_token_id == sdk.tokenizer.eos_token_id:
            result += f"choosen_tokens" + "\n"
            break
    
    return result