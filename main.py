import argparse
import sys
import json
import models
from llm_sdk/__init__ import Small_LLM_Model
from logits import apply_mascara
from parse import load_functions, load_natural_text


def main() -> None:

    # 1. argparse captura las rutas de la terminal (o pone las por defecto)
    parser = argparse.ArgumentParser(description="Function calling with LLMs and constrained decoding.")

    parser.add_argument(
        "--functions_definition",

        type=str,
        default="data/input/functions_definition.json",
        help="Ruta al archivo JSON con los prompts de prueba."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Ruta al archivo JSON con los prompts de prueba."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Ruta donde se guardará el JSON con los resultados."
    )

    # 2. Capturamos los argumentos que haya escrito el usuario en la terminal
    args = parser.parse_args()

    print(f"📂 Cargando funciones desde: {args.functions_definition}")
    print(f"📥 Cargando tests desde: {args.input}")
    print(f"📤 Guardando resultados en: {args.output}")


    # 3. Esas rutas se las pasamos a nuestras funciones que usan Pydantic
    try:
        functions = load_functions(args.functions_definition)
        natural_text = load_natural_text(args.input)

        print("todo leido y guardado")
        print("FUNCIONES: ")
        print(functions)
        print("LENGUAJE NATURAL")
        print(natural_text)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


    #creamos el objeto SDK y le pasamos el texto plano y el dict
    sdk = Small_LLM_Model()

    # 1. Convertimos la lista de funciones Pydantic a una lista de diccionarios puros
    dict_functions = [f.model_dump() for f in functions]

    # 2. CONVERTIMOS LOS DICCIONARIOS A STRING: 
    # json.dumps() coge la lista de diccionarios y la convierte en un gran texto con formato JSON
    functions_str = json.dumps(dict_functions, indent=2)

    #sacamos del diccionario los ids de los qsi valen en general
    id_open_key = sdk.tokenizer.encode("{", add_special_tokens=False)[0]
    id_closed_key = sdk.tokenizer.encode("}", add_special_tokens=False)[0]
    id_open_corchete = sdk.tokenizer.encode("[", add_special_tokens=False)[0]
    id_closed_corchete = sdk.tokenizer.encode("]", add_special_tokens=False)[0]
    id_comilla = sdk.tokenizer.encode('"', add_special_tokens=False)[0]
    id_dos_puntos = sdk.tokenizer.encode(':', add_special_tokens=False)[0]
    id_coma = sdk.tokenizer.encode(',', add_special_tokens=False)[0]
    id_punto = sdk.tokenizer.encode('.', add_special_tokens=False)[0]
    id_prompt = sdk.tokenizer.encode('prompt', add_special_tokens=False)[0]
    id_name = sdk.tokenizer.encode('name', add_special_tokens=False)[0]
    id_parameter = sdk.tokenizer.encode('parameter', add_special_tokens=False)[0]

    allowed_ids = [
    id_open_key, id_closed_key, id_open_corchete, id_closed_corchete, id_comilla, id_coma, id_dos_puntos, id_punto, id_prompt, id_name, id_parameter]


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
            restricted_logits = apply_mascara(logits, allowed_ids)

            # Seleccionamos el token más alto dentro de los permitidos por la máscara
            next_token_id = np.argmax(restricted_logits)

        # 4. Actualizamos el tensor y la lista de tokens generados
        input_tensor = append_token(input_tensor, next_token_id)  # O según acepte tu SDK
        choosen_tokens.append(next_token_id)

        # 5. Condición de parada (EOS token)
        if next_token_id == sdk.tokenizer.eos_token_id:
            break


if __name__ == "__main__":
    main()