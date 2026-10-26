import argparse
import sys
import json
import models
from llm_sdk/__init__ import Small_LLM_Model

def load_functions(path: str) -> list:

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Recorremos la lista de diccionarios y creamos los objetos Pydantic
        functions_list = [models.FunctionDefinition(**item) for item in data]

        return functions_list


def load_natural_text(path: str) -> list:

        with open(path, "r", encoding="utf-8") as f:
            data= json.load(f)
        
        calling_list = [models.FunctionCalling(**item) for item in data]

        return calling_list



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

    for text in natural_text:
        prompt_text = text.prompt

        print(f"\n--- Procesando prompt: {prompt_text} ---")

        #ahora q lo tenemos se lo pasamos al sdk JUNTO
        full_prompt = f"Funciones:\n{functions_str}\n\nPregunta: {prompt_text}"
        input_tensor = sdk.encode(full_prompt)
      
        #calculamos el siguiente loggit
        logits = sdk.get_logits_from_input_ids(input_tensor)




if __name__ == "__main__":
    main()