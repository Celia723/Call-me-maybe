import argparse
import sys
import json
from llm_sdk import Small_LLM_Model
from parse import load_functions, load_natural_text
from result_tokens import make_all_tokens


def main() -> None:

    # 1. argparse captura las rutas de la terminal (o pone las por defecto)
    parser = argparse.ArgumentParser(
        description="Function calling with LLMs and constrained decoding.")

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

    #   creamos el objeto SDK y le pasamos el texto plano y el dict
    sdk = Small_LLM_Model()

    # 1. Convertimos la lista de funciones Pydantic a una
    # lista de diccionarios puros
    dict_functions = [f.model_dump() for f in functions]

    # 2. CONVERTIMOS LOS DICCIONARIOS A STRING:
    # json.dumps() coge la lista de diccionarios y la
    # convierte en un gran texto con formato JSON
    functions_str = json.dumps(dict_functions, indent=2)

    #   3. Creamos l srespuesta a partir de los tokens
    result = make_all_tokens(natural_text, functions_str, sdk)

    #   lo escribimos en la ruta
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"✅ ¡Todo completado con éxito! Guardado en: {args.output}")


if __name__ == "__main__":
    main()
