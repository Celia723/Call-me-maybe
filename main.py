import argparse
import sys
import json
import models


def load_funtions(path: str) -> list:
    try:
        with open(path) as f:
            data = json.load(f)

    except FileNotFoundError and ValueError as e:
        print ("Error: {e}")





def main() -> None:

    # 1. argparse captura las rutas de la terminal (o pone las por defecto)
    parser = argparse.ArgumentParser(description="Function calling with LLMs and constrained decoding.")

    parser.add_argument(
        "--functions_definition"

        type=str
        default="data/input/functions_definition.json"
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
        functions = load_functions(args.funtions_definition)

    except Exception as e: 
        print(f"Error: {e}")
        sys.exit(1)