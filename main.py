import argparse
import sys
import json
import models


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
        functions = load_functions(args.funtions_definition)
        natural_text = load_natural_text(args.input)

        print("todo leido y guardado")
        print("FUNCIONES: ")
        print(functions)
        print("LENGUAJE NATURAL")
        print(natural_text)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()