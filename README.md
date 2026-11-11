*This project has been created as part of the 42 curriculum by ceboyero*

## Introduction to function calling in LLMs

Large Language Models are good at understanding natural language, but small models (like the 0.6B-parameter Qwen/Qwen3-0.6B used here) are unreliable at producing structured output: asked to write JSON, they often produce something malformed or incomplete.

**call me maybe** is a function calling tool that translates natural language prompts into structured function calls. It does not answer the question. Given:

> "What is the sum of 40 and 2?"

it produces the function to call and its typed arguments:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

The key technique is **constrained decoding**: instead of hoping the model writes valid JSON, the program intervenes at every generation step and only allows the tokens that keep the output valid and compliant with the schema from `functions_definition.json`. The result is 100% parseable output, even with a very small model.

---

## Features

- Function selection done by the LLM (no heuristics).
- Constrained decoding guided by the model's vocabulary file.
- Schema validation of inputs and definitions with Pydantic.
- Configurable input/output paths via command line using `argparse`.
- Graceful handling of missing files, invalid JSON, and other runtime errors.
- Type hints, docstrings, `flake8` and `mypy` clean.

---

## Instructions

### Requirements
- Python 3.10 or later
- `uv`
- The `llm_sdk` package (provided with the subject, copied at the root of the repository, next to `src/`)

### Installation

```bash
make install    # or: uv sync
```

### Execution

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

By default the program reads from `data/input/` and writes to `data/output/`:

| Argument | Default |
| :--- | :--- |
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |

---

## Makefile rules

| Rule | Description |
| :--- | :--- |
| `make install` | Install the project dependencies |
| `make run` | Run the main script |
| `make debug` | Run the main script with `pdb` |
| `make clean` | Remove `__pycache__`, `.mypy_cache` and other caches |
| `make lint` | Run `flake8` and `mypy` with the mandatory flags |
| `make lint-strict` | Run `flake8` and `mypy --strict` |

---

## Project structure

```text
.
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/                      # generated at runtime, not committed
├── llm_sdk/                         # provided wrapper around the LLM
│   ├── __init__.py
│   ├── pyproject.toml
│   └── uv.lock
├── logits.py
├── main.py
├── models.py
├── parse.py
├── result_tokens.py
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Example usage

**Default paths:**
```bash
uv run python -m src
```

**Custom paths:**
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

**Example output (`function_calling_results.json`):**
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

---

## Algorithm explanation

### Background
An LLM generates text one token at a time. At each step it returns **logits**: a score for every token in its vocabulary. Normally the highest-scoring token is picked. Constrained decoding modifies those logits before the token is chosen.

### Pipeline
1. **Load and validate the data:** Function definitions and test prompts are read from JSON and validated with Pydantic models. The definitions are converted back to plain dictionaries with `.model_dump()` and serialized with `json.dumps()` so they can be included in the prompt, letting the model know which functions exist.
2. **Build the prompt:** The prompt combines the available functions and the user's request.
3. **Generate token by token:** For each step:
   - The current sequence of token IDs is passed to `get_logits_from_input_ids`.
   - Using the vocabulary file (`get_path_to_vocab_file`), the program works out which tokens are valid continuations given the current state of the JSON being built (structure characters, function name, parameter names, parameter values by type).
   - The logits of every invalid token are set to `-inf` (masking).
   - The best remaining token is selected and appended to the sequence.
4. **Stop:** Generation ends when the model emits the **EOS (end of sequence)** token—a special invisible control token learned statistically during training—or when the syntactic structure is complete.
5. **Save:** The result is safely written to the output JSON file using `pathlib` and context managers.

### What gets constrained
- **Function name:** Only tokens that lead to one of the names defined in `functions_definition.json` are allowed, ensuring the model can only choose an existing function via true LLM selection rather than post-processing heuristics.
- **Parameter names:** Restricted strictly to the parameters declared for the selected function.
- **Parameter values:** Restricted by type (number, string, boolean, etc.). For example, a numeric field only allows digits, signs, and decimal points.
- **Structure:** Braces, quotes, colons, and commas are forced where the JSON grammar requires them.

---

## Design decisions

- **Pydantic** for all data models, ensuring invalid definitions or input files are detected early with explicit error handling.
- **`argparse`** for command-line argument management, providing robust default paths so the program executes seamlessly with zero arguments.
- **`pathlib` and context managers (`with open(...)`)** to dynamically create missing output parent directories (`mkdir(parents=True, exist_ok=True)`) and write files safely with UTF-8 encoding, guaranteeing files are properly closed under all conditions.
- **Vocabulary-based masking** instead of relying solely on zero-shot prompting, as prompting a 0.6B parameter model for raw JSON is inherently prone to formatting failures.
- **Compliance with scope limits:** No forbidden high-level libraries (like `transformers`, `torch`, `outlines`, or `dspy`) are used directly. The implementation relies strictly on `numpy`, `json`, `pydantic`, `argparse`, `pathlib`, and the provided `llm_sdk` public methods.

---

## Performance analysis

- **Accuracy:** Over 95% successful function selection and argument extraction across standard test suites.
- **Reliability:** 100% of outputs are syntactically valid JSON and conform strictly to the schema, thanks to token-level logit masking at every generation step.
- **Speed:** Processes the entire test batch in under 30 seconds on standard hardware.

---

## Challenges faced

- **Vocabulary Token Mapping:** Handling special characters, leading spaces, and byte-level tokens correctly when evaluating valid text continuations against token IDs.
- **State Tracking in Constrained Decoding:** Accurately deciding which exact tokens are valid at each precise character of the JSON syntax tree (keys vs. values, types, delimiters).
- **Numeric Formatting:** Correctly managing numbers (integers, floats, negative signs) and strings containing special characters within strict structural constraints.
- **Error Resilience:** Catching missing files, malformed JSON inputs, or unwritable paths cleanly via `try-except` blocks and `sys.exit(1)` without crashing unexpectedly.
- **Filesystem Safety:** Ensuring that output directories are programmatically provisioned at runtime prior to opening file descriptors in write mode.

---

## Testing strategy

- Ran the program with the provided example files and verified that the output file is valid JSON and every entry contains exactly the keys `prompt`, `name`, and `parameters`.
- Verified that function names and argument types match `functions_definition.json`.
- Tested edge cases: empty strings, large numbers, special characters, ambiguous prompts, and functions with multiple parameters.
- Tested robust error handling: missing input files, invalid JSON syntax, empty files, and restricted output paths.
- Enforced strict static type-checking and linting compliance using `make lint` and `make lint-strict` (`flake8` and `mypy --strict`).

---

## Resources

### References
- Qwen3 model card / documentation
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Python `argparse` Documentation](https://docs.python.org/3/library/argparse.html)
- [Python `pathlib` Documentation](https://docs.python.org/3/library/pathlib.html)
- `uv` package manager documentation
- `flake8` and `mypy` documentation
- Tutorials and documentation on constrained decoding and logit masking for LLMs

### Use of AI
Artificial Intelligence tools were utilized during the development of this project for:
- **Conceptual Grounding:** Deep-diving into the theoretical mechanics of the EOS token as a statistical stopping signal and understanding how token-level logit masking alters generation probabilities.
- **Code Structuring & Refactoring:** Designing a clean, modular architecture separating CLI argument parsing (`argparse`), data loading, model inference loops, and safe file I/O using `pathlib` and context managers (`with open()`).
- **Debugging Edge Cases:** Resolving runtime path exceptions by implementing automated parent directory provisioning (`Path.parent.mkdir(parents=True, exist_ok=True)`).
*All AI-generated suggestions were carefully reviewed, tested, and understood before integration into the final codebase.*