.DEFAULT_GOAL := install

ifneq ($(wildcard /goinfre/$(USER)),)
SGOINFRE ?= /goinfre/$(USER)
else
SGOINFRE ?= $(CURDIR)/.cache
endif

VENV_DIR ?= $(SGOINFRE)/venvs/call-me-maybe
PYTHON ?= python3

export PATH := $(HOME)/.local/bin:$(PATH)
export UV_CACHE_DIR ?= $(SGOINFRE)/uv-cache
export MYPY_CACHE_DIR ?= $(SGOINFRE)/mypy-cache
export UV_LINK_MODE := copy
export UV_PROJECT_ENVIRONMENT := $(VENV_DIR)
export HF_HOME ?= $(SGOINFRE)/huggingface_cache
export HF_HUB_CACHE ?= $(HF_HOME)/hub
export HF_DATASETS_CACHE ?= $(HF_HOME)/datasets
export HF_HUB_VERBOSITY := error
export TMPDIR := $(SGOINFRE)/tmp

.PHONY: install run debug clean lint lint-strict prepare

prepare:
	mkdir -p "$(UV_CACHE_DIR)" "$(HF_HUB_CACHE)" "$(HF_DATASETS_CACHE)" \
		"$(TMPDIR)" "$(dir $(VENV_DIR))" "$(MYPY_CACHE_DIR)"

install: prepare
	uv sync --extra dev --python "$(PYTHON)"

run: prepare
	uv run --python "$(PYTHON)" python -u main.py \
		--functions_definition "data/input/functions_definition.json" \
		--input "data/input/function_calling_tests.json" \
		--output "data/output/function_calling_results.json"

debug: prepare
	uv run --python "$(PYTHON)" python -m pdb main.py \
		--functions_definition