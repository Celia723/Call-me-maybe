VENV = .venv

.PHONY: install run debug clean lint lint-strict

install:
	@export UV_CACHE_DIR=/goinfre/$USER/uv-cache
	@export HF_HOME=/goinfre/$USER/hf
	@command -v uv >/dev/null 2>&1 || { \
		echo "❌ uv no está instalado ni en el PATH."; \
		echo "   Instálalo primero (mira el README)."; \
		exit 1; \
	}
	@echo "🛠️ Creando entorno virtual e instalando dependencias..."
	uv sync --extra dev

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -not -path "./$(VENV)/*" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./$(VENV)/*" -delete

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict