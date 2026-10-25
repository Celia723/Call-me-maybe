VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
MAIN = main.py

.PHONY: install run debug clean lint lint-strict

install:
	@echo "🛠️ Creando entorno virtual..."
	python3 -m venv $(VENV)
	@echo "📦 Instalando dependencias..."
	$(PIP) install --upgrade pip
	$(PIP) install pydantic flake8 mypy

run:
	$(PYTHON) $(MAIN)

debug:
	$(PYTHON) -m pdb $(MAIN)

clean:
	find . -type d -name "__pycache__" -not -path "./$(VENV)/*" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./$(VENV)/*" -delete

lint:
	$(PYTHON) -m flake8 . --exclude $(VENV)
	$(PYTHON) -m mypy . --exclude $(VENV) --warn-return-any \
		--warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(PYTHON) -m flake8 . --exclude $(VENV)
	$(PYTHON) -m mypy . --exclude $(VENV) --strict