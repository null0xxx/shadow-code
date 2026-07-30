PYTHON := .venv/bin/python
VERIFY_TARGETS := test-focused test lint format-check mypy bandit coverage

.PHONY: setup venv dev-deps editable test-focused test lint format-check mypy bandit coverage verify

setup: editable

venv:
	python3 -m venv .venv

dev-deps: venv
	$(PYTHON) -m pip install --requirement requirements-dev.lock

editable: dev-deps
	$(PYTHON) -m pip install --no-deps --editable .

test-focused:
	$(PYTHON) -m pytest tests/test_parser.py tests/test_tools.py tests/test_db.py -q

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

mypy:
	$(PYTHON) -m mypy shadow_code/

bandit:
	$(PYTHON) -m bandit -c pyproject.toml -r shadow_code/

coverage:
	$(PYTHON) -m pytest --cov=shadow_code --cov-report=term-missing

verify:
	@status=0; for target in $(VERIFY_TARGETS); do \
		$(MAKE) --no-print-directory $$target || status=1; \
	done; exit $$status
