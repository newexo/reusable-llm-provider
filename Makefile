# Package name, read from pyproject.toml so this Makefile is reusable across
# projects. This works because the distribution name matches the importable
# package directory -- keep them identical (underscores, not hyphens) when
# renaming. Any mismatch shows up immediately as 0% coverage.
PACKAGE := $(shell awk -F'"' '/^name = / {print $$2; exit}' pyproject.toml)

# Minimum coverage percentage required for tests to pass
COVERAGE_FAIL = 50

# Run the test suite
test:
	poetry run pytest

# Run functional tests (live calls to real LLM providers).
# Requires secrets/.env with appropriate API keys and a running Ollama server.
test-functional:
	poetry run pytest functional_tests

# Format the code using Ruff. Applies safe lint fixes (import sorting,
# pyupgrade rewrites) before formatting, so `make check` passes afterwards.
format:
	poetry run ruff check --fix .
	poetry run ruff format .

# Verify formatting without rewriting anything (this is what CI runs)
format-check:
	poetry run ruff format --check .

# Lint the code using Ruff (configured in pyproject.toml [tool.ruff])
lint:
	poetry run ruff check .

# Run all quality checks. Does not modify files; run `make format` to fix.
check: format-check lint test

# Run tests with coverage enforcement (terminal output only)
# Omit patterns are configured in pyproject.toml [tool.coverage.run].
coverage:
	poetry run coverage run --source=$(PACKAGE) -m pytest
	poetry run coverage report --fail-under=$(COVERAGE_FAIL)

# Run tests with coverage and produce an HTML report
coverage-html:
	poetry run coverage run --source=$(PACKAGE) -m pytest
	poetry run coverage report --fail-under=$(COVERAGE_FAIL)
	poetry run coverage html
	@echo "HTML coverage report generated at htmlcov/index.html"

# Verify every optional dependency is isolated behind a single module.
import-boundaries:
	poetry run python scripts/import_boundaries.py .

# Verify that a backend is usable if and only if its extra is installed.
# Runs here with everything installed, which only catches a vendor import that
# has moved back to module scope; CI runs it in environments built with one
# extra, and with none, where the rest of it can fail.
check-extras:
	poetry run python scripts/check_extras.py anthropic openai vertex ollama

# Verify imported packages are declared, and declared in the right group.
deps-check:
	poetry run deptry .

# Report unused code. ADVISORY: a library's public API is uncalled by
# construction, so read the output rather than trusting it. For a project with
# pre-existing dead code, baseline it once with
#   poetry run vulture --make-whitelist $(PACKAGE) > deadcode-whitelist.py
# commit that file, and pass it as an extra argument below; vulture then
# reports only newly dead code.
deadcode:
	poetry run vulture $(PACKAGE) scripts deadcode-whitelist.py

# Build a wheel, install it into a throwaway virtualenv and run the shipped
# tests against the INSTALLED package. The recipe changes directory first
# because from the repository root `import $(PACKAGE)` resolves to the source
# tree, which would silently test the wrong code.
#
# The wheel is installed BARE, with no extras, because that is the supported
# install: every backend is optional. The `needs_backends` tests are deselected
# -- they patch vendor symbols or construct providers, so they cannot run here
# by design. test_packaging.py covers what must hold with no backend at all.
#
# Installing `[all]` instead would work, but it downloads every vendor SDK on
# every CI job and grows with each backend added.
test-wheel:
	rm -rf dist .wheeltest
	poetry build -f wheel
	python -m venv .wheeltest
	./.wheeltest/bin/pip install --quiet dist/*.whl pytest
	cd "$$(mktemp -d)" && $(CURDIR)/.wheeltest/bin/python -m pytest --pyargs $(PACKAGE).tests -q -m "not needs_backends"
	rm -rf .wheeltest
	@echo "shipped tests pass against the installed wheel"

.PHONY: test test-functional format format-check lint check coverage \
        coverage-html import-boundaries check-extras deps-check deadcode \
        test-wheel
