SHELL := /bin/bash
ENV := $(PWD)/.env
# Include environment variables from .env file, if it exists
# https://dev.to/serhatteker/get-environment-variables-from-a-file-into-makefile-2m5l
-include $(ENV)
# Export the UV_INDEX_FOREST_USERNAME variable to subprocesses
export UV_INDEX_FOREST_USERNAME

.PHONY: pin-python
pin-python: ## Pins Python version to 3.9 for Butterfly SDK compatibility
	uv python pin 3.9  # Need this for Butterfly SDK

.PHONY: install
install: pin-python
	@if ! command -v uv &> /dev/null; then \
		echo "uv not found. Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@uv_version=$$(uv --version | awk '{print $$2}'); \
	if [ "$$(printf '%s\n' "0.4.24" "$$uv_version" | sort -V | head -n1)" != "0.4.24" ]; then \
		echo "Updating uv to latest version..."; \
		uv self update; \
	fi
	@uv sync



.PHONY: lint
lint: ## Runs code formatting, linting, and spell checking
	@echo "🚀 Formatting code: Running ruff"
	uv run --no-sync ruff format src tests
	@echo "🚀 Checking code: Running ruff"
	uv run --no-sync ruff check --fix src tests
	@echo "Checking import-contracts"
	uv run --no-sync lint-imports
	# @echo "🚀 Static type checking: Running pyright"
	# uv run --no-sync pyright
	@echo "🚀 Spell checking"
	uv run --no-sync codespell src --ignore-words=resources/dictionary.txt

.PHONY: lint-check
lint-check: token-exists ## Checks code formatting without making changes
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv sync --locked
	@echo "🚀 Formatting code: Running ruff"
	uv run --no-sync ruff format --diff src tests
	@echo "🚀 Checking code: Running ruff"
	uv run --no-sync ruff check src tests
	@echo "Checking import-contracts"
	uv run --no-sync lint-imports
	# @echo "🚀 Static type checking: Running pyright"
	# uv run --no-sync pyright --warnings
	@echo "🚀 Spell checking"
	uv run --no-sync codespell src --ignore-words=resources/dictionary.txt

.PHONY: test
test: ## Runs Python tests excluding integration tests
	@echo "🚀 Running tests"
	uv run --no-sync pytest tests --ignore tests/integration