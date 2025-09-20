.PHONY: format format-check lint install-hooks help

# Format all Python files with Black
format:
	black --line-length 88 --target-version py38 .

# Check if files are formatted correctly (useful for CI)
format-check:
	black --line-length 88 --target-version py38 --check .

# Run pre-commit hooks on all files
lint:
	pre-commit run --all-files

# Install pre-commit hooks
install-hooks:
	pre-commit install

# Run all formatting and linting
all: format lint

# Help target
help:
	@echo "Available targets:"
	@echo "  format        - Format all Python files with Black"
	@echo "  format-check  - Check if files are formatted correctly"
	@echo "  lint          - Run pre-commit hooks on all files"
	@echo "  install-hooks - Install pre-commit hooks"
	@echo "  all           - Run formatting and linting"
	@echo "  help          - Show this help message"
