.PHONY: help install dev type lint format test fuzz bench bench-dev clean all

# Default target
help:
	@echo "stsw Development Commands:"
	@echo "  make install    - Install package in development mode"
	@echo "  make dev        - Install with all development dependencies"
	@echo "  make type       - Run pyright type checker"
	@echo "  make lint       - Run ruff linter and black formatter check"
	@echo "  make format     - Auto-format code with black"
	@echo "  make test       - Run test suite with coverage"
	@echo "  make fuzz       - Run property-based tests with hypothesis"
	@echo "  make bench      - Run performance benchmarks"
	@echo "  make bench-dev  - Run quick benchmarks for development"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make all        - Run full test suite (type, lint, test, fuzz)"

# Install package in development mode
install:
	pip install -e .

# Install with all development dependencies
dev:
	pip install -e ".[dev,torch,numpy,xxhash,rich,tqdm]"
	pre-commit install

# Run pyright type checker
type:
	@echo "Running pyright..."
	pyright

# Run linters
lint:
	@echo "Running ruff..."
	ruff check .
	@echo "Running black check..."
	black --check .

# Format code
format:
	@echo "Formatting with black..."
	black .
	@echo "Fixing with ruff..."
	ruff check --fix .

# Run tests with coverage
test:
	@echo "Running tests..."
	pytest -n auto --cov=stsw --cov-branch --cov-report=term-missing

# Run property-based tests
fuzz:
	@echo "Running property tests..."
	pytest -m property --hypothesis-show-statistics

# Run full benchmarks
bench:
	@echo "Running benchmarks..."
	asv run

# Run quick benchmarks for development
bench-dev:
	@echo "Running quick benchmarks..."
	asv dev -q

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .hypothesis/
	rm -rf .benchmarks/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type f -name ".DS_Store" -delete

# Run full test suite
all: type lint test fuzz
	@echo "✅ All checks passed!"