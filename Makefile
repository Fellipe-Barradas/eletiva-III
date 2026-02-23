.PHONY: help install test clean check run

# Variables
PYTHON := python
PIP := pip

# Default target
help:
	@echo "=========================================="
	@echo "  Makefile - Scaled Dot-Product Attention"
	@echo "=========================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install   - Install dependencies (numpy)"
	@echo "  make test      - Run unit tests"
	@echo "  make run       - Run usage example"
	@echo "  make check     - Check if dependencies are installed"
	@echo "  make clean     - Remove temporary files"
	@echo "  make help      - Show this message"
	@echo ""

# Install dependencies
install:
	@echo "Installing dependencies..."
	$(PIP) install numpy

# Check if dependencies are installed
check:
	@echo "Checking dependencies..."
	@$(PYTHON) -c "import numpy; print(f'NumPy {numpy.__version__} installed')" || \
		(echo "NumPy is not installed. Run: make install" && exit 1)

# Run tests
test: check
	@echo "Running tests..."
	$(PYTHON) test_attention.py

# Run usage example
run: check
	@echo "Running Scaled Dot-Product Attention example..."
	@$(PYTHON) -c "\
import numpy as np; \
from attention import scaled_dot_product_attention; \
print('Scaled Dot-Product Attention Example:'); \
print('=' * 50); \
Q = np.array([[1.0, 0.0], [0.0, 1.0]]); \
K = np.array([[1.0, 0.0], [0.0, 1.0]]); \
V = np.array([[1.0, 0.0], [0.0, 1.0]]); \
output, weights = scaled_dot_product_attention(Q, K, V); \
print('\nQuery (Q):'); print(Q); \
print('\nKey (K):'); print(K); \
print('\nValue (V):'); print(V); \
print('\nAttention Weights:'); print(weights); \
print('\nOutput:'); print(output); \
print('=' * 50); \
"

# Remove temporary files
clean:
	@echo "Cleaning temporary files..."
	@$(PYTHON) -c "import shutil, pathlib; \
		[shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; \
		[p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]; \
		[p.unlink() for p in pathlib.Path('.').rglob('*.pyo')]; \
		print('Temporary files removed.')"
