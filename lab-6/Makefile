.PHONY: run test install clean help

PYTHON := python

help:
	@echo "==============================================================="
	@echo "Lab 06 - BPE e WordPiece"
	@echo "==============================================================="
	@echo ""
	@echo "Targets disponiveis:"
	@echo "  make run      Executa script principal do laboratorio"
	@echo "  make test     Executa validacoes da parte BPE"
	@echo "  make install  Instala transformers e dependencias"
	@echo "  make clean    Limpa cache Python"
	@echo ""

run:
	@$(PYTHON) bpe_wordpiece.py

test:
	@$(PYTHON) test_bpe_wordpiece.py

install:
	@pip install transformers numpy

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
