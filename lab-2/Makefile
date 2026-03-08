# Makefile para Lab 2 - Transformer Encoder
# Tópicos em Inteligência Artificial – 2026.1

.PHONY: help run test clean all

# Comando padrão quando executar apenas 'make'
help:
	@echo "================================================"
	@echo "  Transformer Encoder - Comandos Disponíveis"
	@echo "================================================"
	@echo ""
	@echo "  make run      - Executa o programa principal"
	@echo "  make test     - Executa os testes unitários"
	@echo "  make all      - Executa testes e depois o programa"
	@echo "  make clean    - Remove arquivos temporários"
	@echo ""

# Executa o programa principal
run:
	@echo "Executando Transformer Encoder..."
	@echo "=================================="
	python transformer_encoder.py

# Executa os testes unitários
test:
	@echo "Executando Testes Unitários..."
	@echo "==============================="
	python test_transformer.py

# Executa testes e depois o programa
all: test run

# Limpa arquivos temporários
clean:
	@echo "Limpando arquivos temporários..."
	@rm -rf __pycache__
	@rm -f *.pyc
	@echo "Limpeza concluída!"
