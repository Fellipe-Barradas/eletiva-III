# Makefile para o Lab 3 - Transformer Encoder
# Tópicos em Inteligência Artificial - 2026.1

.PHONY: run test clean help

# Comando padrão
all: run

# Executar o programa principal
run:
	@echo "Executando Transformer Encoder..."
	python transformer_encoder.py

# Executar os testes
test:
	@echo "Executando testes..."
	python test_transformer.py

# Limpar arquivos de cache
clean:
	@echo "Limpando arquivos de cache..."
	rm -rf __pycache__
	rm -f *.pyc
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Limpeza concluída!"

# Exibir ajuda
help:
	@echo "Comandos disponíveis:"
	@echo "  make run    - Executar o Transformer Encoder"
	@echo "  make test   - Executar os testes unitários"
	@echo "  make clean  - Limpar arquivos de cache"
	@echo "  make help   - Exibir esta mensagem"
