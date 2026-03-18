.PHONY: run test clean help install

PYTHON := python
VENV := venv

help:
	@echo "==================================================================="
	@echo "Lab 04 - Transformer Completo From Scratch"
	@echo "==================================================================="
	@echo ""
	@echo "Targets disponíveis:"
	@echo ""
	@echo "  make run              Executa a demo completa do Transformer"
	@echo "  make test             Executa suite de testes unitários"
	@echo "  make clean            Limpa arquivos gerados (__pycache__, etc)"
	@echo "  make install          Instala dependências (numpy, pandas)"
	@echo "  make help             Mostra esta mensagem"
	@echo ""
	@echo "Exemplo:"
	@echo "  $$ make test     # Testar todos os componentes"
	@echo "  $$ make run      # Rodar inferência completa end-to-end"
	@echo ""

run:
	@echo "===================================================================="
	@echo "EXECUTANDO: Transformer Encoder-Decoder Completo (Tarefa 4)"
	@echo "===================================================================="
	@echo ""
	@$(PYTHON) transformer_completo.py

test:
	@echo "===================================================================="
	@echo "EXECUTANDO: Suite de Testes Unitários"
	@echo "===================================================================="
	@echo ""
	@$(PYTHON) test_transformer.py

install:
	@echo "Instalando dependências..."
	@pip install numpy pandas

clean:
	@echo "Limpando arquivos gerados..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✓ Limpeza concluída"

all: test run
	@echo ""
	@echo "===================================================================="
	@echo "✓ TESTE + DEMO CONCLUÍDOS COM SUCESSO"
	@echo "===================================================================="
