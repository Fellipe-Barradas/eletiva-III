.PHONY: install run clean help test

help:
	@echo "LAB 09: Arquitetura RAG Avançada (HNSW, HyDE e Cross-Encoders)"
	@echo ""
	@echo "Comandos disponíveis:"
	@echo "  make install    - Instala dependências"
	@echo "  make run        - Executa o pipeline RAG"
	@echo "  make test       - Roda testes rápidos"
	@echo "  make clean      - Remove arquivos gerados"
	@echo "  make help       - Exibe esta mensagem"

install:
	@echo "📦 Instalando dependências..."
	pip install -r requirements.txt
	@echo "✓ Dependências instaladas"

run:
	@echo "🚀 Executando pipeline RAG..."
	python rag_advanced.py
	@echo ""
	@echo "✓ Pipeline completo! Resultados salvos em rag_results.json"

test:
	@echo "🧪 Verificando integridade do código..."
	python -m py_compile rag_advanced.py
	@echo "✓ Sintaxe OK"

clean:
	@echo "🧹 Limpando arquivos gerados..."
	rm -f rag_results.json
	rm -f *.pyc
	rm -rf __pycache__
	@echo "✓ Limpeza concluída"
