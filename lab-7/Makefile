.PHONY: help install gen-data train clean

PYTHON := python

help:
	@echo "==============================================================="
	@echo "Lab 07 - LoRA e QLoRA"
	@echo "==============================================================="
	@echo "Targets disponiveis:"
	@echo "  make install   Instala dependencias"
	@echo "  make gen-data  Gera dataset sintetico (JSONL)"
	@echo "  make train     Executa treino LoRA/QLoRA"
	@echo "  make clean     Remove caches"

install:
	@pip install -r requirements.txt

gen-data:
	@$(PYTHON) generate_synthetic_dataset.py --domain "suporte academico" --num-samples 60 --train-ratio 0.9 --output-dir data

train:
	@$(PYTHON) train_lora_qlora.py --base-model meta-llama/Llama-2-7b-hf --train-file data/train.jsonl --test-file data/test.jsonl --output-dir outputs/lora-adapter

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
