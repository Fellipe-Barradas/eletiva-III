"""
Lab 07 - Fine-tuning eficiente com LoRA + QLoRA.

Pipeline:
1) Carrega base model em 4-bit com NF4 (bitsandbytes).
2) Aplica LoRA via PEFT.
3) Treina com SFTTrainer (trl).
4) Salva adaptador LoRA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino LoRA/QLoRA para modelo causal")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--train-file", type=str, default="data/train.jsonl")
    parser.add_argument("--test-file", type=str, default="data/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/lora-adapter")
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    return parser.parse_args()


def format_sample(prompt: str, response: str) -> str:
    return f"<s>[INST] {prompt.strip()} [/INST] {response.strip()}</s>"


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_file)
    test_path = Path(args.test_file)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Arquivos train/test JSONL nao encontrados. Rode generate_synthetic_dataset.py antes.")

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "test": str(test_path)},
    )

    def add_text_field(example):
        return {
            "text": format_sample(example["prompt"], example["response"])
        }

    dataset = dataset.map(add_text_field)

    # QLoRA: quantizacao 4-bit NF4 com computacao float16.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )

    # LoRA obrigatorio: r=64, alpha=16, dropout=0.1.
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=False,
        fp16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=training_args,
    )

    trainer.train()

    # Salva adaptador treinado (nao salva full model).
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Treinamento finalizado. Adaptador salvo em:", args.output_dir)


if __name__ == "__main__":
    main()
