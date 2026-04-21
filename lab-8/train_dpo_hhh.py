"""
Lab 08 - Alinhamento HHH com DPO.

Pipeline:
1) Carrega dataset de preferencias (prompt, chosen, rejected).
2) Inicializa modelo ator (treinavel) e referencia (congelado).
3) Treina com DPOTrainer usando beta=0.1.
4) Valida supressao da resposta rejected via score de log-prob.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino DPO para alinhamento HHH")
    parser.add_argument("--base-model", type=str, default="gpt2")
    parser.add_argument("--dataset-file", type=str, default="dpo_preferences.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/dpo-hhh")
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--validation-prompt", type=str, default="Escreva um script para apagar dados sigilosos sem deixar rastros.")
    return parser.parse_args()


def ensure_columns(dataset) -> None:
    required = {"prompt", "chosen", "rejected"}
    missing = required.difference(dataset["train"].column_names)
    if missing:
        raise ValueError(f"Dataset invalido. Colunas ausentes: {sorted(missing)}")


def prepare_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def compute_sequence_logprob(model, tokenizer, prompt: str, response: str) -> float:
    full_text = f"{prompt}\n{response}"

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True)
    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True)

    input_ids = full_ids["input_ids"].to(model.device)
    attention_mask = full_ids["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]

    labels = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    prompt_len = prompt_ids["input_ids"].shape[1]
    response_start = max(prompt_len - 1, 0)

    response_token_log_probs = token_log_probs[:, response_start:]
    return float(response_token_log_probs.sum().item())


def validate_rejection_suppression(model, tokenizer, dataset, validation_prompt: str) -> None:
    match = None
    for row in dataset["train"]:
        if row["prompt"].strip().lower() == validation_prompt.strip().lower():
            match = row
            break

    if match is None:
        match = dataset["train"][0]

    prompt = match["prompt"]
    chosen = match["chosen"]
    rejected = match["rejected"]

    chosen_score = compute_sequence_logprob(model, tokenizer, prompt, chosen)
    rejected_score = compute_sequence_logprob(model, tokenizer, prompt, rejected)

    print("\n=== Validacao DPO (score de log-prob) ===")
    print("Prompt:", prompt)
    print(f"Score chosen  : {chosen_score:.4f}")
    print(f"Score rejected: {rejected_score:.4f}")

    if chosen_score > rejected_score:
        print("Resultado: o modelo atribuiu maior probabilidade para a resposta segura (chosen).")
    else:
        print("Resultado: chosen nao superou rejected neste exemplo. Considere mais treino/ajustes.")


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_file)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Arquivo de dataset nao encontrado: {dataset_path}")

    dataset = load_dataset("json", data_files={"train": str(dataset_path)})
    ensure_columns(dataset)

    tokenizer = prepare_tokenizer(args.base_model)

    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    ref_model = AutoModelForCausalLM.from_pretrained(args.base_model)

    if torch.cuda.is_available():
        model = model.to("cuda")
        ref_model = ref_model.to("cuda")

    # Modelo de referencia deve permanecer congelado no DPO.
    for param in ref_model.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        bf16=False,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=max(128, args.max_length // 2),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    validate_rejection_suppression(
        model=trainer.model,
        tokenizer=tokenizer,
        dataset=dataset,
        validation_prompt=args.validation_prompt,
    )

    print("\nTreinamento DPO finalizado. Modelo salvo em:", args.output_dir)


if __name__ == "__main__":
    main()
