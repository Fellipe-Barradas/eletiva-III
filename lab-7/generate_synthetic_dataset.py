"""
Lab 07 - Geracao de dataset sintetico de instrucoes via OpenAI API.

Gera pares prompt/response em dominio escolhido e salva em JSONL com split treino/teste.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera dataset sintetico para fine-tuning de LLM")
    parser.add_argument("--domain", type=str, default="atendimento educacional", help="Dominio das instrucoes")
    parser.add_argument("--num-samples", type=int, default=60, help="Quantidade total de exemplos (>= 50)")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Proporcao de treino")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Modelo OpenAI para geracao")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidade")
    parser.add_argument("--output-dir", type=str, default="data", help="Diretorio de saida")
    return parser.parse_args()


def chunk_sizes(total: int, max_chunk: int = 25) -> List[int]:
    sizes: List[int] = []
    remaining = total
    while remaining > 0:
        current = min(max_chunk, remaining)
        sizes.append(current)
        remaining -= current
    return sizes


def ask_openai_for_pairs(client: OpenAI, model: str, domain: str, n: int) -> List[Dict[str, str]]:
    system_prompt = (
        "Voce e um gerador de dataset para SFT (supervised fine-tuning). "
        "Responda APENAS com JSON valido no formato: "
        "{\"samples\": [{\"prompt\": \"...\", \"response\": \"...\"}, ...]}."
    )

    user_prompt = (
        f"Gere {n} pares de instrucao-resposta em portugues brasileiro no dominio '{domain}'. "
        "Os prompts devem variar em dificuldade e intencao (explicar, resumir, comparar, listar passos, etc). "
        "As respostas devem ser corretas, claras e objetivas. "
        "Nao use placeholders nem referencias a ser IA."
    )

    completion = client.chat.completions.create(
        model=model,
        temperature=0.8,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Resposta vazia da OpenAI API.")

    payload = json.loads(content)
    samples = payload.get("samples", [])

    normalized: List[Dict[str, str]] = []
    for item in samples:
        prompt = str(item.get("prompt", "")).strip()
        response = str(item.get("response", "")).strip()
        if prompt and response:
            normalized.append({"prompt": prompt, "response": response})

    return normalized


def deduplicate_keep_order(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for item in items:
        key = (item["prompt"], item["response"])
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def save_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if args.num_samples < 50:
        raise ValueError("num-samples precisa ser >= 50 para atender o laboratorio.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Defina OPENAI_API_KEY antes de executar o script.")

    random.seed(args.seed)
    client = OpenAI(api_key=api_key)

    all_samples: List[Dict[str, str]] = []
    sizes = chunk_sizes(args.num_samples)

    for n in sizes:
        batch = ask_openai_for_pairs(client, args.model, args.domain, n)
        all_samples.extend(batch)

    all_samples = deduplicate_keep_order(all_samples)

    # Garante quantidade minima solicitada mesmo com deduplicacao.
    while len(all_samples) < args.num_samples:
        missing = args.num_samples - len(all_samples)
        refill = ask_openai_for_pairs(client, args.model, args.domain, min(10, missing))
        all_samples.extend(refill)
        all_samples = deduplicate_keep_order(all_samples)

    all_samples = all_samples[: args.num_samples]
    random.shuffle(all_samples)

    train_size = int(len(all_samples) * args.train_ratio)
    train_rows = all_samples[:train_size]
    test_rows = all_samples[train_size:]

    output_dir = Path(args.output_dir)
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"

    save_jsonl(train_path, train_rows)
    save_jsonl(test_path, test_rows)

    print("Dataset sintetico gerado com sucesso.")
    print(f"Dominio: {args.domain}")
    print(f"Total: {len(all_samples)} | Treino: {len(train_rows)} | Teste: {len(test_rows)}")
    print(f"Arquivos: {train_path} e {test_path}")


if __name__ == "__main__":
    main()
