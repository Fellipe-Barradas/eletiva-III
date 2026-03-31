"""
Laboratorio 6 - P2
Construindo um tokenizador BPE e explorando WordPiece (BERT multilingual).
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from typing import Dict, List, Tuple


# Inicializacao estritamente conforme enunciado.
vocab = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3,
}


def get_stats(vocab_in: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """Conta frequencia de pares adjacentes de simbolos no vocabulario."""
    pairs: Dict[Tuple[str, str], int] = defaultdict(int)

    for word, freq in vocab_in.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq

    return dict(pairs)


def merge_vocab(pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
    """Substitui todas as ocorrencias do par isolado pela versao unificada."""
    v_out: Dict[str, int] = {}

    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    replacement = "".join(pair)

    for word, freq in v_in.items():
        merged_word = pattern.sub(replacement, word)
        v_out[merged_word] = freq

    return v_out


def train_bpe(vocab_in: Dict[str, int], num_merges: int = 5) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """Executa K iteracoes de fusao BPE e imprime progresso a cada rodada."""
    current_vocab = copy.deepcopy(vocab_in)
    merges_done: List[Tuple[str, str]] = []

    for i in range(1, num_merges + 1):
        stats = get_stats(current_vocab)
        if not stats:
            print(f"Iteracao {i}: sem pares disponiveis para fusao.")
            break

        best_pair = max(stats, key=stats.get)
        merges_done.append(best_pair)

        current_vocab = merge_vocab(best_pair, current_vocab)

        print(f"Iteracao {i}/{num_merges}")
        print(f"Par mais frequente fundido: {best_pair} (freq={stats[best_pair]})")
        print("Vocab apos fusao:")
        for tokenized_word, freq in current_vocab.items():
            print(f"  {tokenized_word}: {freq}")
        print("-" * 60)

    return current_vocab, merges_done


def tokenize_with_wordpiece(sentence: str) -> List[str]:
    """Tokeniza uma frase com WordPiece via BERT multilingual cased."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    return tokenizer.tokenize(sentence)


def main() -> None:
    print("=" * 60)
    print("TAREFA 1 - MOTOR DE FREQUENCIAS")
    print("=" * 60)

    stats = get_stats(vocab)
    max_pair = max(stats, key=stats.get)
    print(f"Par mais frequente inicial: {max_pair} (freq={stats[max_pair]})")
    print(f"Frequencia obrigatoria de ('e', 's'): {stats.get(('e', 's'), 0)}")

    print("\n" + "=" * 60)
    print("TAREFA 2 - LOOP DE FUSAO (K=5)")
    print("=" * 60)

    final_vocab, merges = train_bpe(vocab, num_merges=5)

    print("Merges realizados:")
    for idx, pair in enumerate(merges, start=1):
        print(f"  {idx}. {pair}")

    print("\nVocab final apos 5 iteracoes:")
    for tokenized_word, freq in final_vocab.items():
        print(f"  {tokenized_word}: {freq}")

    print("\n" + "=" * 60)
    print("TAREFA 3 - WORDPIECE (HUGGING FACE)")
    print("=" * 60)

    sentence = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."

    try:
        tokens = tokenize_with_wordpiece(sentence)
        print("Frase de teste:")
        print(sentence)
        print("\nTokens WordPiece:")
        print(tokens)
    except Exception as exc:
        print("Falha ao carregar/tokenizar com Hugging Face.")
        print("Instale dependencias e garanta acesso a internet para baixar o modelo.")
        print(f"Detalhe do erro: {exc}")


if __name__ == "__main__":
    main()
