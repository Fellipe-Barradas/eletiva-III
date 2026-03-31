"""Testes do Laboratorio 6 (BPE e WordPiece)."""

from bpe_wordpiece import get_stats, merge_vocab, train_bpe, vocab


def test_get_stats_max_pair_is_e_s():
    stats = get_stats(vocab)

    assert stats[("e", "s")] == 9
    assert max(stats, key=stats.get) == ("e", "s")


def test_merge_vocab_e_s_creates_es_tokens():
    merged = merge_vocab(("e", "s"), vocab)

    assert "n e w es t </w>" in merged
    assert "w i d es t </w>" in merged


def test_train_bpe_generates_morphological_suffix():
    final_vocab, _ = train_bpe(vocab, num_merges=5)

    # Esperado: ao menos uma forma contendo est</w>
    assert any("est</w>" in word for word in final_vocab)


if __name__ == "__main__":
    test_get_stats_max_pair_is_e_s()
    test_merge_vocab_e_s_creates_es_tokens()
    test_train_bpe_generates_morphological_suffix()
    print("Todos os testes passaram.")
