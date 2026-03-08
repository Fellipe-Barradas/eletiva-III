"""
Testes Unitários para o Transformer Encoder com Multi-Head Attention
Disciplina: Tópicos em Inteligência Artificial – 2026.1
"""

import numpy as np
from transformer_encoder import (
    TransformerEncoder, PositionalEncoding, MultiHeadAttention,
    LayerNormalization, FeedForwardNetwork, EncoderLayer,
    create_vocabulary, tokenize_sentence
)


def test_positional_encoding():
    """Testa o Positional Encoding."""
    print("\n" + "="*80)
    print("TESTE 1: Positional Encoding")
    print("="*80)
    
    d_model = 64
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    
    # Verificar shape
    assert pe.pe.shape == (max_len, d_model), f"Shape incorreto: {pe.pe.shape}"
    
    # Criar embeddings de teste
    x = np.random.randn(2, 10, d_model)
    encoded = pe.encode(x)
    
    # Verificar que o shape é mantido
    assert encoded.shape == x.shape, "Shape deve ser preservado"
    
    # Verificar que é diferente do input
    assert not np.allclose(encoded, x), "PE deve modificar o input"
    
    print("✓ Positional Encoding funcionando")
    print(f"  - PE shape: {pe.pe.shape}")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {encoded.shape}")
    print(f"  - PE[0, 0:5]: {pe.pe[0, :5]}")
    print(f"  - PE[1, 0:5]: {pe.pe[1, :5]}")
    
    return True


def test_multi_head_attention():
    """Testa o Multi-Head Attention."""
    print("\n" + "="*80)
    print("TESTE 2: Multi-Head Attention")
    print("="*80)
    
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Verificar dimensões das cabeças
    assert mha.d_k == d_model // num_heads, "d_k deve ser d_model / num_heads"
    
    # Input
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Forward
    output, attention_weights = mha.forward(x)
    
    # Verificar shapes
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Output shape incorreto: {output.shape}"
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
        f"Attention shape incorreto: {attention_weights.shape}"
    
    # Verificar que pesos somam 1
    attn_sums = np.sum(attention_weights, axis=-1)
    assert np.allclose(attn_sums, 1.0), "Attention weights devem somar 1"
    
    print("✓ Multi-Head Attention funcionando")
    print(f"  - Número de cabeças: {num_heads}")
    print(f"  - d_k por cabeça: {mha.d_k}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Attention shape: {attention_weights.shape}")
    print(f"  - Exemplo de atenção (cabeça 0, token 0):")
    print(f"    {attention_weights[0, 0, 0, :]}")
    
    return True


def test_split_combine_heads():
    """Testa split_heads e combine_heads."""
    print("\n" + "="*80)
    print("TESTE 3: Split e Combine Heads")
    print("="*80)
    
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Split
    x_split = mha.split_heads(x)
    expected_split_shape = (batch_size, num_heads, seq_len, mha.d_k)
    assert x_split.shape == expected_split_shape, \
        f"Split shape incorreto: {x_split.shape}"
    
    # Combine (deve retornar ao shape original)
    x_combined = mha.combine_heads(x_split)
    assert x_combined.shape == x.shape, "Combine deve retornar shape original"
    assert np.allclose(x, x_combined), "Split + Combine deve ser identidade"
    
    print("✓ Split/Combine funcionando")
    print(f"  - Original: {x.shape}")
    print(f"  - Split: {x_split.shape}")
    print(f"  - Combined: {x_combined.shape}")
    print(f"  - Valores preservados: {np.allclose(x, x_combined)}")
    
    return True


def test_layer_normalization():
    """Testa Layer Normalization."""
    print("\n" + "="*80)
    print("TESTE 4: Layer Normalization")
    print("="*80)
    
    d_model = 64
    ln = LayerNormalization(d_model)
    
    # Input com diferentes escalas
    x = np.random.randn(2, 5, d_model) * 100
    
    # Normalizar
    x_norm = ln.forward(x)
    
    # Verificar shape
    assert x_norm.shape == x.shape, "Shape deve ser preservado"
    
    # Verificar média e variância
    mean = np.mean(x_norm, axis=-1)
    var = np.var(x_norm, axis=-1)
    
    assert np.allclose(mean, 0.0, atol=1e-5), "Média deve ser ~0"
    assert np.allclose(var, 1.0, atol=1e-5), "Variância deve ser ~1"
    
    print("✓ Layer Normalization funcionando")
    print(f"  - Input range: [{np.min(x):.2f}, {np.max(x):.2f}]")
    print(f"  - Output range: [{np.min(x_norm):.2f}, {np.max(x_norm):.2f}]")
    print(f"  - Média: {np.mean(mean):.6f}")
    print(f"  - Variância: {np.mean(var):.6f}")
    
    return True


def test_feed_forward():
    """Testa Feed-Forward Network."""
    print("\n" + "="*80)
    print("TESTE 5: Feed-Forward Network")
    print("="*80)
    
    d_model = 64
    d_ff = 256
    ffn = FeedForwardNetwork(d_model, d_ff)
    
    x = np.random.randn(2, 5, d_model)
    output = ffn.forward(x)
    
    # Verificar shape
    assert output.shape == x.shape, f"FFN deve manter shape"
    
    # Verificar que os pesos têm as dimensões corretas
    assert ffn.W1.shape == (d_model, d_ff), f"W1 shape incorreto"
    assert ffn.W2.shape == (d_ff, d_model), f"W2 shape incorreto"
    
    print("✓ FFN funcionando")
    print(f"  - Input: {x.shape}")
    print(f"  - W1: {ffn.W1.shape} (expansão)")
    print(f"  - W2: {ffn.W2.shape} (contração)")
    print(f"  - Output: {output.shape}")
    
    return True


def test_encoder_layer():
    """Testa uma camada completa do encoder."""
    print("\n" + "="*80)
    print("TESTE 6: Encoder Layer Completa")
    print("="*80)
    
    d_model = 64
    num_heads = 8
    d_ff = 256
    
    layer = EncoderLayer(d_model, num_heads, d_ff)
    
    x = np.random.randn(2, 5, d_model)
    output, attention_weights = layer.forward(x)
    
    # Verificar shapes
    assert output.shape == x.shape, "Shape deve ser preservado"
    assert attention_weights.shape == (2, num_heads, 5, 5), \
        "Attention shape incorreto"
    
    # Verificar que houve processamento
    assert not np.allclose(output, x), "Output deve ser diferente do input"
    
    print("✓ Encoder Layer funcionando")
    print(f"  - Input: {x.shape}")
    print(f"  - Output: {output.shape}")
    print(f"  - Attention: {attention_weights.shape}")
    
    return True


def test_full_transformer():
    """Testa o Transformer completo."""
    print("\n" + "="*80)
    print("TESTE 7: Transformer Encoder Completo")
    print("="*80)
    
    vocab_df, vocab_dict = create_vocabulary()
    vocab_size = len(vocab_dict)
    
    d_model = 64
    num_heads = 8
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=256,
        num_layers=6
    )
    
    sentence = "o banco bloqueou meu cartao"
    token_ids = tokenize_sentence(sentence, vocab_dict)
    
    # Forward (sem prints)
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    Z, attention_list = encoder.forward(token_ids, return_attention=True)
    
    sys.stdout = old_stdout
    
    # Verificar dimensões
    expected_shape = (1, len(token_ids), d_model)
    assert Z.shape == expected_shape, f"Shape final incorreto: {Z.shape}"
    
    # Verificar que temos atenção de todas as camadas
    assert len(attention_list) == 6, "Deve ter 6 camadas de atenção"
    
    # Verificar que não há NaN ou Inf
    assert not np.any(np.isnan(Z)), "Output contém NaN"
    assert not np.any(np.isinf(Z)), "Output contém Inf"
    
    print("✓ Transformer completo funcionando")
    print(f"  - Frase: '{sentence}'")
    print(f"  - Tokens: {len(token_ids)}")
    print(f"  - Output shape: {Z.shape}")
    print(f"  - Camadas de atenção: {len(attention_list)}")
    print(f"  - Estatísticas:")
    print(f"    Mean: {np.mean(Z):.6f}")
    print(f"    Std: {np.std(Z):.6f}")
    
    return True


def test_positional_encoding_properties():
    """Testa propriedades matemáticas do Positional Encoding."""
    print("\n" + "="*80)
    print("TESTE 8: Propriedades do Positional Encoding")
    print("="*80)
    
    d_model = 128
    pe = PositionalEncoding(d_model, 1000)
    
    # Propriedade 1: PE de posições diferentes deve ser diferente
    pe_0 = pe.pe[0, :]
    pe_1 = pe.pe[1, :]
    assert not np.allclose(pe_0, pe_1), "PE deve ser diferente para posições diferentes"
    
    # Propriedade 2: Valores devem estar em range razoável
    assert np.all(np.abs(pe.pe) <= 1.5), "PE deve ter valores em range [-1, 1] (aprox)"
    
    # Propriedade 3: Padrões periódicos (por causa do seno/cosseno)
    # Verificar que há periodicidade
    pe_100 = pe.pe[100, :]
    pe_200 = pe.pe[200, :]
    
    # Não devem ser idênticos (posições diferentes)
    assert not np.allclose(pe_100, pe_200), "Posições diferentes devem ter PE diferente"
    
    print("✓ Propriedades do PE verificadas")
    print(f"  - Range de valores: [{np.min(pe.pe):.4f}, {np.max(pe.pe):.4f}]")
    print(f"  - PE[0] != PE[1]: {not np.allclose(pe_0, pe_1)}")
    print(f"  - PE[100] != PE[200]: {not np.allclose(pe_100, pe_200)}")
    
    return True


def test_attention_symmetry():
    """Testa propriedades da atenção."""
    print("\n" + "="*80)
    print("TESTE 9: Propriedades da Atenção")
    print("="*80)
    
    d_model = 64
    num_heads = 4
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Input simétrico
    x = np.random.randn(1, 3, d_model)
    
    output, attention_weights = mha.forward(x)
    
    # Verificar que cada linha de atenção soma 1
    attn_sums = np.sum(attention_weights, axis=-1)
    assert np.allclose(attn_sums, 1.0), "Cada linha de atenção deve somar 1"
    
    # Verificar que atenção é não-negativa
    assert np.all(attention_weights >= 0), "Atenção deve ser não-negativa"
    
    print("✓ Propriedades da atenção verificadas")
    print(f"  - Soma por linha: {np.mean(attn_sums):.6f}")
    print(f"  - Todos valores >= 0: {np.all(attention_weights >= 0)}")
    print(f"  - Range: [{np.min(attention_weights):.6f}, {np.max(attention_weights):.6f}]")
    
    return True


def test_different_sequence_lengths():
    """Testa com diferentes comprimentos de sequência."""
    print("\n" + "="*80)
    print("TESTE 10: Diferentes Comprimentos de Sequência")
    print("="*80)
    
    vocab_df, vocab_dict = create_vocabulary()
    vocab_size = len(vocab_dict)
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=8,
        d_ff=256,
        num_layers=2
    )
    
    sentences = [
        "banco",
        "o banco",
        "o banco bloqueou",
        "o banco bloqueou meu cartao"
    ]
    
    # Suprimir prints
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    for sentence in sentences:
        tokens = tokenize_sentence(sentence, vocab_dict)
        Z = encoder.forward(tokens)
        
        expected_shape = (1, len(tokens), 64)
        assert Z.shape == expected_shape, \
            f"Shape incorreto para '{sentence}': {Z.shape}"
    
    sys.stdout = old_stdout
    
    print("✓ Diferentes comprimentos funcionando")
    for sentence in sentences:
        tokens = tokenize_sentence(sentence, vocab_dict)
        print(f"  - '{sentence}': {len(tokens)} tokens")
    
    return True


def run_all_tests():
    """Executa todos os testes."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*10 + "TESTES - TRANSFORMER ENCODER COM MULTI-HEAD ATTENTION" + " "*14 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    tests = [
        test_positional_encoding,
        test_multi_head_attention,
        test_split_combine_heads,
        test_layer_normalization,
        test_feed_forward,
        test_encoder_layer,
        test_full_transformer,
        test_positional_encoding_properties,
        test_attention_symmetry,
        test_different_sequence_lengths
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"\n✗ FALHOU: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERRO: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Resumo
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*30 + "RESUMO DOS TESTES" + " "*32 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    print(f"\n  Total de testes: {passed + failed}")
    print(f"  ✓ Passou: {passed}")
    print(f"  ✗ Falhou: {failed}")
    
    if failed == 0:
        print("\n  🎉 TODOS OS TESTES PASSARAM! 🎉\n")
    else:
        print(f"\n  ⚠️  {failed} teste(s) falharam\n")
    
    print("█"*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
