"""
Testes Unitários para o Transformer Encoder
Disciplina: Tópicos em Inteligência Artificial – 2026.1
"""

import numpy as np
from transformer_encoder import TransformerEncoder, create_vocabulary, tokenize_sentence


def test_vocabulary():
    """Testa a criação do vocabulário."""
    print("\n" + "="*70)
    print("TESTE 1: Criação do Vocabulário")
    print("="*70)
    
    vocab_df, vocab_dict = create_vocabulary()
    
    assert len(vocab_dict) > 0, "Vocabulário não pode estar vazio"
    assert "banco" in vocab_dict, "Palavra 'banco' deve estar no vocabulário"
    assert vocab_dict["o"] == 0, "ID da palavra 'o' deve ser 0"
    
    print("✓ Vocabulário criado corretamente")
    print(f"  - Tamanho: {len(vocab_dict)} palavras")
    print(f"  - Primeiras palavras: {list(vocab_dict.keys())[:5]}")
    return True


def test_tokenization():
    """Testa a tokenização de frases."""
    print("\n" + "="*70)
    print("TESTE 2: Tokenização")
    print("="*70)
    
    _, vocab_dict = create_vocabulary()
    sentence = "o banco bloqueou meu cartao"
    token_ids = tokenize_sentence(sentence, vocab_dict)
    
    assert len(token_ids) == 5, f"Esperado 5 tokens, obtido {len(token_ids)}"
    assert token_ids[0] == 0, "Primeiro token deve ser 'o' (ID=0)"
    assert token_ids[1] == 1, "Segundo token deve ser 'banco' (ID=1)"
    
    print(f"✓ Tokenização correta")
    print(f"  - Frase: '{sentence}'")
    print(f"  - Tokens: {token_ids}")
    return True


def test_embedding():
    """Testa a camada de embedding."""
    print("\n" + "="*70)
    print("TESTE 3: Embeddings")
    print("="*70)
    
    vocab_size = 11
    d_model = 64
    encoder = TransformerEncoder(vocab_size=vocab_size, d_model=d_model)
    
    token_ids = [0, 1, 2, 4, 3]
    embeddings = encoder.embed(token_ids)
    
    expected_shape = (1, len(token_ids), d_model)
    assert embeddings.shape == expected_shape, \
        f"Shape esperado {expected_shape}, obtido {embeddings.shape}"
    
    print("✓ Embeddings gerados corretamente")
    print(f"  - Shape: {embeddings.shape}")
    print(f"  - Tipo: {embeddings.dtype}")
    return True


def test_softmax():
    """Testa a implementação do softmax."""
    print("\n" + "="*70)
    print("TESTE 4: Softmax")
    print("="*70)
    
    encoder = TransformerEncoder(vocab_size=10, d_model=64)
    
    # Teste 1: Vetor simples
    x = np.array([1.0, 2.0, 3.0])
    result = encoder.softmax(x)
    
    assert np.allclose(np.sum(result), 1.0), "Soma do softmax deve ser 1.0"
    assert np.all(result >= 0) and np.all(result <= 1), "Valores devem estar entre 0 e 1"
    
    # Teste 2: Matriz 2D
    x_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result_2d = encoder.softmax(x_2d, axis=-1)
    
    assert result_2d.shape == x_2d.shape, "Shape deve ser preservado"
    assert np.allclose(np.sum(result_2d, axis=-1), [1.0, 1.0]), \
        "Cada linha deve somar 1.0"
    
    print("✓ Softmax funcionando corretamente")
    print(f"  - Input: {x}")
    print(f"  - Output: {result}")
    print(f"  - Soma: {np.sum(result)}")
    return True


def test_attention_shapes():
    """Testa as dimensões da camada de atenção."""
    print("\n" + "="*70)
    print("TESTE 5: Scaled Dot-Product Attention")
    print("="*70)
    
    batch_size = 2
    seq_len = 5
    d_model = 64
    
    encoder = TransformerEncoder(vocab_size=10, d_model=d_model)
    
    # Criar Q, K, V aleatórios
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    output, attention_weights = encoder.scaled_dot_product_attention(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Output shape incorreto: {output.shape}"
    assert attention_weights.shape == (batch_size, seq_len, seq_len), \
        f"Attention weights shape incorreto: {attention_weights.shape}"
    
    # Verificar que os pesos de atenção somam 1 para cada token
    attention_sums = np.sum(attention_weights, axis=-1)
    assert np.allclose(attention_sums, 1.0), \
        "Pesos de atenção devem somar 1.0 para cada posição"
    
    print("✓ Attention funcionando corretamente")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Attention weights shape: {attention_weights.shape}")
    print(f"  - Exemplo de pesos (primeiro token): {attention_weights[0, 0, :]}")
    return True


def test_layer_norm():
    """Testa a normalização de camada."""
    print("\n" + "="*70)
    print("TESTE 6: Layer Normalization")
    print("="*70)
    
    encoder = TransformerEncoder(vocab_size=10, d_model=64)
    
    # Criar tensor aleatório
    x = np.random.randn(2, 5, 64) * 10  # Escala maior para testar normalização
    gamma = np.ones(64)
    beta = np.zeros(64)
    
    normalized = encoder.layer_norm(x, gamma, beta)
    
    # Verificar shape
    assert normalized.shape == x.shape, "Shape deve ser preservado"
    
    # Verificar que a média é aproximadamente 0 e variância é aproximadamente 1
    # (na dimensão dos features)
    mean = np.mean(normalized, axis=-1)
    var = np.var(normalized, axis=-1)
    
    assert np.allclose(mean, 0.0, atol=1e-5), \
        f"Média deveria ser ~0, obtido {np.mean(mean)}"
    assert np.allclose(var, 1.0, atol=1e-5), \
        f"Variância deveria ser ~1, obtido {np.mean(var)}"
    
    print("✓ Layer Normalization funcionando corretamente")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {normalized.shape}")
    print(f"  - Média (deve ser ~0): {np.mean(mean):.6f}")
    print(f"  - Variância (deve ser ~1): {np.mean(var):.6f}")
    return True


def test_ffn():
    """Testa a Feed-Forward Network."""
    print("\n" + "="*70)
    print("TESTE 7: Feed-Forward Network")
    print("="*70)
    
    d_model = 64
    d_ff = 256
    encoder = TransformerEncoder(vocab_size=10, d_model=d_model, d_ff=d_ff)
    
    # Criar input aleatório
    x = np.random.randn(2, 5, d_model)
    
    # Pegar pesos da primeira camada
    weights = encoder.layers[0]
    
    # Aplicar FFN
    output = encoder.feed_forward_network(x, weights)
    
    # Verificar shape
    assert output.shape == x.shape, \
        f"FFN deve manter shape, esperado {x.shape}, obtido {output.shape}"
    
    # Verificar que há não-linearidade (por causa do ReLU)
    # Output não deve ser uma transformação puramente linear
    linear_transform = np.matmul(np.matmul(x, weights['W1']) + weights['b1'], weights['W2']) + weights['b2']
    assert not np.allclose(output, linear_transform), \
        "FFN deve ter não-linearidade (ReLU)"
    
    print("✓ FFN funcionando corretamente")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - W1 shape: {weights['W1'].shape} (d_model -> d_ff)")
    print(f"  - W2 shape: {weights['W2'].shape} (d_ff -> d_model)")
    return True


def test_encoder_layer():
    """Testa uma camada completa do encoder."""
    print("\n" + "="*70)
    print("TESTE 8: Camada Completa do Encoder")
    print("="*70)
    
    d_model = 64
    encoder = TransformerEncoder(vocab_size=10, d_model=d_model)
    
    # Criar input aleatório
    x = np.random.randn(2, 5, d_model)
    
    # Processar através de uma camada
    output = encoder.encoder_layer(x, encoder.layers[0])
    
    # Verificar shape
    assert output.shape == x.shape, \
        f"Camada deve preservar shape, esperado {x.shape}, obtido {output.shape}"
    
    # Verificar que a saída é diferente da entrada (houve processamento)
    assert not np.allclose(output, x), "Saída deve ser diferente da entrada"
    
    print("✓ Camada do Encoder funcionando corretamente")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {output.shape}")
    return True


def test_full_forward_pass():
    """Testa o forward pass completo através de todas as 6 camadas."""
    print("\n" + "="*70)
    print("TESTE 9: Forward Pass Completo (N=6 Camadas)")
    print("="*70)
    
    _, vocab_dict = create_vocabulary()
    vocab_size = len(vocab_dict)
    d_model = 64
    n_layers = 6
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers
    )
    
    # Tokenizar frase
    sentence = "o banco bloqueou meu cartao"
    token_ids = tokenize_sentence(sentence, vocab_dict)
    
    # Forward pass
    Z = encoder.forward(token_ids)
    
    # Verificar dimensões finais
    expected_shape = (1, len(token_ids), d_model)
    assert Z.shape == expected_shape, \
        f"Shape final incorreto, esperado {expected_shape}, obtido {Z.shape}"
    
    # Verificar que não há NaN ou Inf
    assert not np.any(np.isnan(Z)), "Saída contém NaN"
    assert not np.any(np.isinf(Z)), "Saída contém Inf"
    
    print("✓ Forward Pass completo funcionando corretamente")
    print(f"  - Frase: '{sentence}'")
    print(f"  - Tokens: {token_ids}")
    print(f"  - Shape final: {Z.shape}")
    print(f"  - Número de camadas: {n_layers}")
    return True


def test_batch_processing():
    """Testa processamento de múltiplas frases (batch)."""
    print("\n" + "="*70)
    print("TESTE 10: Processamento em Batch")
    print("="*70)
    
    _, vocab_dict = create_vocabulary()
    vocab_size = len(vocab_dict)
    d_model = 64
    
    encoder = TransformerEncoder(vocab_size=vocab_size, d_model=d_model)
    
    # Criar batch de tokens (simular 2 frases com padding)
    batch_tokens = np.array([
        [0, 1, 2, 4, 3],  # "o banco bloqueou meu cartao"
        [5, 6, 7, 8, 9]   # "cliente solicitou novo emprestimo"
    ])
    
    # Embeddings
    embeddings = encoder.embedding_table[batch_tokens]
    
    # Processar primeira camada
    output = encoder.encoder_layer(embeddings, encoder.layers[0])
    
    # Verificar shape
    expected_shape = (2, 5, d_model)
    assert output.shape == expected_shape, \
        f"Batch processing falhou, esperado {expected_shape}, obtido {output.shape}"
    
    print("✓ Processamento em batch funcionando")
    print(f"  - Batch size: 2")
    print(f"  - Sequence length: 5")
    print(f"  - Output shape: {output.shape}")
    return True


def run_all_tests():
    """Executa todos os testes."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*15 + "TESTES DO TRANSFORMER ENCODER" + " "*23 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    tests = [
        test_vocabulary,
        test_tokenization,
        test_embedding,
        test_softmax,
        test_attention_shapes,
        test_layer_norm,
        test_ffn,
        test_encoder_layer,
        test_full_forward_pass,
        test_batch_processing
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
            failed += 1
    
    # Resumo
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*25 + "RESUMO DOS TESTES" + " "*27 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    print(f"\n  Total de testes: {passed + failed}")
    print(f"  ✓ Passou: {passed}")
    print(f"  ✗ Falhou: {failed}")
    
    if failed == 0:
        print("\n  🎉 TODOS OS TESTES PASSARAM! 🎉\n")
    else:
        print(f"\n  ⚠️  {failed} teste(s) falharam\n")
    
    print("█"*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
