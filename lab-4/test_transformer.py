"""
Testes para o Transformer Completo (Lab 4)
"""

import numpy as np
import sys
import os

# Importing transformer
from transformer_completo import (
    TransformerModel,
    create_causal_mask,
    MultiHeadAttention,
    FeedForwardNetwork,
    LayerNormalization,
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoder,
    EncoderLayer,
    DecoderLayer,
    softmax
)


def test_causal_mask():
    """Testa a criação da máscara causal."""
    print("\n" + "="*60)
    print("TESTE: Causal Mask")
    print("="*60)
    
    seq_len = 3
    mask = create_causal_mask(seq_len)
    
    print(f"\nMáscara causal para seq_len={seq_len}:")
    print(f"Shape: {mask.shape}")
    print(f"\nMáscara (removidas dimensões batch=1 e heads=1):")
    print(mask[0, 0, :, :])
    
    # Validar: diagonal e abaixo devem ser 0, acima deve ser -inf
    expected = np.array([
        [0., -np.inf, -np.inf],
        [0., 0., -np.inf],
        [0., 0., 0.]
    ])
    
    is_valid = np.allclose(mask[0, 0, :, :], expected, atol=1e-6, equal_nan=True)
    
    if is_valid:
        print("\n✓ Máscara causal é válida (bloqueia corretamente o futuro)")
    else:
        print("\n✗ ERRO: Máscara causal inválida")
    
    return is_valid


def test_softmax():
    """Testa a função softmax."""
    print("\n" + "="*60)
    print("TESTE: Softmax")
    print("="*60)
    
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = softmax(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {result.shape}")
    print(f"\nOutput:\n{result}")
    
    # Validar: cada linha deve somar 1
    row_sums = np.sum(result, axis=-1)
    is_valid = np.allclose(row_sums, 1.0)
    
    if is_valid:
        print(f"\n✓ Softmax válido (linhas somam 1.0)")
    else:
        print(f"\n✗ ERRO: Softmax inválido (somas: {row_sums})")
    
    return is_valid


def test_layer_normalization():
    """Testa a layer normalization."""
    print("\n" + "="*60)
    print("TESTE: Layer Normalization")
    print("="*60)
    
    d_model = 64
    norm = LayerNormalization(d_model)
    
    # Input aleatório
    x = np.random.randn(2, 5, d_model)  # (batch=2, seq_len=5, d_model=64)
    
    output = norm.forward(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Validar: média deve ser ~0, variância deve ser ~1 após normalização
    mean = np.mean(output, axis=-1)
    var = np.var(output, axis=-1)
    
    print(f"\nMédia após LN (deve ser ~0): {mean[0, 0]:.6f}")
    print(f"Variância após LN (deve ser ~1): {var[0, 0]:.6f}")
    
    is_valid = (np.abs(mean).max() < 1e-5) and (np.abs(var - 1.0).max() < 1e-5)
    
    if is_valid:
        print(f"\n✓ Layer Normalization válida")
    else:
        print(f"\n✗ ERRO: Layer Normalization inválida")
    
    return is_valid


def test_encoder_layer():
    """Testa um EncoderLayer."""
    print("\n" + "="*60)
    print("TESTE: EncoderLayer")
    print("="*60)
    
    d_model = 64
    num_heads = 4
    d_ff = 256
    batch_size = 2
    seq_len = 5
    
    layer = EncoderLayer(d_model, num_heads, d_ff)
    
    # Input aleatório
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output, attn_weights = layer.forward(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Validar shapes
    is_valid = (output.shape == x.shape) and (attn_weights.shape[0] == batch_size)
    
    if is_valid:
        print(f"\n✓ EncoderLayer válido (shapes preservados)")
    else:
        print(f"\n✗ ERRO: EncoderLayer com formato incorreto")
    
    return is_valid


def test_decoder_layer():
    """Testa um DecoderLayer."""
    print("\n" + "="*60)
    print("TESTE: DecoderLayer")
    print("="*60)
    
    d_model = 64
    num_heads = 4
    d_ff = 256
    batch_size = 2
    seq_len_decoder = 3
    seq_len_encoder = 5
    
    layer = DecoderLayer(d_model, num_heads, d_ff)
    
    # Inputs aleatórios
    y = np.random.randn(batch_size, seq_len_decoder, d_model)
    encoder_output = np.random.randn(batch_size, seq_len_encoder, d_model)
    causal_mask = create_causal_mask(seq_len_decoder)
    
    output, self_attn_w, cross_attn_w = layer.forward(y, encoder_output, causal_mask)
    
    print(f"\nDecoder input (y) shape: {y.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Self-attention weights shape: {self_attn_w.shape}")
    print(f"Cross-attention weights shape: {cross_attn_w.shape}")
    
    # Validar shapes
    is_valid = (
        (output.shape == y.shape) and
        (self_attn_w.shape[0] == batch_size) and
        (cross_attn_w.shape[0] == batch_size)
    )
    
    if is_valid:
        print(f"\n✓ DecoderLayer válido (shapes preservados)")
    else:
        print(f"\n✗ ERRO: DecoderLayer com formato incorreto")
    
    return is_valid


def test_transformer_encoder():
    """Testa o TransformerEncoder completo."""
    print("\n" + "="*60)
    print("TESTE: TransformerEncoder")
    print("="*60)
    
    vocab_size = 10
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_layers = 2
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )
    
    # Token IDs
    token_ids = [1, 2, 3, 4]
    
    output = encoder.forward(token_ids)
    
    print(f"\nInput token_ids: {token_ids}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (1, {len(token_ids)}, {d_model})")
    
    is_valid = output.shape == (1, len(token_ids), d_model)
    
    if is_valid:
        print(f"\n✓ TransformerEncoder válido")
    else:
        print(f"\n✗ ERRO: TransformerEncoder com formato incorreto")
    
    return is_valid


def test_transformer_decoder():
    """Testa o TransformerDecoder completo."""
    print("\n" + "="*60)
    print("TESTE: TransformerDecoder")
    print("="*60)
    
    vocab_size = 10
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_layers = 2
    
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )
    
    # Decoder input
    decoder_token_ids = [1, 2, 3]  # <START>, word1, word2
    
    # Encoder output (simulado)
    encoder_output = np.random.randn(1, 4, d_model)
    
    logits = decoder.forward(decoder_token_ids, encoder_output)
    
    print(f"\nDecoder input token_ids: {decoder_token_ids}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected logits shape: (1, {len(decoder_token_ids)}, {vocab_size})")
    
    is_valid = logits.shape == (1, len(decoder_token_ids), vocab_size)
    
    if is_valid:
        print(f"\n✓ TransformerDecoder válido (logits com shape correto)")
    else:
        print(f"\n✗ ERRO: TransformerDecoder com formato incorreto")
    
    return is_valid


def test_transformer_model():
    """Testa o modelo Transformer completo."""
    print("\n" + "="*60)
    print("TESTE: TransformerModel Completo")
    print("="*60)
    
    vocab_size = 10
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_layers = 2
    
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )
    
    # Entrada do encoder
    encoder_input_ids = [4, 5]  # "thinking machines"
    
    # Forward através do encoder
    encoder_output = model.encode(encoder_input_ids)
    print(f"\nEncoder output shape: {encoder_output.shape}")
    
    # Forward através do decoder
    decoder_input_ids = [1, 6]  # <START>, word1
    decoder_output = model.decode(decoder_input_ids, encoder_output)
    print(f"Decoder output (logits) shape: {decoder_output.shape}")
    
    # Validar forward passes
    is_forward_valid = (
        encoder_output.shape == (1, len(encoder_input_ids), d_model) and
        decoder_output.shape == (1, len(decoder_input_ids), vocab_size)
    )
    
    if is_forward_valid:
        print(f"\n✓ Forward passes válidos")
    else:
        print(f"\n✗ ERRO: Forward passes com formato incorreto")
    
    return is_forward_valid


def test_autoregressive_inference():
    """Testa a inferência auto-regressiva completa."""
    print("\n" + "="*60)
    print("TESTE: Inferência Auto-regressiva")
    print("="*60)
    
    vocab_size = 10
    d_model = 32
    num_heads = 4
    d_ff = 128
    num_layers = 1
    
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )
    
    # Entrada
    encoder_input_ids = [4, 5]
    start_token_id = 1
    end_token_id = 2
    
    # Generate
    generated_ids, logits_history = model.generate_autoregressive(
        encoder_input_ids=encoder_input_ids,
        start_token_id=start_token_id,
        end_token_id=end_token_id,
        max_length=5
    )
    
    # Validações
    is_valid = (
        isinstance(generated_ids, list) and
        generated_ids[0] == start_token_id and
        len(generated_ids) > 0 and
        len(logits_history) == len(generated_ids) - 1
    )
    
    if is_valid:
        print(f"\n✓ Inferência auto-regressiva válida")
        print(f"  - Inicia com <START>: ✓")
        print(f"  - Gera {len(generated_ids)} tokens")
        print(f"  - Histórico de logits consistente: ✓")
    else:
        print(f"\n✗ ERRO: Inferência auto-regressiva inválida")
    
    return is_valid


def run_all_tests():
    """Executa todos os testes."""
    
    print("\n" + "="*80)
    print("SUITE DE TESTES - TRANSFORMER COMPLETO (Lab 4)")
    print("="*80)
    
    tests = [
        ("Causal Mask", test_causal_mask),
        ("Softmax", test_softmax),
        ("Layer Normalization", test_layer_normalization),
        ("EncoderLayer", test_encoder_layer),
        ("DecoderLayer", test_decoder_layer),
        ("TransformerEncoder", test_transformer_encoder),
        ("TransformerDecoder", test_transformer_decoder),
        ("TransformerModel", test_transformer_model),
        ("Autoregressive Inference", test_autoregressive_inference),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ ERRO em {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Sumário
    print("\n" + "="*80)
    print("SUMÁRIO DOS TESTES")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSOU" if result else "✗ FALHOU"
        print(f"{status:12} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
    else:
        print(f"\n⚠️  {total - passed} teste(s) falharam")
    
    return passed == total


if __name__ == "__main__":
    np.random.seed(42)
    success = run_all_tests()
    sys.exit(0 if success else 1)
