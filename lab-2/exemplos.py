"""
Exemplos de Uso do Transformer Encoder
Disciplina: Tópicos em Inteligência Artificial – 2026.1

Este arquivo demonstra diferentes formas de usar o Transformer Encoder.
"""

import numpy as np
from transformer_encoder import TransformerEncoder, create_vocabulary, tokenize_sentence


def exemplo_1_basico():
    """Exemplo 1: Uso básico com uma frase simples."""
    print("\n" + "="*70)
    print("EXEMPLO 1: Uso Básico")
    print("="*70)
    
    # Criar vocabulário e encoder
    vocab_df, vocab_dict = create_vocabulary()
    encoder = TransformerEncoder(vocab_size=len(vocab_dict), d_model=64, n_layers=6)
    
    # Processar uma frase
    sentence = "o banco bloqueou meu cartao"
    token_ids = tokenize_sentence(sentence, vocab_dict)
    
    # Forward pass (sem verbose)
    encoder.forward = lambda ids: _forward_silent(encoder, ids)
    Z = encoder.forward(token_ids)
    
    print(f"\nFrase: '{sentence}'")
    print(f"Representação contextualizada Z: {Z.shape}")
    print(f"Cada palavra agora tem um vetor de {Z.shape[-1]} dimensões")
    

def _forward_silent(encoder, token_ids):
    """Forward pass silencioso (sem prints)."""
    X = encoder.embed(token_ids)
    for layer_weights in encoder.layers:
        X = encoder.encoder_layer(X, layer_weights)
    return X


def exemplo_2_multiplas_frases():
    """Exemplo 2: Processando múltiplas frases em batch."""
    print("\n" + "="*70)
    print("EXEMPLO 2: Processamento em Batch")
    print("="*70)
    
    vocab_df, vocab_dict = create_vocabulary()
    encoder = TransformerEncoder(vocab_size=len(vocab_dict), d_model=64)
    
    # Múltiplas frases
    sentences = [
        "o banco bloqueou cartao",
        "cliente solicitou novo emprestimo"
    ]
    
    # Tokenizar e criar batch (com padding)
    max_len = 0
    all_tokens = []
    for sent in sentences:
        tokens = tokenize_sentence(sent, vocab_dict)
        all_tokens.append(tokens)
        max_len = max(max_len, len(tokens))
    
    # Padding para deixar todas com o mesmo tamanho
    padded_tokens = []
    for tokens in all_tokens:
        padded = tokens + [vocab_dict['<PAD>']] * (max_len - len(tokens))
        padded_tokens.append(padded)
    
    batch_tokens = np.array(padded_tokens)
    
    print(f"\nFrases:")
    for i, sent in enumerate(sentences):
        print(f"  {i+1}. '{sent}'")
    
    print(f"\nBatch de tokens (com padding):")
    print(batch_tokens)
    
    # Processar batch
    embeddings = encoder.embedding_table[batch_tokens]
    output = encoder.encoder_layer(embeddings, encoder.layers[0])
    
    print(f"\nOutput shape: {output.shape}")
    print(f"  - Batch size: {output.shape[0]}")
    print(f"  - Sequência (com padding): {output.shape[1]}")
    print(f"  - Dimensão do modelo: {output.shape[2]}")


def exemplo_3_visualizar_atencao():
    """Exemplo 3: Visualizar scores de atenção."""
    print("\n" + "="*70)
    print("EXEMPLO 3: Visualizando Scores de Atenção")
    print("="*70)
    
    vocab_df, vocab_dict = create_vocabulary()
    encoder = TransformerEncoder(vocab_size=len(vocab_dict), d_model=64)
    
    # Processar frase
    sentence = "o banco bloqueou meu cartao"
    token_ids = tokenize_sentence(sentence, vocab_dict)
    words = sentence.split()
    
    # Obter embeddings
    X = encoder.embed(token_ids)
    
    # Calcular Q, K, V
    weights = encoder.layers[0]
    Q = np.matmul(X, weights['W_Q'])
    K = np.matmul(X, weights['W_K'])
    V = np.matmul(X, weights['W_V'])
    
    # Calcular atenção
    _, attention_weights = encoder.scaled_dot_product_attention(Q, K, V)
    
    print(f"\nFrase: '{sentence}'")
    print(f"\nMatriz de Atenção (cada linha = quanto cada palavra 'olha' para as outras):\n")
    
    # Cabeçalho
    print("       ", end="")
    for word in words:
        print(f"{word:>10}", end="")
    print()
    
    # Linhas
    attention = attention_weights[0]  # Primeiro item do batch
    for i, word in enumerate(words):
        print(f"{word:>6}:", end="")
        for j in range(len(words)):
            print(f"{attention[i, j]:>10.4f}", end="")
        print()
    
    print("\nInterpretação:")
    print("  - Valores maiores = mais atenção")
    print("  - Cada linha soma 1.0 (softmax)")


def exemplo_4_comparar_camadas():
    """Exemplo 4: Comparar saídas de diferentes camadas."""
    print("\n" + "="*70)
    print("EXEMPLO 4: Evolução através das Camadas")
    print("="*70)
    
    vocab_df, vocab_dict = create_vocabulary()
    encoder = TransformerEncoder(vocab_size=len(vocab_dict), d_model=64, n_layers=6)
    
    sentence = "o banco bloqueou meu cartao"
    token_ids = tokenize_sentence(sentence, vocab_dict)
    
    # Processar e salvar output de cada camada
    X = encoder.embed(token_ids)
    outputs = [X.copy()]
    
    for i, layer_weights in enumerate(encoder.layers):
        X = encoder.encoder_layer(X, layer_weights)
        outputs.append(X.copy())
    
    print(f"\nFrase: '{sentence}'")
    print(f"\nEstatísticas por camada (primeira palavra 'o'):\n")
    print("Camada    Média      Std      Min       Max")
    print("-" * 50)
    
    for i, output in enumerate(outputs):
        token_vec = output[0, 0, :]  # Vetor do primeiro token
        print(f"  {i:2d}    {np.mean(token_vec):>7.4f}  {np.std(token_vec):>7.4f}  "
              f"{np.min(token_vec):>8.4f}  {np.max(token_vec):>8.4f}")
    
    print("\nObservação:")
    print("  - Layer Norm mantém média ~0 e std ~1")
    print("  - Os valores mudam conforme o contexto é processado")


def exemplo_5_diferentes_configuracoes():
    """Exemplo 5: Testar diferentes configurações do modelo."""
    print("\n" + "="*70)
    print("EXEMPLO 5: Diferentes Configurações")
    print("="*70)
    
    vocab_df, vocab_dict = create_vocabulary()
    sentence = "o banco bloqueou meu cartao"
    token_ids = tokenize_sentence(sentence, vocab_dict)
    
    configs = [
        {"d_model": 32, "d_ff": 128, "n_layers": 2, "nome": "Pequeno"},
        {"d_model": 64, "d_ff": 256, "n_layers": 4, "nome": "Médio"},
        {"d_model": 128, "d_ff": 512, "n_layers": 6, "nome": "Grande"},
    ]
    
    print(f"\nFrase: '{sentence}'")
    print(f"\nComparando configurações:\n")
    print("Config     d_model  d_ff   Layers  Parâmetros (aprox.)")
    print("-" * 60)
    
    for config in configs:
        encoder = TransformerEncoder(
            vocab_size=len(vocab_dict),
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            n_layers=config['n_layers']
        )
        
        # Calcular número aproximado de parâmetros por camada
        params_per_layer = (
            3 * config['d_model'] * config['d_model'] +  # Q, K, V
            config['d_model'] * config['d_ff'] +          # W1
            config['d_ff'] * config['d_model'] +          # W2
            config['d_ff'] +                              # b1
            config['d_model'] +                           # b2
            2 * config['d_model']                         # gamma, beta (2x)
        )
        
        total_params = params_per_layer * config['n_layers']
        total_params += len(vocab_dict) * config['d_model']  # Embeddings
        
        print(f"{config['nome']:8s}   {config['d_model']:>4d}    {config['d_ff']:>4d}   "
              f"{config['n_layers']:>4d}    {total_params:>10,}")
    
    print("\nNota:")
    print("  - Mais parâmetros = maior capacidade, mas mais lento")
    print("  - d_model afeta muito o número de parâmetros")


def exemplo_6_analise_embeddings():
    """Exemplo 6: Analisar embeddings antes e depois."""
    print("\n" + "="*70)
    print("EXEMPLO 6: Análise de Embeddings")
    print("="*70)
    
    vocab_df, vocab_dict = create_vocabulary()
    encoder = TransformerEncoder(vocab_size=len(vocab_dict), d_model=64)
    
    sentence = "banco bloqueou banco"  # Palavra "banco" repetida
    token_ids = tokenize_sentence(sentence, vocab_dict)
    
    # Embeddings iniciais (antes do encoder)
    initial_embeddings = encoder.embed(token_ids)
    
    # Embeddings contextualizados (depois do encoder)
    contextual_embeddings = _forward_silent(encoder, token_ids)
    
    print(f"\nFrase: '{sentence}'")
    print("\nEmbeddings da palavra 'banco' (posições 0 e 2):")
    
    # Similaridade entre os dois "banco" nos embeddings iniciais
    vec1_initial = initial_embeddings[0, 0, :]
    vec2_initial = initial_embeddings[0, 2, :]
    
    similarity_initial = np.dot(vec1_initial, vec2_initial) / (
        np.linalg.norm(vec1_initial) * np.linalg.norm(vec2_initial)
    )
    
    print(f"\nSimilaridade ANTES do encoder (não-contextual):")
    print(f"  Cosseno: {similarity_initial:.6f}")
    print(f"  (Deve ser ~1.0 pois é a mesma palavra)")
    
    # Similaridade entre os dois "banco" nos embeddings contextualizados
    vec1_contextual = contextual_embeddings[0, 0, :]
    vec2_contextual = contextual_embeddings[0, 2, :]
    
    similarity_contextual = np.dot(vec1_contextual, vec2_contextual) / (
        np.linalg.norm(vec1_contextual) * np.linalg.norm(vec2_contextual)
    )
    
    print(f"\nSimilaridade DEPOIS do encoder (contextualizada):")
    print(f"  Cosseno: {similarity_contextual:.6f}")
    print(f"  (Pode ser diferente devido ao contexto diferente)")
    
    print("\nObservação:")
    print("  - Embeddings iniciais são idênticos para a mesma palavra")
    print("  - Embeddings contextualizados capturam a posição e contexto")


def main():
    """Executa todos os exemplos."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*10 + "EXEMPLOS DE USO - TRANSFORMER ENCODER" + " "*21 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    exemplo_1_basico()
    exemplo_2_multiplas_frases()
    exemplo_3_visualizar_atencao()
    exemplo_4_comparar_camadas()
    exemplo_5_diferentes_configuracoes()
    exemplo_6_analise_embeddings()
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*20 + "EXEMPLOS CONCLUÍDOS!" + " "*27 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()
