"""
Exemplos de uso do Transformer Encoder

Este arquivo demonstra diferentes aplicações e casos de uso
da implementação do Transformer Encoder.
"""

import numpy as np
from transformer_encoder import TransformerEncoder, create_vocabulary, tokenize_sentence


def exemplo_1_basico():
    """
    Exemplo 1: Uso básico com uma frase simples
    """
    print("\n" + "="*80)
    print("EXEMPLO 1: Uso Básico")
    print("="*80)
    
    vocab_df, vocab_dict = create_vocabulary()
    vocab_size = len(vocab_dict)
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=6
    )
    
    frase = "o banco bloqueou meu cartao"
    tokens = frase.split()
    ids = tokenize_sentence(frase, vocab_dict)
    
    print(f"Frase: '{frase}'")
    print(f"IDs: {ids}")
    
    Z = encoder.forward(ids)
    
    print(f"\nShape de saída: {Z.shape}")
    print(f"Primeiro token representação (5 primeiros valores):")
    print(Z[0, 0, :5])


def exemplo_2_analise_atencao():
    """
    Exemplo 2: Análise dos pesos de atenção
    """
    print("\n" + "="*80)
    print("EXEMPLO 2: Análise dos Pesos de Atenção")
    print("="*80)
    
    vocab_df, vocab_dict = create_vocabulary()
    vocab_size = len(vocab_dict)
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=32,  # Menor para facilitar visualização
        num_layers=2   # Menos camadas para este exemplo
    )
    
    # Frase curta para visualização
    frase = "o banco bloqueou"
    tokens = frase.split()
    ids = tokenize_sentence(frase, vocab_dict)
    
    print(f"Frase: '{frase}'")
    print(f"Tokens: {tokens}")
    
    # Executar forward pass e capturar atenção manualmente
    input_ids = np.array([ids])
    batch_size, seq_len = input_ids.shape
    
    X = encoder.embedding_table[input_ids]
    pe = encoder.positional_encoding(seq_len, encoder.d_model)
    X = X + pe
    
    # Usar forward para obter representação com atenção
    Z, all_attentions = encoder.forward(ids)
    
    if all_attentions and len(all_attentions) > 0:
        attention_weights = all_attentions[0]  # Primeira camada
        # Média das cabeças de atenção
        attention_weights = attention_weights.mean(axis=1)  # (batch, seq, seq)
    else:
        print("Aviso: Atenção não disponível")
        return
    
    print(f"\nPesos de Atenção (Camada 1):")
    print("Matrix de atenção (cada linha mostra onde o token presta atenção):")
    print("           ", end="")
    for token in tokens:
        print(f"{token:>10}", end="")
    print()
    
    for i, token_i in enumerate(tokens):
        print(f"{token_i:>10}", end=" ")
        for j in range(len(tokens)):
            print(f"{attention_weights[0, i, j]:>9.4f}", end=" ")
        print()
    
    print("\nInterpretação:")
    max_attention_idx = np.argmax(attention_weights[0], axis=1)
    for i, token in enumerate(tokens):
        max_token = tokens[max_attention_idx[i]]
        max_score = attention_weights[0, i, max_attention_idx[i]]
        print(f"  - '{token}' presta mais atenção em '{max_token}' ({max_score:.4f})")


def exemplo_3_comparacao_camadas():
    """
    Exemplo 3: Comparação das representações através das camadas
    """
    print("\n" + "="*80)
    print("EXEMPLO 3: Evolução das Representações através das Camadas")
    print("="*80)
    
    vocab_df, vocab_dict = create_vocabulary()
    vocab_size = len(vocab_dict)
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=6
    )
    
    frase = "o banco bloqueou meu cartao"
    tokens = frase.split()
    ids = tokenize_sentence(frase, vocab_dict)
    
    print(f"Frase: '{frase}'")
    
    # Executar forward pass
    print(f"\nFrase: '{frase}'")
    print(f"Tokens: {len(tokens)}")
    
    Z, attentions = encoder.forward(ids)
    
    print(f"\nRepresentação final:")
    print(f"  Shape: {Z.shape}")
    print(f"  Norma L2: {np.linalg.norm(Z):.6f}")
    print(f"  Média: {Z.mean():.6f}")
    print(f"  Desvio padrão: {Z.std():.6f}")
    
    print(f"\nPesos de atenção capturados: {len(attentions)} camadas")


def exemplo_4_batch_processing():
    """
    Exemplo 4: Processamento em batch
    """
    print("\n" + "="*80)
    print("EXEMPLO 4: Processamento em Batch")
    print("="*80)
    
    vocab_df, vocab_dict = create_vocabulary()
    vocab_size = len(vocab_dict)
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=4  # Menos camadas para este exemplo
    )
    
    # Múltiplas frases
    frases = [
        "o banco bloqueou meu cartao",
        "o cliente solicitou novo cartao",
        "o banco bloqueou"
    ]
    
    print("Processando múltiplas frases em batch:\n")
    
    for i, frase in enumerate(frases):
        tokens = frase.split()
        ids = tokenize_sentence(frase, vocab_dict)
        
        Z, _ = encoder.forward(ids)
        
        print(f"Frase {i+1}: '{frase}'")
        print(f"  - Tokens: {len(tokens)}")
        print(f"  - Shape de saída: {Z.shape}")
        print(f"  - Média dos valores: {Z.mean():.6f}")
        print(f"  - Desvio padrão: {Z.std():.6f}")
        print()


def exemplo_5_similaridade():
    """
    Exemplo 5: Calcular similaridade entre representações de palavras
    """
    print("\n" + "="*80)
    print("EXEMPLO 5: Similaridade entre Representações")
    print("="*80)
    
    vocab_df, vocab_dict = create_vocabulary()
    vocab_size = len(vocab_dict)
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=6
    )
    
    frase = "o banco bloqueou meu cartao"
    tokens = frase.split()
    ids = tokenize_sentence(frase, vocab_dict)
    
    print(f"Frase: '{frase}'")
    
    Z, _ = encoder.forward(ids)
    
    # Calcular similaridade coseno entre tokens
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print("\nMatriz de Similaridade Coseno:")
    print("            ", end="")
    for token in tokens:
        print(f"{token[:8]:>10}", end="")
    print()
    
    for i, token_i in enumerate(tokens):
        print(f"{token_i[:10]:>10}", end="  ")
        for j in range(len(tokens)):
            sim = cosine_similarity(Z[0, i], Z[0, j])
            print(f"{sim:>9.4f}", end=" ")
        print()
    
    # Encontrar pares mais similares
    print("\nPares mais similares (excluindo diagonal):")
    similarities = []
    for i in range(len(tokens)):
        for j in range(i+1, len(tokens)):
            sim = cosine_similarity(Z[0, i], Z[0, j])
            similarities.append((tokens[i], tokens[j], sim))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    for i, (token1, token2, sim) in enumerate(similarities[:5]):
        print(f"  {i+1}. '{token1}' ↔ '{token2}': {sim:.4f}")


def exemplo_6_dimensoes():
    """
    Exemplo 6: Experimentar com diferentes dimensões
    """
    print("\n" + "="*80)
    print("EXEMPLO 6: Diferentes Configurações de Dimensão")
    print("="*80)
    
    vocab_df, vocab_dict = create_vocabulary()
    vocab_size = len(vocab_dict)
    
    frase = "o banco bloqueou meu cartao"
    tokens = frase.split()
    ids = tokenize_sentence(frase, vocab_dict)
    
    configuracoes = [
        {"d_model": 32, "d_ff": 128, "num_layers": 2},
        {"d_model": 64, "d_ff": 256, "num_layers": 4},
        {"d_model": 128, "d_ff": 512, "num_layers": 6},
    ]
    
    print(f"Frase: '{frase}'\n")
    print(f"{'d_model':<10} {'d_ff':<10} {'num_layers':<12} {'Output Shape':<20} {'Tempo':<15}")
    print("-" * 80)
    
    import time
    
    for config in configuracoes:
        encoder = TransformerEncoder(
            vocab_size=vocab_size,
            **config
        )
        
        start = time.time()
        Z, _ = encoder.forward(ids)
        elapsed = time.time() - start
        
        print(f"{config['d_model']:<10} {config['d_ff']:<10} {config['num_layers']:<12} {str(Z.shape):<20} {elapsed:.4f}s")


def menu():
    """
    Menu interativo para executar os exemplos
    """
    print("\n" + "="*80)
    print(" "*25 + "EXEMPLOS DO TRANSFORMER ENCODER")
    print("="*80)
    print("\nEscolha um exemplo:")
    print("  1. Uso Básico")
    print("  2. Análise dos Pesos de Atenção")
    print("  3. Evolução das Representações")
    print("  4. Processamento em Batch")
    print("  5. Similaridade entre Palavras")
    print("  6. Diferentes Configurações")
    print("  7. Executar TODOS os exemplos")
    print("  0. Sair")
    
    return input("\nDigite o número do exemplo: ").strip()


def main():
    """
    Função principal
    """
    while True:
        escolha = menu()
        
        if escolha == "0":
            print("\nEncerrando...")
            break
        elif escolha == "1":
            exemplo_1_basico()
        elif escolha == "2":
            exemplo_2_analise_atencao()
        elif escolha == "3":
            exemplo_3_comparacao_camadas()
        elif escolha == "4":
            exemplo_4_batch_processing()
        elif escolha == "5":
            exemplo_5_similaridade()
        elif escolha == "6":
            exemplo_6_dimensoes()
        elif escolha == "7":
            print("\nExecutando todos os exemplos...")
            exemplo_1_basico()
            exemplo_2_analise_atencao()
            exemplo_3_comparacao_camadas()
            exemplo_4_batch_processing()
            exemplo_5_similaridade()
            exemplo_6_dimensoes()
            print("\n" + "="*80)
            print("Todos os exemplos foram executados!")
            print("="*80)
        else:
            print("\nOpção inválida! Tente novamente.")
        
        input("\nPressione ENTER para continuar...")


if __name__ == "__main__":
    # Se executar sem argumentos, mostra menu interativo
    import sys
    if len(sys.argv) == 1:
        main()
    else:
        # Executar todos os exemplos automaticamente
        print("Executando todos os exemplos automaticamente...")
        exemplo_1_basico()
        exemplo_2_analise_atencao()
        exemplo_3_comparacao_camadas()
        exemplo_4_batch_processing()
        exemplo_5_similaridade()
        exemplo_6_dimensoes()
