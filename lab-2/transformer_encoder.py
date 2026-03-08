"""
Transformer Encoder - Implementação From Scratch
Disciplina: Tópicos em Inteligência Artificial – 2026.1
Professor: Prof. Dimmy Magalhães

Baseado no artigo: "Attention Is All You Need" (Vaswani et al., 2017)
Ferramentas: Python 3.x, numpy, pandas
"""

import numpy as np
import pandas as pd


class TransformerEncoder:
    """
    Implementação completa do Transformer Encoder com N=6 camadas.
    """
    
    def __init__(self, vocab_size, d_model=64, d_ff=256, d_k=64, n_layers=6, epsilon=1e-6):
        """
        Inicializa o Transformer Encoder.
        
        Args:
            vocab_size: Tamanho do vocabulário
            d_model: Dimensão do modelo (512 no paper original, 64 para fins práticos)
            d_ff: Dimensão da camada feed-forward (2048 no paper, 256 aqui)
            d_k: Dimensão das chaves para scaling (√d_k)
            n_layers: Número de camadas do encoder (N=6)
            epsilon: Valor pequeno para evitar divisão por zero na normalização
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.n_layers = n_layers
        self.epsilon = epsilon
        
        # Tabela de Embeddings
        self.embedding_table = np.random.randn(vocab_size, d_model) * 0.1
        
        # Inicializar pesos para cada camada
        self.layers = []
        for _ in range(n_layers):
            layer_weights = self._initialize_layer_weights()
            self.layers.append(layer_weights)
    
    def _initialize_layer_weights(self):
        """
        Inicializa os pesos de uma única camada do encoder.
        """
        # Matrizes de projeção para Q, K, V (Self-Attention)
        W_Q = np.random.randn(self.d_model, self.d_model) * np.sqrt(2.0 / self.d_model)
        W_K = np.random.randn(self.d_model, self.d_model) * np.sqrt(2.0 / self.d_model)
        W_V = np.random.randn(self.d_model, self.d_model) * np.sqrt(2.0 / self.d_model)
        
        # Pesos para Feed-Forward Network
        W1 = np.random.randn(self.d_model, self.d_ff) * np.sqrt(2.0 / self.d_model)
        b1 = np.zeros(self.d_ff)
        W2 = np.random.randn(self.d_ff, self.d_model) * np.sqrt(2.0 / self.d_ff)
        b2 = np.zeros(self.d_model)
        
        # Parâmetros para Layer Normalization (gamma e beta)
        gamma_1 = np.ones(self.d_model)
        beta_1 = np.zeros(self.d_model)
        gamma_2 = np.ones(self.d_model)
        beta_2 = np.zeros(self.d_model)
        
        return {
            'W_Q': W_Q, 'W_K': W_K, 'W_V': W_V,
            'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
            'gamma_1': gamma_1, 'beta_1': beta_1,
            'gamma_2': gamma_2, 'beta_2': beta_2
        }
    
    def softmax(self, x, axis=-1):
        """
        Implementação própria da função Softmax usando exponenciais.
        
        Args:
            x: Array de entrada
            axis: Eixo ao longo do qual aplicar softmax
            
        Returns:
            Array após aplicação do softmax
        """
        # Subtrair o máximo para estabilidade numérica
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
        return exp_x / sum_exp
    
    def scaled_dot_product_attention(self, Q, K, V):
        """
        Implementação do Scaled Dot-Product Attention.
        
        Equação: Attention(Q, K, V) = softmax(QK^T / √d_k)V
        
        Args:
            Q: Query matrix (batch_size, seq_len, d_model)
            K: Key matrix (batch_size, seq_len, d_model)
            V: Value matrix (batch_size, seq_len, d_model)
            
        Returns:
            Output após atenção e matriz de scores
        """
        # Passo 1: Calcular o produto escalar entre Q e K^T
        # Q @ K^T -> (batch, seq_len, d_model) @ (batch, d_model, seq_len)
        scores = np.matmul(Q, K.transpose(0, 2, 1))
        
        # Passo 2: Scaling - dividir por √d_k
        scores = scores / np.sqrt(self.d_k)
        
        # Passo 3: Aplicar Softmax
        attention_weights = self.softmax(scores, axis=-1)
        
        # Passo 4: Multiplicar pelos valores V
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def self_attention(self, X, weights):
        """
        Aplica Self-Attention completo com as projeções Q, K, V.
        
        Args:
            X: Tensor de entrada (batch_size, seq_len, d_model)
            weights: Dicionário com os pesos da camada
            
        Returns:
            Output após self-attention
        """
        # Passo 1: Gerar Q, K, V através de projeções lineares
        Q = np.matmul(X, weights['W_Q'])  # (batch, seq_len, d_model)
        K = np.matmul(X, weights['W_K'])  # (batch, seq_len, d_model)
        V = np.matmul(X, weights['W_V'])  # (batch, seq_len, d_model)
        
        # Passo 2: Aplicar Scaled Dot-Product Attention
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V)
        
        return attention_output
    
    def layer_norm(self, x, gamma, beta):
        """
        Implementação da Layer Normalization.
        
        Normaliza na dimensão dos features (último eixo).
        
        Args:
            x: Tensor de entrada
            gamma: Parâmetro de escala
            beta: Parâmetro de deslocamento
            
        Returns:
            Tensor normalizado
        """
        # Calcular média e variância na dimensão dos features (axis=-1)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalizar: (x - mean) / √(var + epsilon)
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        
        # Aplicar escala e deslocamento aprendíveis
        return gamma * x_norm + beta
    
    def feed_forward_network(self, x, weights):
        """
        Implementação da Feed-Forward Network (FFN).
        
        Equação: FFN(x) = max(0, xW1 + b1)W2 + b2
        
        Args:
            x: Tensor de entrada (batch_size, seq_len, d_model)
            weights: Dicionário com os pesos da camada
            
        Returns:
            Output após FFN
        """
        # Primeira transformação linear: d_model -> d_ff
        hidden = np.matmul(x, weights['W1']) + weights['b1']
        
        # Ativação ReLU: max(0, x)
        hidden = np.maximum(0, hidden)
        
        # Segunda transformação linear: d_ff -> d_model
        output = np.matmul(hidden, weights['W2']) + weights['b2']
        
        return output
    
    def encoder_layer(self, X, layer_weights):
        """
        Processa uma única camada do encoder.
        
        Fluxo:
        1. X_att = SelfAttention(X)
        2. X_norm1 = LayerNorm(X + X_att)  # Residual + Norm
        3. X_ffn = FFN(X_norm1)
        4. X_out = LayerNorm(X_norm1 + X_ffn)  # Residual + Norm
        
        Args:
            X: Tensor de entrada (batch_size, seq_len, d_model)
            layer_weights: Pesos da camada
            
        Returns:
            Output da camada (batch_size, seq_len, d_model)
        """
        # Sub-camada 1: Multi-Head Self-Attention + Residual + Norm
        X_att = self.self_attention(X, layer_weights)
        X_norm1 = self.layer_norm(X + X_att, layer_weights['gamma_1'], layer_weights['beta_1'])
        
        # Sub-camada 2: Feed-Forward Network + Residual + Norm
        X_ffn = self.feed_forward_network(X_norm1, layer_weights)
        X_out = self.layer_norm(X_norm1 + X_ffn, layer_weights['gamma_2'], layer_weights['beta_2'])
        
        return X_out
    
    def embed(self, token_ids):
        """
        Converte IDs de tokens em vetores de embedding.
        
        Args:
            token_ids: Lista ou array de IDs de tokens
            
        Returns:
            Tensor de embeddings (batch_size, seq_len, d_model)
        """
        # Adicionar dimensão de batch se necessário
        if isinstance(token_ids, list):
            token_ids = np.array(token_ids)
        
        if len(token_ids.shape) == 1:
            token_ids = token_ids[np.newaxis, :]  # (1, seq_len)
        
        # Buscar embeddings da tabela
        embeddings = self.embedding_table[token_ids]
        
        return embeddings
    
    def forward(self, token_ids):
        """
        Forward pass completo através de todas as N=6 camadas do encoder.
        
        Args:
            token_ids: Lista de IDs de tokens da frase de entrada
            
        Returns:
            Vetor Z final (batch_size, seq_len, d_model)
        """
        # Passo 1: Converter tokens em embeddings
        X = self.embed(token_ids)
        
        print(f"Shape inicial após embeddings: {X.shape}")
        
        # Passo 2: Passar por todas as N camadas do encoder
        for i, layer_weights in enumerate(self.layers):
            X = self.encoder_layer(X, layer_weights)
            print(f"Shape após camada {i+1}: {X.shape}")
        
        print(f"\nShape final do vetor Z: {X.shape}")
        return X


def create_vocabulary():
    """
    Cria um DataFrame simulando um vocabulário simples.
    
    Returns:
        DataFrame com mapeamento palavra -> ID
    """
    vocab_dict = {
        "o": 0,
        "banco": 1,
        "bloqueou": 2,
        "cartao": 3,
        "meu": 4,
        "cliente": 5,
        "solicitou": 6,
        "novo": 7,
        "emprestimo": 8,
        "<PAD>": 9,  # Token de padding
        "<UNK>": 10  # Token desconhecido
    }
    
    vocab_df = pd.DataFrame(list(vocab_dict.items()), columns=['palavra', 'id'])
    return vocab_df, vocab_dict


def tokenize_sentence(sentence, vocab_dict):
    """
    Converte uma frase em lista de IDs de tokens.
    
    Args:
        sentence: String com a frase
        vocab_dict: Dicionário de vocabulário
        
    Returns:
        Lista de IDs de tokens
    """
    words = sentence.lower().split()
    token_ids = [vocab_dict.get(word, vocab_dict['<UNK>']) for word in words]
    return token_ids


def main():
    """
    Função principal que demonstra o funcionamento do Transformer Encoder.
    """
    print("=" * 70)
    print("TRANSFORMER ENCODER - IMPLEMENTAÇÃO FROM SCRATCH")
    print("Tópicos em Inteligência Artificial – 2026.1")
    print("=" * 70)
    print()
    
    # ==================== PASSO 1: PREPARAÇÃO DOS DADOS ====================
    print("PASSO 1: Preparação dos Dados")
    print("-" * 70)
    
    # Criar vocabulário
    vocab_df, vocab_dict = create_vocabulary()
    vocab_size = len(vocab_dict)
    
    print("\nVocabulário criado:")
    print(vocab_df)
    print(f"\nTamanho do vocabulário: {vocab_size}")
    
    # Definir frase de entrada
    sentence = "o banco bloqueou meu cartao"
    print(f"\nFrase de entrada: '{sentence}'")
    
    # Tokenizar a frase
    token_ids = tokenize_sentence(sentence, vocab_dict)
    print(f"IDs dos tokens: {token_ids}")
    print()
    
    # ==================== PASSO 2 e 3: CONSTRUIR O ENCODER ====================
    print("PASSO 2 e 3: Construindo e Executando o Transformer Encoder")
    print("-" * 70)
    
    # Parâmetros do modelo
    d_model = 64  # 512 no paper original
    d_ff = 256    # 2048 no paper original
    n_layers = 6  # 6 camadas como no paper
    
    print(f"\nParâmetros do modelo:")
    print(f"  - d_model (dimensão do modelo): {d_model}")
    print(f"  - d_ff (dimensão FFN): {d_ff}")
    print(f"  - N (número de camadas): {n_layers}")
    print()
    
    # Inicializar o Transformer Encoder
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        d_k=d_model,
        n_layers=n_layers
    )
    
    print("Tabela de Embeddings inicializada com dimensões:", encoder.embedding_table.shape)
    print()
    
    # ==================== FORWARD PASS ====================
    print("Executando Forward Pass através das 6 camadas:")
    print("-" * 70)
    
    # Executar o forward pass
    Z = encoder.forward(token_ids)
    
    print()
    print("=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    print(f"\nVetor Z final (representação contextualizada):")
    print(f"  - Shape: {Z.shape}")
    print(f"  - Batch Size: {Z.shape[0]}")
    print(f"  - Sequence Length: {Z.shape[1]}")
    print(f"  - Model Dimension: {Z.shape[2]}")
    print()
    
    # Mostrar o vetor Z para o primeiro token
    print("Representação do primeiro token ('o'):")
    print(f"  - Primeiras 10 dimensões: {Z[0, 0, :10]}")
    print()
    
    # Validação de sanidade
    assert Z.shape == (1, len(token_ids), d_model), \
        f"Erro: Shape esperado (1, {len(token_ids)}, {d_model}), obtido {Z.shape}"
    
    print("✓ Validação de sanidade passou: dimensões corretas!")
    print()
    
    # Estatísticas do vetor Z
    print("Estatísticas do vetor Z:")
    print(f"  - Média: {np.mean(Z):.6f}")
    print(f"  - Desvio padrão: {np.std(Z):.6f}")
    print(f"  - Mínimo: {np.min(Z):.6f}")
    print(f"  - Máximo: {np.max(Z):.6f}")
    print()
    
    print("=" * 70)
    print("Implementação concluída com sucesso!")
    print("=" * 70)


if __name__ == "__main__":
    main()
