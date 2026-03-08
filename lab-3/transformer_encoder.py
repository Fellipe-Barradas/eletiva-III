"""
Transformer Encoder - Implementação Completa com Multi-Head Attention
Disciplina: Tópicos em Inteligência Artificial – 2026.1
Professor: Prof. Dimmy Magalhães

Implementação COMPLETA baseada em "Attention Is All You Need" (Vaswani et al., 2017)
Inclui: Multi-Head Attention, Positional Encoding, 6 camadas, d_model=512
"""

import numpy as np
import pandas as pd


class PositionalEncoding:
    """
    Implementa Positional Encoding usando funções seno e cosseno.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: Dimensão do modelo
            max_len: Comprimento máximo da sequência
        """
        self.d_model = d_model
        
        # Criar matriz de positional encoding
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        
        # Calcular div_term
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Aplicar seno e cosseno
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def encode(self, x):
        """
        Adiciona positional encoding ao embedding.
        
        Args:
            x: Tensor de embeddings (batch_size, seq_len, d_model)
            
        Returns:
            x + positional encoding
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]


class MultiHeadAttention:
    """
    Implementa Multi-Head Attention com h cabeças paralelas.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    onde head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: Dimensão do modelo (deve ser divisível por num_heads)
            num_heads: Número de cabeças de atenção
        """
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimensão de cada cabeça
        
        # Matrizes de projeção para Q, K, V (compartilhadas entre todas as cabeças)
        self.W_Q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_K = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_V = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
        # Matriz de projeção final (output)
        self.W_O = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def split_heads(self, x):
        """
        Divide a última dimensão em (num_heads, d_k).
        Reorganiza para (batch_size, num_heads, seq_len, d_k).
        
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape: (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x):
        """
        Inverso de split_heads.
        
        Args:
            x: (batch_size, num_heads, seq_len, d_k)
            
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        
        # Transpose: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        x = x.transpose(0, 2, 1, 3)
        
        # Reshape: (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def scaled_dot_product_attention(self, Q, K, V):
        """
        Atenção para uma única cabeça.
        
        Args:
            Q, K, V: (batch_size, num_heads, seq_len, d_k)
            
        Returns:
            output: (batch_size, num_heads, seq_len, d_k)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # Scores: (batch, heads, seq_len, d_k) @ (batch, heads, d_k, seq_len)
        #       = (batch, heads, seq_len, seq_len)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
        
        # Scaling
        scores = scores / np.sqrt(self.d_k)
        
        # Softmax
        attention_weights = self.softmax(scores, axis=-1)
        
        # Aplicar atenção aos valores
        # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, d_k)
        # = (batch, heads, seq_len, d_k)
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x, axis=-1):
        """Implementação estável do softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def forward(self, X):
        """
        Forward pass do Multi-Head Attention.
        
        Args:
            X: (batch_size, seq_len, d_model)
            
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # Projeções lineares
        Q = np.matmul(X, self.W_Q)  # (batch, seq_len, d_model)
        K = np.matmul(X, self.W_K)
        V = np.matmul(X, self.W_V)
        
        # Dividir em múltiplas cabeças
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Aplicar atenção para cada cabeça
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V)
        
        # Combinar cabeças
        attention_output = self.combine_heads(attention_output)  # (batch, seq_len, d_model)
        
        # Projeção final
        output = np.matmul(attention_output, self.W_O)
        
        return output, attention_weights


class LayerNormalization:
    """
    Implementa Layer Normalization com parâmetros aprendíveis.
    """
    
    def __init__(self, d_model, epsilon=1e-6):
        """
        Args:
            d_model: Dimensão do modelo
            epsilon: Valor pequeno para estabilidade numérica
        """
        self.epsilon = epsilon
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x):
        """
        Aplica layer normalization.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor normalizado
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * x_norm + self.beta


class FeedForwardNetwork:
    """
    Implementa a rede Feed-Forward de duas camadas.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model, d_ff):
        """
        Args:
            d_model: Dimensão do modelo
            d_ff: Dimensão da camada intermediária (geralmente 4 * d_model)
        """
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        """
        Forward pass do FFN.
        
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Primeira camada linear + ReLU
        hidden = np.matmul(x, self.W1) + self.b1
        hidden = np.maximum(0, hidden)  # ReLU
        
        # Segunda camada linear
        output = np.matmul(hidden, self.W2) + self.b2
        
        return output


class EncoderLayer:
    """
    Uma única camada do Transformer Encoder.
    
    Contém:
    1. Multi-Head Self-Attention + Add & Norm
    2. Feed-Forward Network + Add & Norm
    """
    
    def __init__(self, d_model, num_heads, d_ff):
        """
        Args:
            d_model: Dimensão do modelo
            num_heads: Número de cabeças de atenção
            d_ff: Dimensão da FFN
        """
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = LayerNormalization(d_model)
    
    def forward(self, x):
        """
        Forward pass da camada do encoder.
        
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: Pesos de atenção
        """
        # Sub-camada 1: Multi-Head Attention + Add & Norm
        attn_output, attention_weights = self.attention.forward(x)
        x = self.norm1.forward(x + attn_output)
        
        # Sub-camada 2: FFN + Add & Norm
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output)
        
        return x, attention_weights


class TransformerEncoder:
    """
    Transformer Encoder completo com N camadas.
    
    Implementação próxima ao paper original:
    - d_model = 512
    - num_heads = 8
    - d_ff = 2048
    - N = 6 camadas
    """
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, 
                 num_layers=6, max_seq_len=5000):
        """
        Args:
            vocab_size: Tamanho do vocabulário
            d_model: Dimensão do modelo
            num_heads: Número de cabeças de atenção
            d_ff: Dimensão da FFN
            num_layers: Número de camadas do encoder
            max_seq_len: Comprimento máximo da sequência
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        
        # Tabela de embeddings
        self.embedding_table = np.random.randn(vocab_size, d_model) * 0.1
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Camadas do encoder
        self.layers = []
        for _ in range(num_layers):
            layer = EncoderLayer(d_model, num_heads, d_ff)
            self.layers.append(layer)
        
        # Normalização final
        self.final_norm = LayerNormalization(d_model)
    
    def embed(self, token_ids):
        """
        Converte IDs de tokens em embeddings + positional encoding.
        
        Args:
            token_ids: Lista ou array de IDs
            
        Returns:
            Embeddings com positional encoding (batch_size, seq_len, d_model)
        """
        # Converter para array e adicionar batch dimension
        if isinstance(token_ids, list):
            token_ids = np.array(token_ids)
        
        if len(token_ids.shape) == 1:
            token_ids = token_ids[np.newaxis, :]
        
        # Buscar embeddings
        embeddings = self.embedding_table[token_ids]
        
        # Adicionar positional encoding
        embeddings = self.pos_encoding.encode(embeddings)
        
        return embeddings
    
    def forward(self, token_ids, return_attention=False):
        """
        Forward pass completo através de todas as camadas.
        
        Args:
            token_ids: Lista de IDs de tokens
            return_attention: Se True, retorna os pesos de atenção
            
        Returns:
            Z: Vetor final (batch_size, seq_len, d_model)
            attention_weights_list: Lista de pesos de atenção (opcional)
        """
        # Embeddings + Positional Encoding
        x = self.embed(token_ids)
        
        print(f"Shape após embeddings + pos encoding: {x.shape}")
        
        # Passar por todas as camadas
        attention_weights_list = []
        
        for i, layer in enumerate(self.layers):
            x, attention_weights = layer.forward(x)
            attention_weights_list.append(attention_weights)
            print(f"Shape após camada {i+1}: {x.shape}")
        
        # Normalização final
        x = self.final_norm.forward(x)
        
        print(f"\nShape final do vetor Z: {x.shape}")
        
        if return_attention:
            return x, attention_weights_list
        return x


def create_vocabulary():
    """
    Cria um vocabulário simulado.
    
    Returns:
        DataFrame e dicionário com mapeamento palavra -> ID
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
        "foi": 9,
        "aprovado": 10,
        "conta": 11,
        "corrente": 12,
        "<PAD>": 13,
        "<UNK>": 14
    }
    
    vocab_df = pd.DataFrame(list(vocab_dict.items()), columns=['palavra', 'id'])
    return vocab_df, vocab_dict


def tokenize_sentence(sentence, vocab_dict):
    """
    Converte uma frase em lista de IDs.
    
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
    Demonstração do Transformer Encoder completo.
    """
    print("=" * 80)
    print("TRANSFORMER ENCODER - IMPLEMENTAÇÃO COMPLETA")
    print("Multi-Head Attention (h=8) + Positional Encoding")
    print("Tópicos em Inteligência Artificial – 2026.1")
    print("=" * 80)
    print()
    
    # ==================== PASSO 1: PREPARAÇÃO ====================
    print("PASSO 1: Preparação dos Dados")
    print("-" * 80)
    
    vocab_df, vocab_dict = create_vocabulary()
    vocab_size = len(vocab_dict)
    
    print("\nVocabulário:")
    print(vocab_df.head(10))
    print(f"...\nTotal: {vocab_size} palavras")
    
    sentence = "o banco bloqueou meu cartao"
    token_ids = tokenize_sentence(sentence, vocab_dict)
    
    print(f"\nFrase: '{sentence}'")
    print(f"Tokens: {token_ids}")
    print()
    
    # ==================== PASSO 2 e 3: CONSTRUIR ENCODER ====================
    print("PASSO 2 e 3: Construindo o Transformer Encoder")
    print("-" * 80)
    
    # Configuração (pode usar d_model=64 para CPU ou 512 como no paper)
    use_full_size = False  # Mude para True para usar dimensões do paper
    
    if use_full_size:
        d_model = 512
        d_ff = 2048
        num_heads = 8
        print("\n⚠️  Usando configuração COMPLETA do paper (pode ser lento)")
    else:
        d_model = 64
        d_ff = 256
        num_heads = 8
        print("\n✓ Usando configuração OTIMIZADA para CPU")
    
    print(f"\nParâmetros:")
    print(f"  - d_model: {d_model}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - d_k (por cabeça): {d_model // num_heads}")
    print(f"  - d_ff: {d_ff}")
    print(f"  - Camadas: 6")
    print()
    
    # Criar encoder
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=6
    )
    
    print(f"Tabela de Embeddings: {encoder.embedding_table.shape}")
    print(f"Positional Encoding: criado para sequências até 5000 tokens")
    print()
    
    # ==================== FORWARD PASS ====================
    print("Executando Forward Pass:")
    print("-" * 80)
    
    Z, attention_weights_list = encoder.forward(token_ids, return_attention=True)
    
    print()
    print("=" * 80)
    print("RESULTADO FINAL")
    print("=" * 80)
    
    print(f"\nVetor Z (representação contextualizada):")
    print(f"  - Shape: {Z.shape}")
    print(f"  - Batch: {Z.shape[0]}")
    print(f"  - Tokens: {Z.shape[1]}")
    print(f"  - Dimensão: {Z.shape[2]}")
    
    # Validação
    expected_shape = (1, len(token_ids), d_model)
    assert Z.shape == expected_shape, f"Shape incorreto: esperado {expected_shape}, obtido {Z.shape}"
    
    print("\n✓ Validação passou!")
    
    # Estatísticas
    print(f"\nEstatísticas:")
    print(f"  - Média: {np.mean(Z):.6f}")
    print(f"  - Std: {np.std(Z):.6f}")
    print(f"  - Min: {np.min(Z):.6f}")
    print(f"  - Max: {np.max(Z):.6f}")
    
    # Análise de atenção
    print(f"\nPesos de Atenção:")
    print(f"  - {len(attention_weights_list)} camadas")
    print(f"  - Shape de cada camada: {attention_weights_list[0].shape}")
    print(f"    (batch, num_heads, seq_len, seq_len)")
    
    # Mostrar atenção média da primeira camada
    avg_attention = np.mean(attention_weights_list[0][0], axis=0)  # Média sobre as cabeças
    
    print(f"\nAtenção média (camada 1, média de {num_heads} cabeças):")
    words = sentence.split()
    
    print("\n       ", end="")
    for word in words:
        print(f"{word:>10}", end="")
    print()
    
    for i, word in enumerate(words):
        print(f"{word:>6}:", end="")
        for j in range(len(words)):
            print(f"{avg_attention[i, j]:>10.4f}", end="")
        print()
    
    print()
    print("=" * 80)
    print("✓ Implementação completa com sucesso!")
    print("=" * 80)


if __name__ == "__main__":
    main()
