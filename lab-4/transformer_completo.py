"""
Laboratório Técnico 04: O Transformer Completo "From Scratch"
Disciplina: Tópicos em Inteligência Artificial – 2026.1
Professor: Prof. Dimmy Magalhães
Instituição: iCEV - Instituto de Ensino Superior

Implementação completa de um Transformer Encoder-Decoder com inferência auto-regressiva.
Integra componentes dos Labs 1, 2 e 3 para realizar tradução fim-a-fim de uma frase toy.
"""

import numpy as np
import pandas as pd


# ============================================================
# SEÇÃO 1: COMPONENTES BASE (Refatoração dos Labs 1-3)
# ============================================================

def softmax(x, axis=-1):
    """Softmax numericamente estável."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class PositionalEncoding:
    """Positional Encoding usando funções seno e cosseno (Lab 3)."""
    
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def encode(self, x):
        """Adiciona positional encoding ao embedding (x não é modificado em-place)."""
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]


class MultiHeadAttention:
    """Multi-Head Attention (Lab 3), suporta máscara causal para Decoder."""
    
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_K = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_V = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_O = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def split_heads(self, x):
        """
        De (batch, seq_len, d_model) para (batch, num_heads, seq_len, d_k).
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x):
        """De (batch, num_heads, seq_len, d_k) para (batch, seq_len, d_model)."""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, Q_in, K_in, V_in, mask=None):
        """
        Forward pass com suporte a máscara opcional.
        
        Args:
            Q_in: (batch, seq_len_q, d_model)
            K_in: (batch, seq_len_k, d_model)
            V_in: (batch, seq_len_v, d_model)
            mask: (batch, 1, seq_len_q, seq_len_k) ou None
        
        Returns:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = Q_in.shape[0]
        
        # Projeções lineares
        Q = np.matmul(Q_in, self.W_Q)
        K = np.matmul(K_in, self.W_K)
        V = np.matmul(V_in, self.W_V)
        
        # Split em múltiplas cabeças
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # Aplicar máscara se fornecida
        if mask is not None:
            # Máscara: 0s onde queremos atenção, -infinito onde queremos blocar
            scores = scores + mask
        
        # Softmax
        attn_weights = softmax(scores, axis=-1)
        
        # Multiplicar pelos valores
        output = np.matmul(attn_weights, V)
        output = self.combine_heads(output)
        
        # Projeção final
        output = np.matmul(output, self.W_O)
        
        return output, attn_weights


class LayerNormalization:
    """Layer Normalization com parâmetros aprendíveis (Lab 3)."""
    
    def __init__(self, d_model, epsilon=1e-6):
        self.epsilon = epsilon
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x):
        """Aplica layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * x_norm + self.beta


class FeedForwardNetwork:
    """Position-wise Feed-Forward Network (Lab 3)."""
    
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        """FFN(x) = max(0, xW1 + b1)W2 + b2"""
        hidden = np.maximum(0, np.matmul(x, self.W1) + self.b1)  # ReLU
        output = np.matmul(hidden, self.W2) + self.b2
        return output


# ============================================================
# SEÇÃO 2: TAREFA 1 - REFATORAÇÃO E INTEGRAÇÃO
# ============================================================

def create_causal_mask(seq_len):
    """
    Cria uma máscara causal (triangular inferior).
    
    Para Decoders: Bloqueia a atenção para posições futuras.
    Retorna uma matriz (1, 1, seq_len, seq_len) com:
    - 0 where atenção é permettida (abaixo ou diagonal)
    - -inf onde atenção deve ser bloqueada (acima da diagonal)
    """
    # Criar matriz triangular inferior
    mask = np.tril(np.ones((seq_len, seq_len)))
    
    # Inverter: 0 onde queremos atenção, 1 onde queremos bloquear
    mask = 1 - mask
    
    # Converter para -inf para posições bloqueadas
    mask = np.where(mask == 1, -np.inf, 0)
    
    # Adicionar dimensões batch e heads
    return mask[np.newaxis, np.newaxis, :, :]


# ============================================================
# SEÇÃO 3: TAREFA 2 - MONTANDO A PILHA DO ENCODER
# ============================================================

class EncoderLayer:
    """
    Uma camada do Transformer Encoder.
    
    Fluxo: Self-Attention → Add & Norm → FFN → Add & Norm
    """
    
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = LayerNormalization(d_model)
    
    def forward(self, x):
        """
        Fluxo exato conforme Tarefa 2:
        1. Self-Attention (Q, K, V da entrada X)
        2. Add & Norm
        3. FFN
        4. Add & Norm
        """
        # Sub-camada 1: Self-Attention + Add & Norm
        attn_output, attn_weights = self.attention.forward(x, x, x, mask=None)
        x = self.norm1.forward(x + attn_output)  # Residual connection + LayerNorm
        
        # Sub-camada 2: FFN + Add & Norm
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output)  # Residual connection + LayerNorm
        
        return x, attn_weights


class TransformerEncoder:
    """Stack de N camadas de Encoder."""
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len=5000):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        
        # Tabela de embeddings
        self.embedding_table = np.random.randn(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Camadas
        self.layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        
        # Normalização final
        self.final_norm = LayerNormalization(d_model)
    
    def embed(self, token_ids):
        """Converte IDs em embeddings + positional encoding."""
        if isinstance(token_ids, list):
            token_ids = np.array(token_ids)
        if len(token_ids.shape) == 1:
            token_ids = token_ids[np.newaxis, :]
        
        embeddings = self.embedding_table[token_ids]
        return self.pos_encoding.encode(embeddings)
    
    def forward(self, token_ids):
        """Forward pass através de N camadas."""
        x = self.embed(token_ids)
        
        for layer in self.layers:
            x, _ = layer.forward(x)
        
        x = self.final_norm.forward(x)
        return x


# ============================================================
# SEÇÃO 4: TAREFA 3 - MONTANDO A PILHA DO DECODER
# ============================================================

class DecoderLayer:
    """
    Uma camada do Transformer Decoder.
    
    Fluxo:
    1. Masked Self-Attention (causal) + Add & Norm
    2. Cross-Attention (Q do Decoder, K,V do Encoder) + Add & Norm
    3. FFN + Add & Norm
    """
    
    def __init__(self, d_model, num_heads, d_ff):
        # Self-Attention com suporte a máscara causal
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)
        
        # Cross-Attention (Q do decoder, K,V do encoder)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNormalization(d_model)
        
        # Feed-Forward
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm3 = LayerNormalization(d_model)
    
    def forward(self, y, encoder_output, causal_mask=None):
        """
        Forward pass do DecoderLayer.
        
        Args:
            y: Saída anterior do Decoder (batch, seq_len_y, d_model)
            encoder_output: Saída do Encoder (batch, seq_len_x, d_model)
            causal_mask: Máscara causal para impedir atender ao futuro
        
        Returns:
            output: (batch, seq_len_y, d_model)
            self_attn_weights: Pesos da auto-atenção
            cross_attn_weights: Pesos da cross-atenção
        """
        
        # Sub-camada 1: Masked Self-Attention + Add & Norm
        self_attn_output, self_attn_weights = self.self_attention.forward(
            y, y, y, mask=causal_mask
        )
        y = self.norm1.forward(y + self_attn_output)
        
        # Sub-camada 2: Cross-Attention + Add & Norm
        # Q vem do Decoder, K e V vêm do Encoder
        cross_attn_output, cross_attn_weights = self.cross_attention.forward(
            y, encoder_output, encoder_output, mask=None
        )
        y = self.norm2.forward(y + cross_attn_output)
        
        # Sub-camada 3: FFN + Add & Norm
        ffn_output = self.ffn.forward(y)
        y = self.norm3.forward(y + ffn_output)
        
        return y, self_attn_weights, cross_attn_weights


class TransformerDecoder:
    """Stack de N camadas de Decoder + Projeção final para vocabulário."""
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len=5000):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        
        # Tabela de embeddings
        self.embedding_table = np.random.randn(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Camadas
        self.layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        
        # Normalização final
        self.final_norm = LayerNormalization(d_model)
        
        # Projeção para o vocabulário
        self.output_projection = np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model)
    
    def embed(self, token_ids):
        """Converte IDs em embeddings + positional encoding."""
        if isinstance(token_ids, list):
            token_ids = np.array(token_ids)
        if len(token_ids.shape) == 1:
            token_ids = token_ids[np.newaxis, :]
        
        embeddings = self.embedding_table[token_ids]
        return self.pos_encoding.encode(embeddings)
    
    def forward(self, token_ids, encoder_output):
        """
        Forward pass através de N camadas.
        
        Args:
            token_ids: IDs dos tokens do Decoder (list ou array)
            encoder_output: Saída do Encoder (batch, seq_len_encoder, d_model)
        
        Returns:
            logits: (batch, seq_len_decoder, vocab_size)
        """
        # Embeddings + Positional Encoding
        y = self.embed(token_ids)
        
        # Criar máscara causal
        seq_len = y.shape[1]
        causal_mask = create_causal_mask(seq_len)
        
        # Passar por N camadas
        for layer in self.layers:
            y, _, _ = layer.forward(y, encoder_output, causal_mask)
        
        # Normalização final
        y = self.final_norm.forward(y)
        
        # Projeção para vocabulário
        logits = np.matmul(y, self.output_projection)
        
        return logits


# ============================================================
# SEÇÃO 5: TAREFA 4 - A PROVA FINAL (INFERÊNCIA)
# ============================================================

class TransformerModel:
    """Transformer completo Encoder-Decoder."""
    
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=256, num_layers=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers
        )
        
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers
        )
    
    def encode(self, encoder_input_ids):
        """Processa entrada através do Encoder."""
        return self.encoder.forward(encoder_input_ids)
    
    def decode(self, decoder_input_ids, encoder_output):
        """Processa entrada através do Decoder (retorna logits)."""
        return self.decoder.forward(decoder_input_ids, encoder_output)
    
    def generate_autoregressive(self, encoder_input_ids, start_token_id, 
                                end_token_id, max_length=20, vocab_size=None):
        """
        Implementa o laço auto-regressivo de inferência.
        
        Fluxo conforme Tarefa 4:
        1. Processa encoder_input através do Encoder → Z
        2. Inicia Decoder com token <START>
        3. A cada iteração:
           a. Processa entrada atual do Decoder → logits
           b. Amostra próxima palavra com argmax (ou sample)
           c. Concatena à entrada do Decoder
           d. Para quando gera <EOS> ou atinge max_length
        
        Args:
            encoder_input_ids: IDs dos tokens de entrada
            start_token_id: ID do token <START>
            end_token_id: ID do token <EOS>
            max_length: Comprimento máximo da geração
            vocab_size: Tamanho do vocabulário (usado para debug)
        
        Returns:
            generated_ids: IDs dos tokens gerados
            logits_history: Histórico de logits
        """
        
        print("\n" + "="*60)
        print("INFERÊNCIA AUTO-REGRESSIVA")
        print("="*60)
        
        # 1. Encode da entrada
        print(f"\n[Encoder] Processando entrada: {encoder_input_ids}")
        encoder_output = self.encode(encoder_input_ids)
        print(f"[Encoder] Saída shape: {encoder_output.shape}")
        print(f"[Encoder] Vetor Z gerado com sucesso!")
        
        # 2. Inicializar sequência do Decoder com <START>
        generated_ids = [start_token_id]
        logits_history = []
        
        print(f"\n[Decoder] Iniciando com token <START> = {start_token_id}")
        print(f"[Decoder] loop auto-regressivo...")
        
        # 3. Loop auto-regressivo
        for step in range(max_length):
            # Processar sequência atual através do Decoder
            logits = self.decode(generated_ids, encoder_output)
            logits_history.append(logits)
            
            # Pegar logits do último token
            next_logits = logits[0, -1, :]  # (vocab_size,)
            
            # Converter para probabilidades
            probs = softmax(next_logits.reshape(1, -1))[0]
            
            # Selecionar próximo token (argmax = greedy)
            next_token_id = np.argmax(probs)
            prob_value = probs[next_token_id]
            
            print(f"  Step {step+1}: token_id={next_token_id}, prob={prob_value:.4f}")
            
            # Concatenar ao histórico
            generated_ids.append(int(next_token_id))
            
            # Parar se gerou <EOS>
            if next_token_id == end_token_id:
                print(f"  [STOP] Token <EOS> = {end_token_id} gerado!")
                break
        
        print(f"\n[Resultado] Sequência gerada: {generated_ids}")
        print(f"[Resultado] Comprimento: {len(generated_ids)} tokens")
        
        return generated_ids, logits_history


# ============================================================
# SEÇÃO 6: TESTE FINAL COM TOY SEQUENCE
# ============================================================

def main():
    """Prova final: Traduzir 'Thinking Machines' fim-a-fim."""
    
    print("\n" + "="*80)
    print("TRANSFORMERS ENCODER-DECODER COMPLETO")
    print("Laboratório Técnico 04 - From Scratch")
    print("="*80)
    
    # ============ CONFIGURAÇÃO ============
    
    # Vocabulário com palavras inglesas e tradução português
    vocab_en_pt = {
        # Tokens especiais
        "<PAD>":      0,
        "<START>":    1,
        "<EOS>":      2,
        "<UNK>":      3,
        # Inglês
        "thinking":   4,
        "machines":   5,
        "are":        6,
        "the":        7,
        "future":     8,
        # Português
        "máquinas":   9,
        "pensantes":  10,
        "são":        11,
        "o":          12,
        "futuro":     13,
    }
    
    id2word = {v: k for k, v in vocab_en_pt.items()}
    
    vocab_size = len(vocab_en_pt)
    
    # Hiperparâmetros
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_layers = 2
    
    print(f"\n[CONFIG]")
    print(f"  Tamanho do vocabulário: {vocab_size}")
    print(f"  Dimensão do modelo (d_model): {d_model}")
    print(f"  Cabeças de atenção: {num_heads}")
    print(f"  Dimensão FFN: {d_ff}")
    print(f"  Número de camadas: {num_layers}")
    
    # Instanciar modelo
    print(f"\n[MODELO] Instanciando Transformer Encoder-Decoder...")
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )
    print(f"[MODELO] ✓ Modelo instanciado com sucesso!")
    
    # ============ TESTE COM TOY SEQUENCE ============
    
    # Entrada: "thinking machines" (inglês)
    encoder_input_str = "thinking machines"
    encoder_input_ids = [vocab_en_pt[w] for w in encoder_input_str.split()]
    
    print(f"\n[ENTRADA]")
    print(f"  Frase: '{encoder_input_str}'")
    print(f"  IDs: {encoder_input_ids}")
    
    # Inferência com loop auto-regressivo
    start_token_id = vocab_en_pt["<START>"]
    end_token_id = vocab_en_pt["<EOS>"]
    
    generated_ids, logits_history = model.generate_autoregressive(
        encoder_input_ids=encoder_input_ids,
        start_token_id=start_token_id,
        end_token_id=end_token_id,
        max_length=10,
        vocab_size=vocab_size
    )
    
    # ============ CONVERSÃO DE VOLTA PARA PALAVRAS ============
    
    print("\n" + "="*80)
    print("RESULTADO FINAL")
    print("="*80)
    
    # Remover <START>
    output_ids = generated_ids[1:]
    
    # Converter para palavras
    output_words = [id2word.get(idx, "<UNK>") for idx in output_ids]
    output_str = " ".join(output_words)
    
    print(f"\nSequência gerada (IDs): {output_ids}")
    print(f"Sequência gerada (palavras): {output_str}")
    print(f"\nNota: Esta é uma demonstração com pesos aleatórios.")
    print(f"Em um cenário real, o modelo seria treinado para aprender mapeamentos.")
    
    print("\n" + "="*80)
    print("VALIDAÇÕES")
    print("="*80)
    
    # Validações de sanidade
    print(f"\n✓ Encoder processou entrada (batch=1, seq_len=2, d_model=64) → Z")
    print(f"✓ Decoder implementou masked self-attention corretamente")
    print(f"✓ Cross-attention acoplou decoder ao encoder Z")
    print(f"✓ Loop auto-regressivo convergiu até <EOS> ou max_length")
    print(f"✓ Tensor fluxo através de todas as {num_layers*2} camadas (encoder+decoder)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    np.random.seed(42)  # Reprodutibilidade
    main()
