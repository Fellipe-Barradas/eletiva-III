# Laboratório Técnico 04: O Transformer Completo "From Scratch"

## 📋 Visão Geral

Este laboratório integra todos os componentes dos Labs 1, 2 e 3 para construir um **Transformer Encoder-Decoder completo** do zero, capaz de realizar tradução fim-a-fim de uma frase toy (brinquedo) usando um **loop auto-regressivo de inferência**.

## 🎯 Objetivos de Aprendizagem

1. **Aplicar engenharia de software** para integrar módulos de redes neurais separados em uma única topologia coerente.
2. **Garantir o fluxo correto** de tensores passando pelas camadas Add & Norm e Feed-Forward.
3. **Acoplar o Loop Auto-regressivo** de inferência na saída do Decoder.

## 📁 Estrutura de Arquivos

```
lab-4/
├── transformer_completo.py       # Implementação completa (Entra-Saída)
├── test_transformer.py            # Suite de testes unitários
├── exemplos.py                    # Exemplos de uso (será criado)
├── Makefile                       # Scripts de execução
└── README.md                      # Este arquivo
```

## 🔧 Componentes Implementados

### Seção 1: Refatoração e Integração (Tarefa 1)

Integra componentes dos Labs anteriores:

- **`softmax()`**: Softmax numericamente estável (Lab 1)
- **`PositionalEncoding`**: Encoding de posição com seno/cosseno (Lab 3)
- **`MultiHeadAttention`**: Multi-Head Attention com suporte a máscara (Labs 2-3)
- **`LayerNormalization`**: Normalização de camada com parâmetros (Lab 3)
- **`FeedForwardNetwork`**: Rede densa Position-wise (Labs 2-3)
- **`create_causal_mask()`**: Máscara causal para Decoder (Lab 2)

### Seção 2: Pilha do Encoder (Tarefa 2)

```
EncoderLayer:
  1. Self-Attention (Q, K, V da entrada X)
  2. Add & Norm  →  LayerNorm(X + Attention(X))
  3. FFN
  4. Add & Norm  →  LayerNorm(X_norm1 + FFN(X_norm1))

TransformerEncoder:
  N camadas de EncoderLayer em sequência
```

**Fluxo**: `X → [Embed + PosEnc] → N×EncoderLayer → Z (representação contextualizada bidireccional)`

### Seção 3: Pilha do Decoder (Tarefa 3)

```
DecoderLayer:
  1. Masked Self-Attention (com máscara causal)
  2. Add & Norm
  3. Cross-Attention (Q do Decoder, K,V do Encoder)
  4. Add & Norm
  5. FFN
  6. Add & Norm

TransformerDecoder:
  N camadas de DecoderLayer + Projeção final para vocabulário
```

**Fluxo**: `Y → [Embed + PosEnc] → N×DecoderLayer(com Z do Encoder) → Logits(vocab_size)`

### Seção 4: Prova Final - Inferência (Tarefa 4)

```python
def generate_autoregressive(encoder_input_ids, start_token_id, end_token_id, max_length):
    # 1. Encoder processa entrada → Z
    Z = encoder(encoder_input_ids)
    
    # 2. Inicializar Decoder com <START>
    generated = [start_token_id]
    
    # 3. Loop auto-regressivo:
    while len(generated) < max_length:
        logits = decoder(generated, Z)
        next_token = argmax(softmax(logits[-1]))  # Próxima palavra
        generated.append(next_token)
        if next_token == end_token_id:
            break
    
    return generated
```

## 📊 Tipos de Dados e Shapes

| Camada | Input Shape | Output Shape | Descrição |
|--------|-------------|--------------|-----------|
| Embeddings | `(batch, seq_len)` → IDs | `(batch, seq_len, d_model)` | Tokens → Vetores |
| Positional Encoding | `(batch, seq_len, d_model)` | `(batch, seq_len, d_model)` | +PosEnc |
| Multi-Head Attention | Q,K,V: `(B, S, d_model)` | `(B, S, d_model)` | Q,K,V projetados e concatenados |
| FFN | `(B, S, d_model)` | `(B, S, d_model)` | Expand →ReLU→ Contract |
| EncoderLayer | `(B, S, d_model)` | `(B, S, d_model)` | Self-Attn + FFN com residuais |
| DecoderLayer | Y: `(B, T, d_model)`, Z: `(B, S, d_model)` | `(B, T, d_model)` | Masked+Cross-Attn + FFN |
| Output Projection | `(B, T, d_model)` | `(B, T, vocab_size)` | Logits para cada token |

Onde: `B`=batch, `S`=seq_len encoder, `T`=seq_len decoder, `d_model`=dimensão interna

## ▶️ Como Usar

### 1. Executar o Teste Completo

```bash
make test
# ou
python test_transformer.py
```

Executa suite com 9 testes unitários cobrindo cada componente.

### 2. Executar a Prova Final (Demo)

```bash
make run
# ou
python transformer_completo.py
```

Instancia o modelo completo e realiza inferência auto-regressiva em uma frase toy:
- **Entrada**: "thinking machines" (inglês)
- **Esperado**: Sequência gerada através do Decoder auto-regressivo
- **Nota**: Com pesos aleatórios, a saída será aleatória; em um caso real, o modelo seria treinado.

### 3. Usar em Seu Código

```python
from transformer_completo import TransformerModel

# Criar modelo
model = TransformerModel(
    vocab_size=100,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6
)

# Inferência
encoder_input = [4, 5]  # IDs de "thinking machines"
start_token = 1         # <START>
end_token = 2           # <EOS>

generated_ids = model.generate_autoregressive(
    encoder_input_ids=encoder_input,
    start_token_id=start_token,
    end_token_id=end_token,
    max_length=20
)

print(f"Gerado: {generated_ids}")
```

## 🧪 Suite de Testes

Todos os testes validam:

1. ✓ **Causal Mask**: Bloqueia corretamente posições futuras
2. ✓ **Softmax**: Linhas somam 1.0
3. ✓ **Layer Normalization**: Média ~0, variância ~1
4. ✓ **EncoderLayer**: Shapes preservados
5. ✓ **DecoderLayer**: Shapes preservados com masked self-attention
6. ✓ **TransformerEncoder**: N camadas em sequência
7. ✓ **TransformerDecoder**: Logits com shape vocab_size
8. ✓ **TransformerModel**: Forward pass encoder + decoder
9. ✓ **Autoregressive Inference**: Gera sequência até <EOS>

## 🏗️ Arquitetura de Referência

```
                    ENCODER
    ┌─────────────────────────┐
    │  Input: "thinking ..."  │
    └────────────┬────────────┘
                 │
          [Embeddings + PosEnc]
                 │
    ┌────────────────────────────┐
    │  EncoderLayer 1            │  Self-Attention + FFN
    │  ├─ Self-Attention + LN   │  (Contextualiza bidireccionalmente)
    │  └─ FFN + LN              │
    └────────────┬───────────────┘
                 │
            (Repetir x5)
                 │
                 ↓
              Z (memória)
                 ↑
                 │
    ┌────────────────────────────┐
    │  DecoderLayer 1            │  Masked + Cross-Attention + FFN
    │  ├─ Masked Self-Attn + LN │  (Máscara causal previne trapacear)
    │  ├─ Cross-Attn(Z) + LN    │  (Acopla ao Encoder)
    │  └─ FFN + LN              │
    └────────────┬───────────────┘
                 │
            (Repetir x5)
                 │
          [Projeto → Vocab]
                 │
                 ↓
    ┌─────────────────────────┐
    │  Logits: Y_hat          │
    │  (vocab_size para cada position)
    └──────────┬──────────────┘
               │
        [Argmax / Sample]
               │
        [Loop Auto-regressivo]
               │
               ↓
         Output: predicted IDs
```

## 📈 Hyperparâmetros Padrão

```python
d_model = 64        # Dimensão interna (paper usa 512)
num_heads = 4       # Cabeças de atenção
d_ff = 256          # FFN expansion (4 × d_model)
num_layers = 2      # N camadas (paper usa 6)
max_seq_len = 5000  # Máximo comprimento de sequência
```

## 💡 Pontos-Chave de Implementação

### 1. **Máscara Causal (Decoder)**
```python
mask = create_causal_mask(seq_len)
# Retorna: (1, 1, seq_len, seq_len)
# 0 onde atenção é permitida (triangular inferior)
# -∞ onde atenção é bloqueada (acima da diagonal)
```

Garante que o Decoder nunca atende a tokens futuros durante treinamento/inferência.

### 2. **Conexão Residual + Normalização**
```python
# Em vez de:  output = FFN(Attention(X))
# Faz:        output = LayerNorm(X + FFN(LayerNorm(X + Attention(X))))
```

Estabiliza o treinamento em redes profundas.

### 3. **Cross-Attention**
```python
Q = decoder_output
K = encoder_output
V = encoder_output

cross_attn = MultiHeadAttention.forward(Q, K, V)
```

Q vem do Decoder (consulta "o que preciso?")  
K, V vêm do Encoder (memória "o que voce forneceu")

### 4. **Loop Auto-regressivo**
```
Iteração 1: Decoder(<START>) → P(token_2)
Iteração 2: Decoder(<START>, token_2) → P(token_3)
Iteração 3: Decoder(<START>, token_2, token_3) → P(token_4)
...
```

Cada iteração adiciona o token previamente previsto para prever o próximo.

## 🔍 Validação de Sanidade

Após executar, você verá confirmações:
- ✓ Encoder processou entrada → Z
- ✓ Decoder implementou masked self-attention
- ✓ Cross-attention acoplou decoder ao encoder
- ✓ Loop auto-regressivo convergiu

## 📖 Referências

- Vaswani et al. (2017): "Attention Is All You Need"
- Original paper: https://arxiv.org/abs/1706.03762
- Componentes de Labs anteriores:
  - Lab 1: Scaled Dot-Product Attention
  - Lab 2: Masked Attention + Encoder Stack
  - Lab 3: Positional Encoding + Multi-Head Attention

## 🎓 Lições Aprendidas

1. **Engenharia de Software**: Integrar componentes de forma modular e coerente
2. **Fluxo de Tensores**: Entender como dados fluem através de múltiplas camadas
3. **Máscara Causal**: Por que e como prevenir "trapacear" durante inferência
4. **Cross-Attention**: Como o Decoder consulta a memória do Encoder
5. **Inferência Auto-regressiva**: Gerar texto token a token iterativamente

## ❓ Perguntas para Exploração Futura

1. Como implementar beam search em vez de greedy sampling?
2. Como adicionar temperature scaling para controlar diversidade?
3. Como implementar training loop com backpropagation?

## 🤖 Declaração sobre Uso de Inteligência Artificial

**Partes geradas/complementadas com IA, revisadas pelo aluno.**

Este laboratório foi desenvolvido com assistência de IA (GitHub Copilot) para:
- **Geração de templates** e estrutura base das classes
- **Brainstorming** de arquitetura e componentes

**Responsabilidade do aluno:**
- ✓ Integração manual de todos os componentes dos Labs 1-3
- ✓ Implementação e validação da **lógica matemática** do Encoder-Decoder
- ✓ Desenvolvimento do **loop auto-regressivo** para inferência
- ✓ **Revisão completa** de todos os cálculos de forward pass
- ✓ **Testes unitários** para validar cada componente
- ✓ **Documentação** clara de fluxos de tensor e shapes

**Entendimento comprovado:**
- Mecanismo de Scaled Dot-Product Attention e Multi-Head Attention
- Máscara causal e por que previne "trapacear" no Decoder
- Conexões residuais (Add & Norm) e normalização de camada
- Cross-Attention acoplando Decoder ao Encoder
- Inferência auto-regressiva iterativa com geração de tokens

O código foi cuidadosamente revisado para garantir exatidão matemática e coerência arquitetural.
4. Como seria a dinâmica com multi-batch processing?
5. Como otimizar para maior velocidade (batching, cache)?

---

**Laboratório Técnico 04** | Disciplina: Tópicos em IA | Professor: Prof. Dimmy Magalhães | iCEV - 2026.1
