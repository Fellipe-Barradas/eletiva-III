# Lab 2 - Transformer Encoder From Scratch

**Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
**Professor:** Prof. Dimmy Magalhães  
**Instituição:** iCEV - Instituto de Ensino Superior

## 📋 Objetivo

Implementar a passagem direta (Forward Pass) de um bloco **Encoder completo** do Transformer, conforme descrito no artigo seminal "Attention Is All You Need" (Vaswani et al., 2017).

## 🚫 Restrições

- ✅ **Permitido:** Python 3.x, NumPy, Pandas
- ❌ **Proibido:** PyTorch, TensorFlow, Keras, bibliotecas de atenção prontas

## 🏗️ Arquitetura Implementada

### Componentes Principais

1. **Embedding Layer**
   - Converte tokens (IDs) em vetores densos de dimensão `d_model`
   - Tabela de embeddings: `(vocab_size, d_model)`

2. **Scaled Dot-Product Attention**
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k)V
   ```
   - Projeções Q, K, V usando matrizes de peso
   - Scaling por √d_k para estabilidade
   - Softmax implementado manualmente com NumPy

3. **Layer Normalization**
   ```
   LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
   ```
   - Normalização na dimensão dos features (axis=-1)
   - Parâmetros aprendíveis γ (gamma) e β (beta)

4. **Feed-Forward Network (FFN)**
   ```
   FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
   ```
   - Expande de `d_model` para `d_ff`
   - Ativação ReLU
   - Contrai de volta para `d_model`

5. **Residual Connections**
   ```
   Output = LayerNorm(x + Sublayer(x))
   ```
   - Combate o problema de vanishing gradient
   - Aplicado após atenção e FFN

### Estrutura de Uma Camada do Encoder

```
Input X (batch, seq_len, d_model)
    ↓
┌───────────────────────────────┐
│  Self-Attention               │
│  X_att = Attention(X, X, X)   │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│  Add & Norm                   │
│  X₁ = LayerNorm(X + X_att)    │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│  Feed-Forward Network         │
│  X_ffn = FFN(X₁)              │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│  Add & Norm                   │
│  X_out = LayerNorm(X₁ + X_ffn)│
└───────────────────────────────┘
    ↓
Output (batch, seq_len, d_model)
```

### Stack de N=6 Camadas

O output de uma camada se torna o input da próxima, mantendo sempre as dimensões `(batch_size, seq_len, d_model)`.

## 🎯 Parâmetros do Modelo

| Parâmetro | Valor no Paper | Valor Implementado | Descrição |
|-----------|---------------|-------------------|-----------|
| `d_model` | 512 | 64 | Dimensão do modelo |
| `d_ff` | 2048 | 256 | Dimensão da FFN |
| `N` | 6 | 6 | Número de camadas |
| `d_k` | 64 | 64 | Dimensão para scaling |

*Nota: Valores reduzidos para processamento em CPU sem GPU.*

## 📦 Estrutura do Código

```
lab-2/
├── transformer_encoder.py    # Implementação principal
├── test_transformer.py        # Testes unitários
├── Makefile                   # Comandos úteis
└── README.md                  # Este arquivo
```

## 🚀 Como Executar

### Execução Principal

```bash
python transformer_encoder.py
```

### Executar Testes

```bash
python test_transformer.py
```

### Usando o Makefile

```bash
make run          # Executa o programa principal
make test         # Executa os testes
make clean        # Limpa arquivos temporários
```

## 📊 Exemplo de Saída

```
Frase de entrada: 'o banco bloqueou meu cartao'
IDs dos tokens: [0, 1, 2, 4, 3]

Shape inicial após embeddings: (1, 5, 64)
Shape após camada 1: (1, 5, 64)
Shape após camada 2: (1, 5, 64)
Shape após camada 3: (1, 5, 64)
Shape após camada 4: (1, 5, 64)
Shape após camada 5: (1, 5, 64)
Shape após camada 6: (1, 5, 64)

Vetor Z final (representação contextualizada):
  - Shape: (1, 5, 64)
  - Batch Size: 1
  - Sequence Length: 5
  - Model Dimension: 64
```

## 🧪 Validações Implementadas

1. **Sanidade de Dimensões:** Verifica que o tensor mantém shape `(batch, seq_len, d_model)` através de todas as camadas
2. **Estabilidade Numérica:** Softmax com subtração do máximo para evitar overflow
3. **Layer Norm:** Epsilon (1e-6) para evitar divisão por zero

## 📚 Conceitos-Chave

### Self-Attention
Permite que cada palavra da frase "olhe" para todas as outras palavras, capturando dependências de longo alcance.

### Residual Connections
Ajudam o gradiente a fluir através de redes profundas durante o treinamento (neste lab, apenas forward pass).

### Layer Normalization
Estabiliza os valores dos tensores, normalizando pela média e variância dos features.

### Feed-Forward Network
Adiciona capacidade de representação não-linear após a atenção.

## 🎓 Aprendizados

- Compreensão profunda da mecânica interna do Transformer
- Manipulação de tensores multidimensionais com NumPy
- Broadcasting e multiplicação de matrizes
- Implementação de funções matemáticas fundamentais (softmax, relu, layer norm)

## 📖 Referências

- Vaswani et al. (2017). "Attention Is All You Need". NeurIPS.
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need - Paper](https://arxiv.org/abs/1706.03762)

## 👨‍💻 Autor

Implementado para a disciplina de Tópicos em IA, iCEV 2026.1
