# Lab 3: Transformer Encoder "From Scratch"

**Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
**Professor:** Prof. Dimmy Magalhães  
**Instituição:** iCEV - Instituto de Ensino Superior

## 📋 Objetivo

Implementar a passagem direta (Forward Pass) de um bloco Encoder completo do Transformer, conforme descrito no artigo original ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

## 🎯 Requisitos

- ✅ Apenas Python 3.x, numpy e pandas
- ❌ Proibido: PyTorch, TensorFlow, Keras ou bibliotecas de atenção prontas
- 📐 d_model = 64 (otimizado para CPU)
- 🔢 N = 6 camadas do encoder
- 📊 Entrada: frase simples → Saída: representação contínua densa (Z)

## 🏗️ Arquitetura Implementada

### 1. Preparação dos Dados
- **Vocabulário**: Mapeamento de palavras para IDs inteiros usando pandas
- **Embeddings**: Tabela de vetores aleatórios (vocab_size × d_model)
- **Formato**: Tensor 3D (BatchSize, SequenceLength, d_model)

### 2. Componentes Principais

#### 2.1 Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```
- Inicialização de matrizes W_Q, W_K, W_V
- Cálculo de Q, K, V por projeção linear
- Scaling por sqrt(d_k)
- Softmax customizado (sem libs prontas)
- Multiplicação pelos valores V

#### 2.2 Layer Normalization
```
Output = LayerNorm(x + Sublayer(x))
```
- Normalização na dimensão dos features
- Cálculo de média e variância
- Parâmetros aprendíveis gamma e beta
- Epsilon para estabilidade numérica

#### 2.3 Feed-Forward Network (FFN)
```
FFN(x) = max(0, xW1 + b1)W2 + b2
```
- Expansão: d_model → d_ff (256)
- Ativação ReLU
- Contração: d_ff → d_model

#### 2.4 Positional Encoding
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Encoding absoluto de posição
- Funções seno/cosseno de diferentes frequências

### 3. Fluxo de uma Camada do Encoder

```
1. X_att = SelfAttention(X)
2. X_norm1 = LayerNorm(X + X_att)        # Add & Norm
3. X_ffn = FFN(X_norm1)
4. X_out = LayerNorm(X_norm1 + X_ffn)    # Add & Norm
5. X = X_out  (para próxima camada)
```

### 4. Empilhamento de N=6 Camadas

A saída de cada camada se torna a entrada da próxima, mantendo sempre as dimensões (Batch, Tokens, 512).

## 📁 Estrutura de Arquivos

```
lab-3/
├── transformer_encoder.py   # Implementação completa do Transformer
├── test_transformer.py       # Testes unitários de todos os componentes
├── README.md                 # Esta documentação
└── Makefile                  # Comandos para executar e testar
```

## 🚀 Como Executar

### Executar o programa principal:
```bash
python transformer_encoder.py
```

ou usando Make:
```bash
make run
```

### Executar os testes:
```bash
python test_transformer.py
```

ou:
```bash
make test
```

### Limpar cache:
```bash
make clean
```

## 📊 Exemplo de Saída

```
FRASE DE ENTRADA: 'o banco bloqueou meu cartao de credito'

Tokens: ['o', 'banco', 'bloqueou', 'meu', 'cartao', 'de', 'credito']
IDs: [0, 1, 2, 4, 3, 5, 6]

EXECUTANDO FORWARD PASS
Camada 1 concluída. Shape: (1, 8, 64)
Camada 2 concluída. Shape: (1, 8, 64)
Camada 3 concluída. Shape: (1, 8, 64)
Camada 4 concluída. Shape: (1, 8, 64)
Camada 5 concluída. Shape: (1, 8, 64)
Camada 6 concluída. Shape: (1, 8, 64)

RESULTADOS
Shape de entrada: (1, 8, 64)
Shape de saída (Z): (1, 8, 64)
✓ Shape correto: (1, 8, 64) == (1, 8, 64)
✓ Diferença média dos embeddings originais: 2.145678
```

## 🧪 Testes Implementados

1. **Softmax** - Verifica normalização e soma = 1
2. **Scaled Dot-Product Attention** - Valida dimensões e attention weights
3. **Layer Normalization** - Confirma média ≈ 0, variância ≈ 1
4. **Feed-Forward Network** - Testa projeções e ReLU
5. **Encoder Layer** - Valida camada completa com residuais
6. **Positional Encoding** - Verifica padrões seno/cosseno
7. **Forward Pass Completo** - Testa 6 camadas sequenciais
8. **Validação de Dimensões** - Sanidade das shapes

## 📐 Validação de Sanidade

O tensor deve:
- ✅ Entrar com dimensões (Batch, Tokens, 64)
- ✅ Sair com dimensões (Batch, Tokens, 64)
- ✅ Ter valores contextualizados (diferentes dos embeddings originais)
- ✅ Não conter NaN ou Inf

## 🔍 Conceitos Matemáticos Importantes

### Broadcasting no NumPy
```python
# Adicionar positional encoding
X + pe  # (batch, seq, d_model) + (seq, d_model) ✓
```

### Multiplicação de Matrizes
```python
# Atenção: QK^T
np.matmul(Q, K.transpose(0, 2, 1))
# (batch, seq, d_model) @ (batch, d_model, seq) → (batch, seq, seq)
```

### Softmax Numericamente Estável
```python
x_shifted = x - np.max(x, axis=-1, keepdims=True)  # Evita overflow
exp_x = np.exp(x_shifted)
softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

## 📚 Referências

- Vaswani, A., et al. (2017). "Attention Is All You Need". NeurIPS.
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

## 👨‍💻 Autor

Implementado para o curso de Tópicos em Inteligência Artificial - iCEV 2026.1

## 📝 Notas

- A implementação usa d_model=64 ao invés de 512 para otimização em CPU
- Todos os componentes foram implementados do zero, sem uso de frameworks de deep learning
- Os pesos são inicializados aleatoriamente (Xavier/Glorot initialization)
- O código inclui validações extensivas para garantir correção matemática
