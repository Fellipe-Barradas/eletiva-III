# Laboratorio 6 - P2: Tokenizador BPE e Exploracao de WordPiece

## Visao Geral

Neste laboratorio, foi implementado do zero o nucleo do algoritmo **Byte Pair Encoding (BPE)** para aprender sub-palavras por fusao de pares frequentes, e depois realizada uma comparacao pratica com o tokenizador **WordPiece** do BERT multilingue da Hugging Face.

## Estrutura

```
lab-6/
|-- bpe_wordpiece.py
|-- test_bpe_wordpiece.py
|-- Makefile
`-- README.md
```

## Tarefa 1 - Motor de Frequencias

Foi inicializado o vocabulario exatamente como solicitado:

```python
vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}
```

A funcao `get_stats(vocab)` percorre cada palavra tokenizada por espacos e soma as frequencias dos pares adjacentes.

Validacao atendida:

- O par `('e', 's')` retorna frequencia **9**.

## Tarefa 2 - Loop de Fusao

A funcao `merge_vocab(pair, v_in)` substitui ocorrencias do par isolado por sua versao fundida (por exemplo, `('e', 's') -> 'es'`) usando expressao regular com fronteiras de token.

Foi implementado um loop principal com $K=5$ iteracoes:

1. Calcula pares com `get_stats`
2. Seleciona o par mais frequente
3. Aplica fusao com `merge_vocab`
4. Imprime par fundido e estado do vocabulario

Resultado esperado observado:

- Formacao de tokens morfologicos como `est</w>` durante as iteracoes.

## Tarefa 3 - Integracao Industrial com WordPiece

Foi utilizado:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
```

Frase de teste:

- `Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar.`

A tokenizacao e impressa no terminal via `tokenizer.tokenize(...)`.

### Relatorio: significado de `##`

No WordPiece do BERT, tokens com prefixo `##` indicam que aquela sub-palavra **continua** uma palavra iniciada por um token anterior. Exemplo: `inconstitucional`, `##mente` representa um sufixo anexado ao radical. Esse mecanismo reduz drasticamente o problema de vocabulario desconhecido (OOV), porque, mesmo que a palavra completa nunca tenha aparecido no treinamento, o modelo consegue decomp-la em partes menores conhecidas (radicais, prefixos, sufixos) e ainda produzir representacao numerica valida sem travar.

## Como executar

No diretorio do laboratorio:

```bash
make install
make test
make run
```

Alternativa direta:

```bash
python test_bpe_wordpiece.py
python bpe_wordpiece.py
```

## Versionamento e Entrega

- Realizar commit do conteudo no GitHub.
- Criar a tag obrigatoria final:

```bash
git tag v1.0
git push origin v1.0
```

## Citacao de uso de IA generativa (obrigatoria)

Trechos de apoio gerados com IA generativa foram usados na construcao inicial das funcoes de manipulacao de strings e regex para fusao de pares no BPE (`merge_vocab`), e depois revisados manualmente para garantir conformidade com o enunciado e corretude dos resultados.
