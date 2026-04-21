# Laboratorio 08 - Alinhamento Humano com DPO (HHH)

## Objetivo

Implementar um pipeline de alinhamento de LLM para produzir comportamento HHH (Helpful, Honest, Harmless) com DPO (Direct Preference Optimization), priorizando respostas seguras e suprimindo respostas inadequadas.

## Estrutura

lab-8/
|- generate_hhh_dataset.py
|- Makefile
|- dpo_preferences.jsonl
|- train_dpo_hhh.py
|- requirements.txt
`- README.md

## Passo 1 - Dataset de Preferencias (HHH)

Arquivo: dpo_preferences.jsonl

- Formato JSONL com colunas obrigatorias: prompt, chosen, rejected.
- Quantidade: 32 exemplos (>= 30), focados em:
1. Restricoes de seguranca
2. Etica corporativa
3. Adequacao de tom profissional

Geracao/regeracao do dataset:

```powershell
python generate_hhh_dataset.py --output-file dpo_preferences.jsonl
```

Validacao rapida de colunas:

```powershell
python -c "import json; from pathlib import Path; p=Path('dpo_preferences.jsonl'); rows=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]; req={'prompt','chosen','rejected'}; assert all(req.issubset(r) for r in rows); print('OK:', len(rows), 'linhas')"
```

## Passo 2 - Pipeline DPO com trl

Arquivo: train_dpo_hhh.py

- Usa `trl.DPOTrainer`.
- Carrega dois modelos:
1. Modelo Ator (`model`): atualizado durante treino
2. Modelo de Referencia (`ref_model`): congelado para o termo de regularizacao (KL)

Comando sugerido:

```powershell
python train_dpo_hhh.py --base-model gpt2 --dataset-file dpo_preferences.jsonl --output-dir outputs/dpo-hhh --beta 0.1
```

## Passo 3 - Papel Matematico do Beta (beta = 0.1)

No DPO, o treinamento aumenta a preferencia relativa pela resposta `chosen` sobre `rejected`, mas sempre em comparacao com um modelo de referencia. O hiperparametro $\beta$ controla a intensidade dessa atualizacao: quando $\beta$ e pequeno, o modelo muda mais devagar e tende a preservar a distribuicao original; quando $\beta$ e grande, a otimizacao de preferencia fica mais agressiva e pode degradar fluencia, cobertura linguistica e estabilidade. Em termos práticos, $\beta$ funciona como um "imposto" sobre desvios em relacao ao comportamento base, porque aumenta o custo implícito de afastar a politica treinada da politica de referencia (efeito equivalente a uma regularizacao por KL). Com $\beta = 0.1$, adota-se um compromisso conservador: reforcar seguranca e adequacao sem destruir a qualidade geral de linguagem aprendida no pretreino/fine-tuning anterior.

## Passo 4 - Treino e Inferencia

Configuracoes de economia de memoria implementadas:

- `optim="paged_adamw_32bit"`
- `gradient_accumulation_steps` configuravel
- `fp16` automaticamente ativado quando CUDA estiver disponivel

Treino:

```powershell
python train_dpo_hhh.py --base-model gpt2 --dataset-file dpo_preferences.jsonl --output-dir outputs/dpo-hhh --num-epochs 1 --batch-size 1 --grad-accum 8 --beta 0.1
```

Validacao no console (automatica apos treino):

- O script calcula score de log-prob para `chosen` e `rejected` no mesmo prompt.
- Esperado: `Score chosen > Score rejected`.
- Se isso nao ocorrer no primeiro ciclo, aumentar epocas, ajustar taxa de aprendizado ou ampliar dataset.

## Critrios de Entrega (Contrato Pedagogico)

1. Versionamento Git:
- Entregar via Git e marcar versao final com tag `v1.0`.

2. Funcionalidade:
- Pipeline carrega `DPOTrainer` sem erro de sintaxe.
- Dataset contem estritamente as colunas `prompt`, `chosen`, `rejected`.
- Explicacao matematica do beta esta documentada.

3. Integridade e uso de IA:
- Permitido uso de IA para pesquisa, brainstorming e templates com revisao critica.
- Nota obrigatoria (manter no README):

"Partes geradas/complementadas com IA, revisadas por [Seu Nome]".

4. Prazos e penalidades:
- Ate 23h59 da data estipulada: sem penalidade.
- 1 dia de atraso: -20%.
- 2 a 3 dias: -50%.
- Mais de 3 dias: nota 0 (salvo justificativa oficial).

## Comandos de Entrega

```powershell
git add .
git commit -m "Lab 08 - Alinhamento HHH com DPO"
git tag v1.0
git push origin main --tags
```

## Nota obrigatoria sobre uso de IA

"Partes geradas/complementadas com IA, revisadas por [Seu Nome]".
