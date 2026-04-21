# Laboratorio 07 - Especializacao de LLMs com LoRA e QLoRA

## Objetivo

Construir um pipeline completo de fine-tuning de um modelo fundacional com tecnicas de eficiencia de parametros (PEFT/LoRA) e quantizacao (QLoRA), viabilizando treino em hardware limitado.

## Estrutura

lab-7/
|- generate_synthetic_dataset.py
|- train_lora_qlora.py
|- requirements.txt
|- Makefile
|- README.md
`- data/
   |- train.jsonl
   `- test.jsonl

## Passo 1 - Engenharia de Dados Sinteticos

Arquivo: generate_synthetic_dataset.py

- Usa a OpenAI API para gerar pares prompt/response no dominio escolhido.
- Gera no minimo 50 amostras (padrao: 60).
- Divide os dados em treino/teste (padrao 90/10).
- Salva em JSONL: data/train.jsonl e data/test.jsonl.

Exemplo de execucao:

python generate_synthetic_dataset.py --domain "suporte academico" --num-samples 60 --train-ratio 0.9 --output-dir data

## Passo 2 - Quantizacao QLoRA (4-bit)

Arquivo: train_lora_qlora.py

Configuracao implementada com bitsandbytes:

- load_in_4bit=True
- bnb_4bit_quant_type="nf4"
- bnb_4bit_compute_dtype=torch.float16

## Passo 3 - Arquitetura LoRA

Configuracao implementada com peft (LoraConfig):

- task_type=CAUSAL_LM
- r=64
- lora_alpha=16
- lora_dropout=0.1

## Passo 4 - Treinamento e Otimizacao

Treinamento implementado com trl.SFTTrainer.

TrainingArguments configurado com:

- optim="paged_adamw_32bit"
- lr_scheduler_type="cosine"
- warmup_ratio=0.03

Ao final, o script salva o adaptador:

- trainer.model.save_pretrained(output_dir)

## Requisitos

1. Python 3.10+
2. GPU com suporte a CUDA (recomendado para QLoRA)
3. OPENAI_API_KEY configurada no ambiente
4. Permissao/licenca para baixar o modelo base (exemplo: Llama 2)

## Instalacao

pip install -r requirements.txt

ou

make install

## Execucao rapida

1. Gerar dataset sintetico:

make gen-data

1. Treinar adaptador LoRA/QLoRA:

make train

## Versionamento e Entrega (iCEV)

1. Subir todo o codigo e os arquivos JSONL para o repositorio no GitHub.
2. Marcar a versao final com tag v1.0.

Comandos:

git add .
git commit -m "Lab 07 - LoRA e QLoRA"
git tag v1.0
git push origin main --tags

## Nota obrigatoria sobre uso de IA

"Partes geradas/complementadas com IA, revisadas por Luis Fellipe Bezerra Barradas".

Essa nota deve permanecer no README da versao entregue.
