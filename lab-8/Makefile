install:
	pip install -r requirements.txt

gen-data:
	python generate_hhh_dataset.py --output-file dpo_preferences.jsonl

train:
	python train_dpo_hhh.py --base-model gpt2 --dataset-file dpo_preferences.jsonl --output-dir outputs/dpo-hhh --beta 0.1

validate-data:
	python -c "import json; from pathlib import Path; p=Path('dpo_preferences.jsonl'); rows=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]; req={'prompt','chosen','rejected'}; assert all(req.issubset(r) for r in rows); print('OK:', len(rows), 'linhas')"
