# CyberWell RAG Pipeline

RAG over CyberWell research PDFs in `data/raw/`. Answers are grounded in retrieved chunks with inline citations.

## Setup

```bash
pip install -r requirements.txt
```

Place a GCP service account key at `gcp_key.json` (Vertex AI enabled). See `config.py` for models and chunking.

## Build index

```bash
python src/build_index.py
```

## Query

```bash
python src/rag_system.py "Your question here"
```

## Evaluation

**Retrieval** (default gold set):

```bash
python eval/run_eval.py
```

Gold sets (chunk IDs must match the built index): `eval/gold_set_paragrph_window.jsonl`, `eval/gold_set_fixed.jsonl`. Experiment outputs: `eval/results/`.

## Report

[`report.md`](report.md)

## Layout

```
data/raw/          PDF corpus
data/processed/    chunks + Chroma (generated)
src/               pipeline
eval/              gold_set_*.jsonl, run_eval.py, results/
report.md
config.py
```
