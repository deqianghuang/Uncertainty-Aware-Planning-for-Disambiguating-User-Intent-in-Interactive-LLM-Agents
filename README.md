# Uncertainty-Aware Planning for Disambiguating User Intent in Interactive LLM Agents (UAP)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18060682.svg)](https://doi.org/10.5281/zenodo.18060682)


This repository contains a clean-room, minimal implementation of **uncertainty-aware planning** for map-style tasks:

- **POI**: 兴趣点搜索  
- **Navi**: 导航  
- **Transit**: 公共交通  
- **Taxi**: 打车  

Backends:
- **Qianfan (OpenAI-compatible)** API backend
- **Local vLLM** backend (e.g., **Qwen2-7B-Instruct**)

Dataset note:
- The original dataset contains private user queries and cannot be released.
- This repo provides a small **20-query demo dataset** for illustration only.

---

## Environment Setup

Recommended: Python 3.10+

Install dependencies:

```bash
pip install -U openai numpy pandas spacy
python -m spacy download zh_core_web_lg
```

If you want to use the **vLLM** backend:

```bash
pip install -U vllm
```

Make module import work:

```bash
export PYTHONPATH=src
```

---

## Backends

### Qianfan (OpenAI-compatible)

Set your Qianfan API key via environment variable (DO NOT hardcode in repo):

```bash
export QIANFAN_API_KEY="YOUR_QIANFAN_BEARER_TOKEN"
```

Example model:

- `ernie-4.5-turbo-128k`

### Local vLLM (Qwen2-7B-Instruct)

Example local model path:

- `/ssd3/huangdeqiang/LLM/Qwen2-7B-Instruct`

---

## Run Single Query

### Qianfan

```bash
python -m uap.cli \
  --task Transit \
  --goal "从百度科技园去天安门坐几路公交" \
  --backend qianfan \
  --qianfan_model "ernie-4.5-turbo-128k"
```

### vLLM (Qwen2-7B-Instruct)

```bash
python -m uap.cli \
  --task Transit \
  --goal "从百度科技园去天安门坐几路公交" \
  --backend vllm \
  --vllm_model_path /ssd3/huangdeqiang/LLM/Qwen2-7B-Instruct
```

---

## Run Batch (Demo Dataset)

Input CSV format (header required):

- `id,task,goal,label`

Run (Qianfan):

```bash
python -m uap.cli \
  --input_csv data/demo_20.csv \
  --backend qianfan \
  --qianfan_model "ernie-4.5-turbo-128k" \
  --output outputs/preds.jsonl
```

Run (vLLM / Qwen2):

```bash
python -m uap.cli \
  --input_csv data/demo_20.csv \
  --backend vllm \
  --vllm_model_path /ssd3/huangdeqiang/LLM/Qwen2-7B-Instruct \
  --output outputs/preds.jsonl
```

Output format:

- Default: **JSONL** (one JSON per line)
- If you want a JSON list:

```bash
python -m uap.cli \
  --input_csv data/demo_20.csv \
  --backend qianfan \
  --qianfan_model "ernie-4.5-turbo-128k" \
  --output outputs/preds.json \
  --pretty_json
```

---

## Intention Recognition (Optional)

`intension_recognition.py` labels raw queries into one of:

- `POI`, `Navi`, `Transit`, `Taxi`, `Others`

Input CSV format (header required):

- `id,goal`

Example (Qianfan):

```bash
python -m uap.intension_recognition \
  --input_csv data/raw_queries.csv \
  --output_csv outputs/labeled_queries.csv \
  --backend qianfan \
  --qianfan_model "ernie-4.5-turbo-128k"
```

Example (vLLM / Qwen2):

```bash
python -m uap.intension_recognition \
  --input_csv data/raw_queries.csv \
  --output_csv outputs/labeled_queries.csv \
  --backend vllm \
  --vllm_model_path /ssd3/huangdeqiang/LLM/Qwen2-7B-Instruct
```

Output CSV format:

- `id,task,goal,label`

  - `task` is the predicted category
  - `label` is optional; if you need it, you can set a rule in the script (e.g., `task == Others` -> `label = 1`)

---

## Repository Layout

Place files in the following layout:

```text
.
├── README.md
├── data
│   ├── demo_20.csv
│   └── raw_queries.csv                # (optional) for intention recognition; columns: id,goal
├── outputs                            # (optional) created after running commands
│   ├── preds.jsonl
│   └── labeled_queries.csv
└── src
    └── uap
        ├── __init__.py
        ├── actions.py
        ├── parsing.py
        ├── prompting.py
        ├── uncertainty.py
        ├── llm_client.py
        ├── cli.py
        ├── intension_recognition.py
        └── search
            ├── __init__.py
            └── beam.py
```

---

## Notes

- This repo focuses on planning + uncertainty scoring (training-free).
- The demo dataset is intentionally small due to privacy constraints.
- Do not upload private user queries or logs containing sensitive data.
