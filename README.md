# Uncertainty-Aware Planning for Disambiguating User Intent in Interactive LLM Agents

TL;DR — Code and materials for the KDD’26 ADS paper *“Uncertainty-Aware Planning for Disambiguating User Intent in Interactive LLM Agents: Application to Baidu Maps.”*

## Status
- [ ] Code release (coming soon)
- [x] Repo skeleton + artifacts layout
- [ ] Reproduction instructions

## Repo Layout
- `prompts/` — prompt templates and examples.
- `dispatch/` — command dispatch logic and flow explanations.
- `docs/cases/` — qualitative examples used in the paper.
- `extras/` — few-shot baselines / additional results.
- `scripts/run_demo.py` — minimal demo entry (placeholder).
- `data/` — small synthetic/sample inputs only (no proprietary data).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_demo.py --queries data/sample_queries.jsonl
