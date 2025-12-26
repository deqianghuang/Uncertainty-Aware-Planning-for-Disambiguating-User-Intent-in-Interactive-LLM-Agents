# src/uap/cli.py
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, Any, List, Optional

from .actions import TASK_SPECS
from .uncertainty import Embedder
from .llm_client import QianfanOpenAIClient, VLLMChatClient
from .search.beam import beam_search


DEFAULT_QWEN2_PATH = "/ssd3/huangdeqiang/LLM/Qwen2-7B-Instruct"


def load_fewshots(task: str) -> list[str]:
    # 先返回空也能跑；后续你可以把原来的 PROMPTS_* 接进来
    return []


def _make_llm(args):
    if args.backend == "qianfan":
        return QianfanOpenAIClient(model=args.qianfan_model)
    if args.backend == "vllm":
        model_path = (args.vllm_model_path or "").strip()
        if not model_path:
            raise RuntimeError("--vllm_model_path is required when backend=vllm")
        return VLLMChatClient(
            model_path=model_path,
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_mem,
            tensor_parallel_size=args.vllm_tp,
        )
    raise RuntimeError(f"Unknown backend: {args.backend}")


def _run_one(
    llm,
    embedder: Embedder,
    fewshots_cache: Dict[str, list[str]],
    task: str,
    goal: str,
    beam: int,
    max_steps: int,
    n_samples: int,
    temperature: float,
) -> Dict[str, Any]:
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task: {task}. Must be one of {list(TASK_SPECS.keys())}")

    goal = (goal or "").strip()
    if not goal:
        raise ValueError("Empty goal")

    if task not in fewshots_cache:
        fewshots_cache[task] = load_fewshots(task)

    node = beam_search(
        llm=llm,
        task=task,
        goal=goal,
        few_shots=fewshots_cache[task],
        beam=beam,
        max_steps=max_steps,
        n_samples=n_samples,
        temperature=temperature,
        embedder=embedder,
    )

    return {
        "task": task,
        "goal": goal,
        "steps": list(node.steps),
        "uncts": list(node.uncts),
    }


def _write_outputs(records: List[Dict[str, Any]], out_path: str, pretty_json: bool = False) -> None:
    if not out_path:
        for r in records:
            print(json.dumps(r, ensure_ascii=False))
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if out_path.lower().endswith(".json"):
        with open(out_path, "w", encoding="utf-8") as f:
            if pretty_json:
                json.dump(records, f, ensure_ascii=False, indent=2)
            else:
                json.dump(records, f, ensure_ascii=False)
        return

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()

    # single vs batch
    ap.add_argument("--task", type=str, default="", choices=list(TASK_SPECS.keys()),
                    help="Single mode: required unless using --input_csv with per-row task column.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--goal", type=str, help="Single goal (single-run mode).")
    group.add_argument("--input_csv", type=str, help="Batch mode: CSV path with columns like id,task,goal,label.")

    # backends
    ap.add_argument("--backend", type=str, default="qianfan", choices=["qianfan", "vllm"])
    ap.add_argument("--qianfan_model", type=str, default="ernie-4.5-turbo-128k")

    # vllm args (默认你的 Qwen2)
    ap.add_argument("--vllm_model_path", type=str, default=DEFAULT_QWEN2_PATH)
    ap.add_argument("--vllm_max_model_len", type=int, default=2048)
    ap.add_argument("--vllm_gpu_mem", type=float, default=0.90)
    ap.add_argument("--vllm_tp", type=int, default=1)

    # search params
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=3)
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.2)

    # embedding model
    ap.add_argument("--spacy_model", type=str, default="zh_core_web_lg")

    # batch IO
    ap.add_argument("--output", type=str, default="",
                    help="Output path. If endswith .json -> write a JSON list; else write JSONL. Default: stdout JSONL.")
    ap.add_argument("--pretty_json", action="store_true", help="If output is .json, indent it.")

    # batch column mapping
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--task_col", type=str, default="task")
    ap.add_argument("--goal_col", type=str, default="goal")
    ap.add_argument("--label_col", type=str, default="label")

    args = ap.parse_args()

    llm = _make_llm(args)
    embedder = Embedder(args.spacy_model)
    fewshots_cache: Dict[str, list[str]] = {}

    # ---------- single mode ----------
    if args.goal is not None:
        if not args.task:
            raise RuntimeError("--task is required in single mode.")
        out = _run_one(
            llm=llm,
            embedder=embedder,
            fewshots_cache=fewshots_cache,
            task=args.task,
            goal=args.goal,
            beam=args.beam,
            max_steps=args.max_steps,
            n_samples=args.n_samples,
            temperature=args.temperature,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # ---------- batch mode ----------
    assert args.input_csv is not None
    records: List[Dict[str, Any]] = []

    with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            rid = (row.get(args.id_col) or str(row_idx)).strip()
            task = (row.get(args.task_col) or args.task or "").strip()
            goal = (row.get(args.goal_col) or "").strip()
            label_raw = row.get(args.label_col, "")
            label: Optional[int] = None
            try:
                if str(label_raw).strip() != "":
                    label = int(float(label_raw))
            except Exception:
                label = None

            rec: Dict[str, Any] = {
                "id": rid,
                "task": task,
                "goal": goal,
            }
            if label is not None:
                rec["label"] = label

            if not task:
                rec["error"] = f"Missing task (no '{args.task_col}' column and no --task fallback)."
                records.append(rec)
                continue
            if not goal:
                rec["error"] = "Empty goal"
                records.append(rec)
                continue

            try:
                pred = _run_one(
                    llm=llm,
                    embedder=embedder,
                    fewshots_cache=fewshots_cache,
                    task=task,
                    goal=goal,
                    beam=args.beam,
                    max_steps=args.max_steps,
                    n_samples=args.n_samples,
                    temperature=args.temperature,
                )
                rec.update({"steps": pred["steps"], "uncts": pred["uncts"]})
            except Exception as e:
                rec["error"] = str(e)

            records.append(rec)
            print(f"[INFO] {rid} {task}: done", file=sys.stderr)

    _write_outputs(records, args.output, pretty_json=args.pretty_json)


if __name__ == "__main__":
    main()
