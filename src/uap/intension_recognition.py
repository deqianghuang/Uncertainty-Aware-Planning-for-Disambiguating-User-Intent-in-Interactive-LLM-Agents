# src/uap/intension_recognition.py
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Any, List, Optional

import pandas as pd

from .llm_client import QianfanOpenAIClient, VLLMChatClient, LLMClient

ALLOWED_TASKS = ["POI", "Navi", "Transit", "Taxi", "Others"]

SYSTEM_PROMPT = """你是一个“用户意图分类器”。

请你从以下五个类别中选择最匹配的类别，以抽象用户的意图：
1. POI (兴趣点)：用户希望搜索特定地点或位置，如停车场、餐馆、景点等。
2. Navi (导航)：用户希望获取导航或路线规划信息。
3. Transit (公共交通)：用户希望获取有关公共交通的信息。
4. Taxi (打车)：用户希望获取打车或出租车服务相关的信息。
5. Others (其他)：用户希望获取其他信息。

要求：
- 你最终只能输出 POI / Navi / Transit / Taxi / Others 中的一个类别
- 不要输出任何分析过程、不要输出多行、不要输出解释
- 如果用户意图模糊，请尽量选择最接近的选项，尽量避免选择 Others
"""

USER_TEMPLATE = """用户查询：
{goal}

请输出一个类别（POI / Navi / Transit / Taxi / Others），只输出类别本身："""


def _make_llm(backend: str, qianfan_model: str, vllm_model_path: str) -> LLMClient:
    if backend == "qianfan":
        return QianfanOpenAIClient(model=qianfan_model)
    if backend == "vllm":
        if not vllm_model_path:
            raise RuntimeError("--vllm_model_path is required when backend=vllm")
        return VLLMChatClient(model_path=vllm_model_path)
    raise RuntimeError(f"Unknown backend: {backend}")


def _normalize_label(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "Others"

    # 常见脏输出兜底：只取第一行、去掉引号/标点
    t = t.splitlines()[0].strip().strip('"').strip("'").strip()
    # 有的模型可能输出类似 "类别：POI"
    for key in ALLOWED_TASKS:
        if key.lower() in t.lower():
            return key
    return "Others"


def classify_intention(
    llm: LLMClient,
    goal: str,
    temperature: float = 0.0,
) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(goal=goal)}"
    # 为了稳定：n=1，temperature=0 or small
    out = llm.generate(prompt, n=1, temperature=temperature, max_tokens=16)[0]
    return _normalize_label(out)


def _read_rows(input_path: str, id_col: str, goal_col: str) -> List[Dict[str, str]]:
    """
    Returns list of dict rows that at least have {id, goal}.
    Supports .csv and .xlsx/.xls
    """
    if input_path.lower().endswith(".csv"):
        rows: List[Dict[str, str]] = []
        with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                rid = (row.get(id_col) or str(idx + 1)).strip()
                goal = (row.get(goal_col) or "").strip()
                rows.append({"id": rid, "goal": goal})
        return rows

    if input_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_path)
        if goal_col in df.columns:
            goals = df[goal_col].astype(str).tolist()
        else:
            # 兼容你旧代码：默认取第3列
            goals = df.iloc[:, 2].astype(str).tolist()

        rows = []
        for i, g in enumerate(goals):
            rid = str(i + 1)
            rows.append({"id": rid, "goal": (g or "").strip()})
        return rows

    raise RuntimeError("Unsupported input format. Please use .csv or .xlsx/.xls")


def write_labeled_csv(
    out_path: str,
    rows: List[Dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = ["id", "task", "goal", "label"]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser(description="Intention recognition -> output CSV in (id,task,goal,label) format")

    ap.add_argument("--input", type=str, required=True, help="Input CSV or Excel (.xlsx/.xls)")
    ap.add_argument("--output", type=str, required=True, help="Output CSV path (id,task,goal,label)")

    ap.add_argument("--id_col", type=str, default="id", help="CSV column name for id")
    ap.add_argument("--goal_col", type=str, default="goal", help="CSV column name for goal")

    ap.add_argument("--backend", type=str, default="qianfan", choices=["qianfan", "vllm"])
    ap.add_argument("--qianfan_model", type=str, default="ernie-4.5-turbo-128k")
    ap.add_argument("--vllm_model_path", type=str, default="")

    ap.add_argument("--temperature", type=float, default=0.0)

    # label策略：默认不强行给标签；你可以让 Others -> 1
    ap.add_argument("--label_ambiguous", type=int, default=-1,
                    help="If set to 0/1, assign this label when predicted task is Others. Default: -1 means keep label empty.")

    args = ap.parse_args()

    llm = _make_llm(args.backend, args.qianfan_model, args.vllm_model_path)

    in_rows = _read_rows(args.input, args.id_col, args.goal_col)
    out_rows: List[Dict[str, Any]] = []

    for r in in_rows:
        rid = r["id"]
        goal = (r["goal"] or "").strip()
        if not goal:
            out_rows.append({"id": rid, "task": "Others", "goal": goal, "label": ""})
            continue

        task = classify_intention(llm=llm, goal=goal, temperature=args.temperature)

        label_val = ""
        if args.label_ambiguous in (0, 1) and task == "Others":
            label_val = int(args.label_ambiguous)

        out_rows.append({"id": rid, "task": task, "goal": goal, "label": label_val})

    write_labeled_csv(args.output, out_rows)
    print(f"[OK] wrote {len(out_rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
