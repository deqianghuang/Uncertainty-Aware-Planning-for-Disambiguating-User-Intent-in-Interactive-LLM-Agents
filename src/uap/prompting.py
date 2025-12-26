# src/uap/prompting.py
from __future__ import annotations
from typing import List, Sequence
from .actions import TASK_SPECS, TaskSpec, ActionSpec


def build_prompt(
    task: str,
    goal: str,
    few_shots: Sequence[str],
    history_steps: Sequence[str],
) -> str:
    spec: TaskSpec = TASK_SPECS[task]

    lines: List[str] = []
    lines.append(spec.intro)
    lines.append("你的动作只包括以下动作名（必须逐字一致）：")
    for a in spec.actions:
        lines.append(f"- {a.name}: {a.desc}")
        if a.examples:
            ex = a.examples[0]
            lines.append(f"  例: robot action: robot.{a.name}({ex})")

    lines.append("\n以下是一些例子：")
    for s in few_shots:
        lines.append(s.strip() + "\n")

    # This is the key: history is per-beam-node, not a global planner field.
    if history_steps:
        lines.append("目前已选择的序列有：")
        for step in history_steps:
            lines.append(step.strip())

    lines.append(f"\n目标: {goal}")
    lines.append(
        "请输出下一步动作，严格只输出一行，格式必须为："
        "robot action: robot.<动作名>(<参数>)。不要输出任何分析、列表、JSON、英文动作名。"
    )
    lines.append("robot action: robot.")
    return "\n".join(lines)
