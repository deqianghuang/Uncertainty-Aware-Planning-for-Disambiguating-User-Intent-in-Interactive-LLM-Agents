# src/uap/parsing.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


# Matches: robot action: robot.<ACTION>(<ARGS>)
_FULL = re.compile(
    r"robot\s*action\s*:\s*robot\.(?P<act>[^(\n]+?)\((?P<args>[^)]*?)\)",
    flags=re.IGNORECASE,
)

# A more relaxed matcher in case the model omits the "robot action: robot." prefix:
_RELAXED = re.compile(
    r"(?P<act>搜索特定类别或特色的地点|查询地点相关信息|查询导航路线相关信息|查询导航路线|查询整体时间|返回费用估算|查询公交信息|搜索特定线路|询问到达时间|返回下一站信息|返回实时交通信息|查询打车方案|返回排队人数|返回到达时间|选择车型或价格偏好)\((?P<args>[^)]*?)\)"
)


@dataclass(frozen=True)
class ParsedAction:
    name: str
    args: str

    def to_text(self) -> str:
        return f"robot action: robot.{self.name}({self.args})"


def canonicalize_text(text: str) -> str:
    # remove duplicated prefixes like "robot action: robot.robot action: robot."
    t = text.replace("robot action: robot.robot action: robot.", "robot action: robot.")
    # normalize whitespace
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def parse_actions(text: str) -> List[ParsedAction]:
    """Extract actions from raw LLM output. Returns possibly empty list."""
    text = canonicalize_text(text)

    actions: List[ParsedAction] = []
    for m in _FULL.finditer(text):
        act = canonicalize_text(m.group("act"))
        args = canonicalize_text(m.group("args"))
        if act:
            actions.append(ParsedAction(name=act, args=args))

    if actions:
        return actions

    # fallback
    for m in _RELAXED.finditer(text):
        act = canonicalize_text(m.group("act"))
        args = canonicalize_text(m.group("args"))
        actions.append(ParsedAction(name=act, args=args))

    return actions


def is_done(text: str) -> bool:
    # Accept a few variants
    t = text.lower()
    return "done" in t or "robot.done" in t or "robot action: robot.done" in t
