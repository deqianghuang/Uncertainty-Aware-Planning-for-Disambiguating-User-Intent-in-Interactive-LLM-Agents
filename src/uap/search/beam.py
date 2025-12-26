# src/uap/search/beam.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..actions import allowed_action_names
from ..prompting import build_prompt
from ..parsing import parse_actions, is_done
from ..uncertainty import clara_uncertainty, Embedder
from ..llm_client import LLMClient


@dataclass(frozen=True)
class Candidate:
    action_text: str
    freq_score: float
    unct: float
    done: bool


@dataclass(frozen=True)
class BeamNode:
    steps: Tuple[str, ...]
    uncts: Tuple[float, ...]
    cum_unct: float
    done: bool


def propose_candidates(
    llm: LLMClient,
    task: str,
    goal: str,
    few_shots: Sequence[str],
    history_steps: Sequence[str],
    n_samples: int,
    temperature: float,
    embedder: Optional[Embedder],
) -> List[Candidate]:
    prompt = build_prompt(task=task, goal=goal, few_shots=few_shots, history_steps=history_steps)
    texts = llm.generate(prompt, n=n_samples, temperature=temperature, max_tokens=256)

    allowed = set(allowed_action_names(task))

    # --------- CLARA-style uncertainty samples (NO filtering) ----------
    subj_samples: List[str] = []
    obj_samples: List[str] = []

    # --------- candidate collection (filter by allowed actions) --------
    parsed_first_actions: List[str] = []

    for t in texts:
        if is_done(t):
            # treat done as a valid sample for unct as well (matches old behavior)
            subj_samples.append("done")
            obj_samples.append("done")
            parsed_first_actions.append("robot action: robot.done()")
            continue

        acts = parse_actions(t)
        if not acts:
            continue

        # take the first parsed action (keep consistent with your run loop behavior)
        a0 = acts[0]

        # for uncertainty: always record the raw first action name+args (even if not allowed)
        subj_samples.append(a0.name)
        obj_samples.append(a0.args)

        # for candidates: only keep allowed actions (your affordance whitelist analogue)
        if a0.name in allowed:
            parsed_first_actions.append(a0.to_text())

    if not parsed_first_actions:
        return []

    # frequency as a simple score proxy (clean-room replacement of your counting logic)
    counts = {}
    for a in parsed_first_actions:
        counts[a] = counts.get(a, 0) + 1
    total = sum(counts.values())

    # ✅ restore paper/old-code uncertainty definition
    unct_obj = clara_uncertainty(subj_samples=subj_samples, obj_samples=obj_samples, embedder=embedder)
    unct_val = float(unct_obj.total)

    cands: List[Candidate] = []
    for a, c in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        done = "done" in a.lower()
        cands.append(Candidate(action_text=a, freq_score=c / total, unct=unct_val, done=done))
    return cands


def beam_search(
    llm: LLMClient,
    task: str,
    goal: str,
    few_shots: Sequence[str],
    beam: int = 2,
    max_steps: int = 3,
    n_samples: int = 6,
    temperature: float = 0.2,
    embedder: Optional[Embedder] = None,
) -> BeamNode:
    nodes: List[BeamNode] = [BeamNode(steps=tuple(), uncts=tuple(), cum_unct=0.0, done=False)]
    best_done: Optional[BeamNode] = None

    for _step in range(max_steps):
        expanded: List[BeamNode] = []

        for node in nodes:
            if node.done:
                expanded.append(node)
                continue

            cands = propose_candidates(
                llm=llm,
                task=task,
                goal=goal,
                few_shots=few_shots,
                history_steps=list(node.steps),
                n_samples=n_samples,
                temperature=temperature,
                embedder=embedder,
            )

            for cand in cands:
                # only block exact duplicates; DO NOT block by prefix (prevents step2=0 bug)
                if cand.action_text in node.steps:
                    continue

                new_steps = node.steps + (cand.action_text,)
                new_uncts = node.uncts + (cand.unct,)
                new_cum = node.cum_unct + cand.unct
                new_node = BeamNode(steps=new_steps, uncts=new_uncts, cum_unct=new_cum, done=cand.done)

                if new_node.done and (best_done is None or new_node.cum_unct < best_done.cum_unct):
                    best_done = new_node
                expanded.append(new_node)

        if not expanded:
            break

        # prune: lower cumulative uncertainty is better
        expanded.sort(key=lambda n: (n.done is False, n.cum_unct))
        nodes = expanded[:beam]

        if any(n.done for n in nodes):
            break

    return best_done if best_done is not None else min(nodes, key=lambda n: n.cum_unct)
