# src/uap/uncertainty.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional, List
import numpy as np

try:
    import spacy
except Exception:
    spacy = None


@dataclass
class Uncertainty:
    total: float
    detail: dict


class Embedder:
    """Light wrapper. Loads spaCy model once."""
    def __init__(self, model_name: str = "zh_core_web_lg"):
        self.model_name = model_name
        self._nlp = None

    def _get(self):
        if self._nlp is None:
            if spacy is None:
                raise RuntimeError("spaCy not available")
            self._nlp = spacy.load(self.model_name)
        return self._nlp

    def embed(self, text: str) -> np.ndarray:
        nlp = self._get()
        return nlp(text).vector


def avg_pairwise_cosine_distance(vectors: np.ndarray) -> float:
    n = vectors.shape[0]
    if n <= 1:
        return 0.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    v = vectors / norms
    sim = v @ v.T
    dis = 1.0 - sim
    return float((dis.sum() - np.trace(dis)) / (n * (n - 1)))


def avg_pairwise_euclidean_distance(vectors: np.ndarray) -> float:
    """
    Match your old CLARA code more closely:
      dis = scipy.spatial.distance_matrix(vecs, vecs)
      div = sum(dis) / (n*(n-1))
    Here we compute it with pure numpy.
    """
    n = vectors.shape[0]
    if n <= 1:
        return 0.0
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
    norms = np.sum(vectors * vectors, axis=1, keepdims=True)  # (n,1)
    d2 = norms + norms.T - 2.0 * (vectors @ vectors.T)
    d2 = np.maximum(d2, 0.0)
    d = np.sqrt(d2)
    return float((d.sum() - np.trace(d)) / (n * (n - 1)))


def diversity_uncertainty(texts: Sequence[str], embedder: Optional[Embedder] = None) -> Uncertainty:
    """
    A simple uncertainty proxy: diversity among candidate strings.
    (kept for other uses; this is NOT the CLARA/type2 uncertainty)
    """
    uniq = list(dict.fromkeys([t.strip() for t in texts if t and t.strip()]))
    if len(uniq) <= 1:
        return Uncertainty(total=0.0, detail={"diversity": 0.0, "n": len(uniq)})

    if embedder is None:
        sets = [set(x) for x in uniq]
        n = len(sets)
        dis = []
        for i in range(n):
            for j in range(i + 1, n):
                inter = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j]) + 1e-12
                dis.append(1.0 - inter / union)
        d = float(np.mean(dis)) if dis else 0.0
        return Uncertainty(total=d, detail={"diversity": d, "n": len(uniq), "mode": "char-jaccard"})

    vecs = np.vstack([embedder.embed(x) for x in uniq])
    d = avg_pairwise_cosine_distance(vecs)
    return Uncertainty(total=d, detail={"diversity": d, "n": len(uniq), "mode": "spacy-cosine"})


def clara_uncertainty(
    subj_samples: Sequence[str],
    obj_samples: Sequence[str],
    embedder: Optional[Embedder],
    normalize_div: float = 5.0,
) -> Uncertainty:
    """
    Restore your original CLARA/type2 uncertainty definition:

      obj_raw = avg_pairwise_distance(obj_samples)   (NO de-dup)
      sub_raw = avg_pairwise_distance(subj_samples)  (NO de-dup)

      obj = obj_raw / 5
      sub = sub_raw / 5
      total = (obj_raw + sub_raw) / 5

    We use avg pairwise EUCLIDEAN distance to match your old scipy.distance_matrix behavior.
    """
    subj = [s.strip() for s in subj_samples if isinstance(s, str) and s.strip()]
    obj = [s.strip() for s in obj_samples if isinstance(s, str) and s.strip()]

    if embedder is None:
        # fallback: if no embedder, just return 0 (paper/old code assumes embedder exists)
        return Uncertainty(total=0.0, detail={"mode": "no-embedder", "obj": 0.0, "sub": 0.0, "n_obj": len(obj), "n_sub": len(subj)})

    if len(subj) <= 1 and len(obj) <= 1:
        return Uncertainty(total=0.0, detail={"mode": "clara-euclidean", "obj": 0.0, "sub": 0.0, "n_obj": len(obj), "n_sub": len(subj)})

    obj_raw = 0.0
    sub_raw = 0.0

    if len(obj) > 1:
        obj_vecs = np.vstack([embedder.embed(x) for x in obj])
        obj_raw = avg_pairwise_euclidean_distance(obj_vecs)

    if len(subj) > 1:
        sub_vecs = np.vstack([embedder.embed(x) for x in subj])
        sub_raw = avg_pairwise_euclidean_distance(sub_vecs)

    obj_scaled = float(obj_raw / normalize_div)
    sub_scaled = float(sub_raw / normalize_div)
    total = float((obj_raw + sub_raw) / normalize_div)

    return Uncertainty(
        total=total,
        detail={
            "mode": "clara-euclidean",
            "obj": obj_scaled,
            "sub": sub_scaled,
            "obj_raw": float(obj_raw),
            "sub_raw": float(sub_raw),
            "n_obj": len(obj),
            "n_sub": len(subj),
            "normalize_div": normalize_div,
        },
    )
