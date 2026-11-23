"""Constraint filtering and multi-objective normalization & scoring for CompressRoute.

Filters candidate models violating latency or memory SLAs, normalizes metrics
into [0, 1], and scores models using weighted user presets.
"""

from typing import List, Dict, Any, Tuple
import yaml
import numpy as np


DEFAULT_PRESETS = {
    "max_accuracy": {"w_acc": 0.70, "w_lat": 0.15, "w_mem": 0.10, "w_thr": 0.05},
    "balanced": {"w_acc": 0.35, "w_lat": 0.35, "w_mem": 0.15, "w_thr": 0.15},
    "fastest": {"w_acc": 0.10, "w_lat": 0.60, "w_mem": 0.10, "w_thr": 0.20},
    "edge": {"w_acc": 0.20, "w_lat": 0.30, "w_mem": 0.40, "w_thr": 0.10}
}


def filter_candidates(
    candidates: List[Dict[str, Any]],
    constraints: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Filters models that violate specified latency, memory, or hardware constraints.

    Returns:
        Tuple of (valid_candidates, rejection_reasons).
    """
    valid: List[Dict[str, Any]] = []
    rejections: List[Dict[str, str]] = []

    max_lat = constraints.get("max_latency_ms")
    max_mem = constraints.get("max_memory_mb")
    hw = constraints.get("hardware")

    for m in candidates:
        m_lat = m["latency_ms"]["bs1"]["p95"]
        m_size = m["size_mb"]

        # 1. Latency check
        if max_lat is not None and m_lat > max_lat:
            rejections.append({
                "name": m["name"],
                "reason": f"Latency {m_lat:.2f}ms exceeds limit {max_lat:.1f}ms"
            })
            continue

        # 2. Memory check
        if max_mem is not None and m_size > max_mem:
            rejections.append({
                "name": m["name"],
                "reason": f"Size {m_size:.1f}MB exceeds memory limit {max_mem:.1f}MB"
            })
            continue

        # 3. Edge hardware check (prefers INT8 or compact models <= 180MB)
        if hw == "Edge" and m_size > 180.0:
            rejections.append({
                "name": m["name"],
                "reason": f"Model size {m_size:.1f}MB too large for Edge deployment (max 180MB)"
            })
            continue

        valid.append(m)

    return valid, rejections


def score_candidates(
    candidates: List[Dict[str, Any]],
    preset: str = "balanced",
    weights_override: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """Normalizes candidate metrics and computes composite multi-criteria scores.

    Formula:
        Score = w_acc*Acc + w_lat*(1 - Lat) + w_mem*(1 - Mem) + w_thr*Thr
    """
    if not candidates:
        return []

    weights = weights_override or DEFAULT_PRESETS.get(preset, DEFAULT_PRESETS["balanced"])
    w_acc = weights.get("w_acc", 0.35)
    w_lat = weights.get("w_lat", 0.35)
    w_mem = weights.get("w_mem", 0.15)
    w_thr = weights.get("w_thr", 0.15)

    # Extract arrays
    accs = np.array([m["accuracy"] for m in candidates], dtype=float)
    lats = np.array([m["latency_ms"]["bs1"]["p95"] for m in candidates], dtype=float)
    mems = np.array([m["size_mb"] for m in candidates], dtype=float)
    thrs = np.array([m["throughput"]["bs1"] for m in candidates], dtype=float)

    def min_max_norm(arr: np.ndarray) -> np.ndarray:
        span = arr.max() - arr.min()
        if span == 0:
            return np.ones_like(arr) * 0.5
        return (arr - arr.min()) / span

    norm_acc = min_max_norm(accs)
    norm_lat = min_max_norm(lats)
    norm_mem = min_max_norm(mems)
    norm_thr = min_max_norm(thrs)

    scored_candidates = []
    for idx, m in enumerate(candidates):
        score = (
            w_acc * norm_acc[idx] +
            w_lat * (1.0 - norm_lat[idx]) +
            w_mem * (1.0 - norm_mem[idx]) +
            w_thr * norm_thr[idx]
        )
        entry = dict(m)
        entry["composite_score"] = round(float(score), 4)
        entry["norm_acc"] = round(float(norm_acc[idx]), 3)
        entry["norm_lat"] = round(float(norm_lat[idx]), 3)
        entry["norm_mem"] = round(float(norm_mem[idx]), 3)
        entry["norm_thr"] = round(float(norm_thr[idx]), 3)
        scored_candidates.append(entry)

    # Sort descending by composite score
    scored_candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored_candidates
