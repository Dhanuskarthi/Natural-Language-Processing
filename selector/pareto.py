"""Pareto efficiency filter for CompressRoute model selection.

Computes the non-dominated Pareto frontier across (Accuracy, Latency, Memory)
and tags models as Pareto-optimal.
"""

from typing import List, Dict, Any


def is_dominated(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Returns True if model b dominates model a."""
    a_acc = a["accuracy"]
    a_lat = a["latency_ms"]["bs1"]["p95"]
    a_mem = a["size_mb"]

    b_acc = b["accuracy"]
    b_lat = b["latency_ms"]["bs1"]["p95"]
    b_mem = b["size_mb"]

    # b dominates a if b is at least as good in all 3 and strictly better in at least one
    weakly_better = (b_acc >= a_acc) and (b_lat <= a_lat) and (b_mem <= a_mem)
    strictly_better = (b_acc > a_acc) or (b_lat < a_lat) or (b_mem < a_mem)

    return weakly_better and strictly_better


def filter_pareto_frontier(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filters candidate list to return only non-dominated Pareto-optimal models."""
    frontier = []
    for i, a in enumerate(candidates):
        dominated = False
        for j, b in enumerate(candidates):
            if i != j and is_dominated(a, b):
                dominated = True
                break
        if not dominated:
            frontier.append(a)
    return frontier


def tag_pareto_status(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adds 'is_pareto': True/False attribute to each candidate in the list."""
    frontier = filter_pareto_frontier(candidates)
    pareto_names = set(m["name"] for m in frontier)

    tagged = []
    for m in candidates:
        m_copy = dict(m)
        m_copy["is_pareto"] = m["name"] in pareto_names
        tagged.append(m_copy)

    return tagged
