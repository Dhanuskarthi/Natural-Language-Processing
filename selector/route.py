"""End-to-end model selection and routing pipeline for CompressRoute.

Analyzes natural language requests, classifies task domain, parses resource
constraints, computes Pareto optimality, and ranks models with transparent rationales.
"""

from typing import Dict, Any, List, Optional
import json
import os
import yaml

from selector.task_classifier import TaskClassifier
from selector.constraint_parser import parse_constraints
from selector.scoring import filter_candidates, score_candidates, DEFAULT_PRESETS
from selector.pareto import tag_pareto_status
from selector.explain import (
    find_teacher_baseline,
    generate_recommendation_reason,
    generate_no_fit_explanation,
)


def load_registry(registry_path: str = "benchmark/results/registry.json") -> List[Dict[str, Any]]:
    """Loads benchmarked model variants from registry.json."""
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Registry file not found at: {registry_path}. Run benchmark first.")
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def select(
    query: str,
    preset: str = "balanced",
    registry_path: str = "benchmark/results/registry.json"
) -> Dict[str, Any]:
    """Routes and recommends the top-3 best NLP models for a plain English query.

    Args:
        query: User description (e.g., 'bank chatbot, CPU only, under 20 ms').
        preset: Priority preset ('max_accuracy', 'balanced', 'fastest', 'edge').
        registry_path: Path to benchmark registry.json.

    Returns:
        Structured recommendation dictionary with detected task, constraints,
        top-3 ranked models, and explanations.
    """
    # 1. Classify NLP Task
    clf = TaskClassifier()
    detected_task, task_conf = clf.predict(query)

    # 2. Parse Constraints
    constraints = parse_constraints(query)

    # 3. Load Model Registry
    all_models = load_registry(registry_path)
    teacher_model = find_teacher_baseline(all_models)

    # 4. Filter Candidates by Constraints
    valid_candidates, rejections = filter_candidates(all_models, constraints)

    if not valid_candidates:
        no_fit_msg = generate_no_fit_explanation(constraints, rejections, all_models)
        return {
            "status": "no_fit",
            "query": query,
            "detected_task": detected_task,
            "task_confidence": task_conf,
            "constraints": constraints,
            "rejections": rejections,
            "explanation": no_fit_msg,
            "top_models": []
        }

    # 5. Tag Pareto Frontier Status
    tagged_candidates = tag_pareto_status(valid_candidates)

    # 6. Multi-Criteria Scoring
    scored_candidates = score_candidates(tagged_candidates, preset=preset)

    # 7. Select Top 3 and Generate Explanations
    top_3 = scored_candidates[:3]
    recommendations = []
    for rank, m in enumerate(top_3, start=1):
        reason = generate_recommendation_reason(m, teacher_model, rank=rank)
        rec = dict(m)
        rec["rank"] = rank
        rec["explanation"] = reason
        recommendations.append(rec)

    return {
        "status": "success",
        "query": query,
        "detected_task": detected_task,
        "task_confidence": task_conf,
        "preset": preset,
        "constraints": constraints,
        "total_viable_models": len(valid_candidates),
        "top_models": recommendations,
        "explanation": f"Recommended {len(recommendations)} viable models out of {len(valid_candidates)} matching candidates."
    }


if __name__ == "__main__":
    test_queries = [
        ("classify banking queries, CPU only, under 20 ms", "fastest"),
        ("bank chatbot on server CPU with max accuracy", "max_accuracy"),
        ("mobile edge banking assistant with < 130mb memory", "edge"),
        ("classify banking queries, under 5 ms", "fastest")  # Should trigger no_fit
    ]

    print("=" * 80)
    print("COMPRESSROUTE SELECTOR DEMO")
    print("=" * 80)
    for q, preset in test_queries:
        print(f"\nUser Query: '{q}' (Preset: {preset})")
        res = select(q, preset=preset)
        print(f"Status: {res['status'].upper()} | Detected Task: {res['detected_task']} ({res['task_confidence']*100:.1f}%)")
        print(f"Constraints: Latency <= {res['constraints']['max_latency_ms']}ms, Memory <= {res['constraints']['max_memory_mb']}MB, HW: {res['constraints']['hardware']}")

        if res["status"] == "success":
            for top in res["top_models"]:
                print(f"  --> {top['explanation']}")
        else:
            print(f"  [NO FIT MESSAGE]:\n{res['explanation']}")
    print("=" * 80)
