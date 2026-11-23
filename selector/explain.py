"""Human-readable explanation generation for model recommendations in CompressRoute.

Produces concrete, data-driven rationales comparing selected models against
the PyTorch Teacher baseline using exact registry metrics.
"""

from typing import Dict, Any, List, Optional
import math


def find_teacher_baseline(all_models: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Finds the PyTorch FP32 Teacher model in registry."""
    for m in all_models:
        if m["name"] == "teacher_bert_pytorch_fp32":
            return m
    return all_models[0] if all_models else None


def generate_recommendation_reason(
    model: Dict[str, Any],
    teacher: Optional[Dict[str, Any]],
    rank: int = 1
) -> str:
    """Generates an intuitive, number-backed explanation for a recommended model."""
    m_lat = model["latency_ms"]["bs1"]["p95"]
    m_thr = model["throughput"]["bs1"]
    m_size = model["size_mb"]
    m_acc = model["accuracy"] * 100
    is_pareto = model.get("is_pareto", False)

    if teacher is None or model["name"] == teacher["name"]:
        return (
            f"Rank #{rank}: Baseline Teacher model offering peak reference accuracy "
            f"({m_acc:.2f}%) with {m_lat:.1f}ms p95 latency and {m_size:.1f}MB footprint."
        )

    t_lat = teacher["latency_ms"]["bs1"]["p95"]
    t_thr = teacher["throughput"]["bs1"]
    t_size = teacher["size_mb"]
    t_acc = teacher["accuracy"] * 100

    speedup = t_lat / max(m_lat, 0.01)
    thr_boost = m_thr / max(t_thr, 0.01)
    size_saving = (1.0 - (m_size / max(t_size, 0.01))) * 100
    acc_diff = m_acc - t_acc

    acc_str = f"+{acc_diff:.2f}% higher" if acc_diff >= 0 else f"{abs(acc_diff):.2f}% less"
    pareto_badge = "★ Pareto-optimal" if is_pareto else "Viable candidate"

    reason = (
        f"Rank #{rank} ({pareto_badge}): {speedup:.1f}x faster than the Teacher "
        f"({m_lat:.1f}ms vs {t_lat:.1f}ms p95) with {thr_boost:.1f}x higher throughput "
        f"({m_thr:.1f} vs {t_thr:.1f} smp/s), {size_saving:.1f}% lower disk footprint "
        f"({m_size:.1f}MB vs {t_size:.1f}MB), and {acc_str} accuracy ({m_acc:.2f}%)."
    )

    return reason


def generate_no_fit_explanation(
    constraints: Dict[str, Any],
    rejections: List[Dict[str, str]],
    all_models: List[Dict[str, Any]]
) -> str:
    """Generates clear, actionable explanation when no model fits the constraints."""
    lines = ["No model fits the specified constraints:"]

    if constraints.get("max_latency_ms") is not None:
        lines.append(f"  * Requested Latency Limit: <= {constraints['max_latency_ms']} ms")
    if constraints.get("max_memory_mb") is not None:
        lines.append(f"  * Requested Memory Limit: <= {constraints['max_memory_mb']} MB")
    if constraints.get("hardware") is not None:
        lines.append(f"  * Target Hardware: {constraints['hardware']}")

    # Identify fastest and smallest models available in registry
    if all_models:
        fastest = min(all_models, key=lambda x: x["latency_ms"]["bs1"]["p95"])
        smallest = min(all_models, key=lambda x: x["size_mb"])

        f_lat = fastest["latency_ms"]["bs1"]["p95"]
        s_size = smallest["size_mb"]

        lines.append(f"\nRegistry limits on current machine:")
        lines.append(f"  * Fastest model available: {fastest['display_name']} ({f_lat:.2f} ms p95)")
        lines.append(f"  * Smallest model available: {smallest['display_name']} ({s_size:.1f} MB)")

        recommendation = []
        if constraints.get("max_latency_ms") and constraints["max_latency_ms"] < f_lat:
            recommendation.append(f"increase latency SLA to at least {math.ceil(f_lat)} ms")
        if constraints.get("max_memory_mb") and constraints["max_memory_mb"] < s_size:
            recommendation.append(f"increase memory allowance to at least {math.ceil(s_size)} MB")

        if recommendation:
            lines.append(f"\nRecommendation: To find viable models, please {' and '.join(recommendation)}.")

    return "\n".join(lines)
