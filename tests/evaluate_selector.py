"""Selector Evaluation Harness for CompressRoute.

Evaluates recommendation consistency and accuracy over 40 diverse test scenarios
(queries, resource limits, and expected optimal models), reporting Top-1 and Top-3 agreement.
"""

from typing import List, Dict, Any
import json
import os
import pandas as pd
from selector.route import select


EVALUATION_SCENARIOS = [
    # 1-10: Ultra-low latency CPU queries (< 20ms) -> Expect Distilled Student INT8 (14.1ms)
    {
        "query": "classify banking queries, CPU only, under 20 ms",
        "preset": "fastest",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8", "baseline_distilbert_onnx_int8"]
    },
    {
        "query": "real-time banking chatbot under 18ms latency on cpu",
        "preset": "fastest",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8", "baseline_distilbert_onnx_int8"]
    },
    {
        "query": "financial customer intent classifier with latency < 15ms",
        "preset": "fastest",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8"]
    },
    {
        "query": "fastest bank query classification under 20 milliseconds",
        "preset": "fastest",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8", "baseline_distilbert_onnx_int8"]
    },
    {
        "query": "interactive teller assistant sub-16ms on standard cpu",
        "preset": "fastest",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8", "baseline_distilbert_onnx_int8"]
    },
    {
        "query": "intent detection for voice banking under 20ms",
        "preset": "fastest",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8", "baseline_distilbert_onnx_int8"]
    },
    {
        "query": "high throughput query classification under 18ms latency",
        "preset": "fastest",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8", "baseline_distilbert_onnx_int8"]
    },
    {
        "query": "cpu banking intent router with SLA < 19 ms",
        "preset": "fastest",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8", "baseline_distilbert_onnx_int8"]
    },
    {
        "query": "mobile banking backend query filter under 15 milliseconds",
        "preset": "fastest",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8"]
    },
    {
        "query": "fast financial intent classifier on cpu below 20ms",
        "preset": "fastest",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8", "baseline_distilbert_onnx_int8"]
    },

    # 11-20: Edge / Mobile constrained memory (< 130MB - 150MB) -> Expect Distilled Student INT8 or DistilBERT INT8
    {
        "query": "mobile edge banking assistant with < 120mb memory",
        "preset": "edge",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8"]
    },
    {
        "query": "deploy on mobile phone client with memory under 125 MB",
        "preset": "edge",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8"]
    },
    {
        "query": "embedded device banking intent recognition max 135mb",
        "preset": "edge",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8", "baseline_distilbert_onnx_int8"]
    },
    {
        "query": "raspberry pi banking kiosk with size < 130 megabytes",
        "preset": "edge",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8"]
    },
    {
        "query": "on-device customer support model under 120 mb",
        "preset": "edge",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8"]
    },
    {
        "query": "offline mobile banking app with memory < 140mb",
        "preset": "edge",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8", "baseline_distilbert_onnx_int8"]
    },
    {
        "query": "iot edge device banking query router max 125mb memory",
        "preset": "edge",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8"]
    },
    {
        "query": "smart atm interface with memory under 140 megabytes",
        "preset": "edge",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8", "baseline_distilbert_onnx_int8"]
    },
    {
        "query": "edge phone banking assistant with max 130mb footprint",
        "preset": "edge",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8"]
    },
    {
        "query": "embedded pos terminal intent parser size below 120mb",
        "preset": "edge",
        "expected_top1": "student_distilled_onnx_int8",
        "expected_in_top3": ["student_distilled_onnx_int8"]
    },

    # 21-30: Balanced presets with moderate SLA (< 40ms - 50ms) -> Expect DistilBERT ONNX FP32 or Student ONNX FP32/INT8
    {
        "query": "balanced banking intent classifier under 40ms latency",
        "preset": "balanced",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "student_distilled_onnx_int8"]
    },
    {
        "query": "customer service ticket routing with latency < 35ms",
        "preset": "balanced",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "student_distilled_onnx_int8"]
    },
    {
        "query": "banking intent detection with moderate SLA under 50ms on cpu",
        "preset": "balanced",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "student_distilled_onnx_int8"]
    },
    {
        "query": "classify financial transactions with latency < 45 milliseconds",
        "preset": "balanced",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "student_distilled_onnx_int8"]
    },
    {
        "query": "cloud microservice for banking queries with SLA under 40ms",
        "preset": "balanced",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "student_distilled_onnx_int8"]
    },
    {
        "query": "general customer intent routing on server cpu under 38ms",
        "preset": "balanced",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "student_distilled_onnx_int8"]
    },
    {
        "query": "balanced accuracy and speed for banking queries under 45ms",
        "preset": "balanced",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "student_distilled_onnx_int8"]
    },
    {
        "query": "production banking bot with latency below 42ms on cpu",
        "preset": "balanced",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "student_distilled_onnx_int8"]
    },
    {
        "query": "financial dialogue understanding with under 36ms response time",
        "preset": "balanced",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "student_distilled_onnx_int8"]
    },
    {
        "query": "server cpu query routing under 50ms",
        "preset": "balanced",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "student_distilled_onnx_int8"]
    },

    # 31-40: Max Accuracy queries on Server CPU (relaxed or no latency limit) -> Expect DistilBERT or Student ONNX FP32
    {
        "query": "bank chatbot on server CPU with max accuracy",
        "preset": "max_accuracy",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "baseline_distilbert_pytorch_fp32"]
    },
    {
        "query": "accurate banking intent classification on server cpu",
        "preset": "max_accuracy",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "baseline_distilbert_pytorch_fp32"]
    },
    {
        "query": "highest quality customer intent recognition on cpu server",
        "preset": "max_accuracy",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "baseline_distilbert_pytorch_fp32"]
    },
    {
        "query": "maximum precision banking query classification no strict latency",
        "preset": "max_accuracy",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "baseline_distilbert_pytorch_fp32"]
    },
    {
        "query": "accurate dispute query categorization on cpu server",
        "preset": "max_accuracy",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "baseline_distilbert_pytorch_fp32"]
    },
    {
        "query": "best accuracy banking support bot on enterprise cpu",
        "preset": "max_accuracy",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "baseline_distilbert_pytorch_fp32"]
    },
    {
        "query": "top accuracy classification for wire transfer queries",
        "preset": "max_accuracy",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "baseline_distilbert_pytorch_fp32"]
    },
    {
        "query": "high quality financial intent classification on cpu with max 250mb",
        "preset": "max_accuracy",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32"]
    },
    {
        "query": "accurate customer intent detection for online banking portal",
        "preset": "max_accuracy",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "baseline_distilbert_pytorch_fp32"]
    },
    {
        "query": "enterprise banking query classification with best accuracy",
        "preset": "max_accuracy",
        "expected_top1": "student_distilled_onnx_fp32",
        "expected_in_top3": ["student_distilled_onnx_fp32", "baseline_distilbert_onnx_fp32", "baseline_distilbert_pytorch_fp32"]
    }
]


def run_selector_evaluation() -> Dict[str, Any]:
    """Runs all 40 scenarios through the selector and computes Top-1 and Top-3 agreement."""
    results = []
    top1_correct = 0
    top3_correct = 0

    print("=" * 80)
    print(f"RUNNING SELECTOR EVALUATION SUITE ({len(EVALUATION_SCENARIOS)} Test Scenarios)")
    print("=" * 80)

    for idx, sc in enumerate(EVALUATION_SCENARIOS, start=1):
        query = sc["query"]
        preset = sc["preset"]
        exp_top1 = sc["expected_top1"]
        exp_top3 = sc["expected_in_top3"]

        res = select(query, preset=preset)
        if res["status"] != "success":
            actual_top1 = "no_fit"
            actual_top3 = []
            hit_top1 = False
            hit_top3 = False
        else:
            top_names = [m["name"] for m in res["top_models"]]
            actual_top1 = top_names[0]
            actual_top3 = top_names

            # Top-1 matches expected
            hit_top1 = (actual_top1 == exp_top1)
            # Any model in top_3 matches the expected set
            hit_top3 = any(name in exp_top3 for name in actual_top3)

        if hit_top1:
            top1_correct += 1
        if hit_top3:
            top3_correct += 1

        results.append({
            "id": idx,
            "query": query,
            "preset": preset,
            "expected_top1": exp_top1,
            "actual_top1": actual_top1,
            "top1_match": hit_top1,
            "top3_match": hit_top3,
            "actual_top3": ", ".join(actual_top3)
        })

    top1_acc = top1_correct / len(EVALUATION_SCENARIOS)
    top3_acc = top3_correct / len(EVALUATION_SCENARIOS)

    summary = {
        "total_scenarios": len(EVALUATION_SCENARIOS),
        "top1_correct": top1_correct,
        "top1_agreement": round(top1_acc, 4),
        "top3_correct": top3_correct,
        "top3_agreement": round(top3_acc, 4)
    }

    # Save to benchmark/results
    results_dir = "benchmark/results"
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "selector_evaluation.json")
    csv_path = os.path.join(results_dir, "selector_evaluation.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "scenarios": results}, f, indent=2)

    pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"Top-1 Agreement: {top1_acc * 100:.1f}% ({top1_correct}/{len(EVALUATION_SCENARIOS)})")
    print(f"Top-3 Agreement: {top3_acc * 100:.1f}% ({top3_correct}/{len(EVALUATION_SCENARIOS)})")
    print(f"Saved evaluation report to {json_path} and {csv_path}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    run_selector_evaluation()
