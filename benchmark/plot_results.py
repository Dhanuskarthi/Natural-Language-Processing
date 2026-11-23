"""Trade-off charting and Pareto efficiency visualization for CompressRoute.

Identifies non-dominated models across Accuracy, Latency (p95), and Memory footprint,
and renders publication-ready trade-off and batch scaling charts to reports/.
"""

from typing import List, Dict, Any
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def compute_pareto_frontier(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identifies models that are Pareto non-dominated on (Acc, Latency_p95, Size_MB).

    A model A dominates model B if:
      - A.accuracy >= B.accuracy
      - A.latency_p95 <= B.latency_p95
      - A.size_mb <= B.size_mb
      - With at least one strict inequality.
    """
    pareto_optimal = []

    for i, a in enumerate(entries):
        a_acc = a["accuracy"]
        a_lat = a["latency_ms"]["bs1"]["p95"]
        a_size = a["size_mb"]

        dominated = False
        for j, b in enumerate(entries):
            if i == j:
                continue
            b_acc = b["accuracy"]
            b_lat = b["latency_ms"]["bs1"]["p95"]
            b_size = b["size_mb"]

            # Check if b dominates a
            if (b_acc >= a_acc and b_lat <= a_lat and b_size <= a_size) and (
                b_acc > a_acc or b_lat < a_lat or b_size < a_size
            ):
                dominated = True
                break

        if not dominated:
            pareto_optimal.append(a)

    return pareto_optimal


def generate_all_plots(registry_path: str = "benchmark/results/registry.json", output_dir: str = "reports") -> None:
    """Generates trade-off charts, Pareto highlights, and batch scaling curves."""
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Registry file not found at {registry_path}. Run benchmark first.")

    os.makedirs(output_dir, exist_ok=True)
    with open(registry_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    pareto_models = compute_pareto_frontier(entries)
    pareto_names = set(m["name"] for m in pareto_models)

    print(f"\n[PARETO] Identified {len(pareto_models)} non-dominated Pareto frontier models:")
    for p in pareto_models:
        print(f"  * {p['display_name']} -> Acc: {p['accuracy']*100:.2f}%, Lat(p95): {p['latency_ms']['bs1']['p95']}ms, Size: {p['size_mb']}MB")

    # Set publication styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    # --------------------------------------------------------------------------
    # 1. Trade-off Chart: Accuracy vs p95 Latency (Bubble size = Memory / Size)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    color_map = {
        "BERT-base": "#1f77b4",
        "DistilBERT": "#ff7f0e",
        "Student-4L": "#2ca02c"
    }

    for item in entries:
        lat = item["latency_ms"]["bs1"]["p95"]
        acc = item["accuracy"] * 100
        size = item["size_mb"]
        family = item["family"]
        is_pareto = item["name"] in pareto_names

        bubble_size = max(80, int(size * 1.5))
        color = color_map.get(family, "#7f7f7f")
        edge_color = "red" if is_pareto else "black"
        edge_width = 2.5 if is_pareto else 0.8

        ax.scatter(
            lat, acc,
            s=bubble_size,
            color=color,
            alpha=0.75,
            edgecolors=edge_color,
            linewidths=edge_width,
            zorder=3
        )

        label_text = f"{item['precision']}\n({size:.0f}MB)"
        if is_pareto:
            label_text = f"★ {item['precision']}\n({size:.0f}MB)"

        ax.annotate(
            label_text,
            (lat, acc),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            fontweight="bold" if is_pareto else "normal"
        )

    # Sort Pareto models by latency to draw frontier curve
    pareto_sorted = sorted(pareto_models, key=lambda x: x["latency_ms"]["bs1"]["p95"])
    if len(pareto_sorted) > 1:
        p_lats = [m["latency_ms"]["bs1"]["p95"] for m in pareto_sorted]
        p_accs = [m["accuracy"] * 100 for m in pareto_sorted]
        ax.plot(p_lats, p_accs, "r--", alpha=0.8, linewidth=1.8, label="Pareto Frontier (Non-dominated)", zorder=2)

    ax.set_title("CompressRoute Model Trade-Off: Accuracy vs. p95 Latency (BS=1)", fontsize=13, fontweight="bold")
    ax.set_xlabel("p95 Latency (ms) [Lower is Better]", fontsize=11)
    ax.set_ylabel("Test Accuracy (%) [Higher is Better]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="BERT-base", markerfacecolor=color_map["BERT-base"], markersize=10),
        Line2D([0], [0], marker="o", color="w", label="DistilBERT", markerfacecolor=color_map["DistilBERT"], markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Student-4L", markerfacecolor=color_map["Student-4L"], markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Pareto Optimal (Red Edge)", markeredgecolor="red", markerfacecolor="gray", markersize=10, markeredgewidth=2),
    ]
    ax.legend(handles=legend_elements, loc="best", frameon=True)

    tradeoff_img_path = os.path.join(output_dir, "tradeoff_accuracy_latency.png")
    fig.tight_layout()
    fig.savefig(tradeoff_img_path)
    plt.close(fig)
    print(f"[PLOT] Trade-off chart saved to: {tradeoff_img_path}")

    # --------------------------------------------------------------------------
    # 2. Batch Scaling Latency Curves
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    batch_keys = [k for k in entries[0]["latency_ms"].keys() if k.startswith("bs") and "_seq" not in k]
    batch_ints = sorted([int(k.replace("bs", "")) for k in batch_keys])

    for item in entries:
        lats = [item["latency_ms"][f"bs{b}"]["p50"] for b in batch_ints if f"bs{b}" in item["latency_ms"]]
        if len(lats) == len(batch_ints):
            style = "-" if "int8" in item["name"] else "--"
            marker = "s" if "int8" in item["name"] else "o"
            ax.plot(batch_ints, lats, linestyle=style, marker=marker, label=item["display_name"], alpha=0.85)

    ax.set_title("Inference Latency (p50) vs. Batch Size", fontsize=13, fontweight="bold")
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("p50 Latency (ms)", fontsize=11)
    ax.set_xticks(batch_ints)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)

    batch_lat_path = os.path.join(output_dir, "batch_scaling_latency.png")
    fig.tight_layout()
    fig.savefig(batch_lat_path)
    plt.close(fig)
    print(f"[PLOT] Batch scaling latency chart saved to: {batch_lat_path}")

    # --------------------------------------------------------------------------
    # 3. Throughput Scaling Curves
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    for item in entries:
        thrs = [item["throughput"][f"bs{b}"] for b in batch_ints if f"bs{b}" in item["throughput"]]
        if len(thrs) == len(batch_ints):
            style = "-" if "int8" in item["name"] else "--"
            marker = "^" if "int8" in item["name"] else "o"
            ax.plot(batch_ints, thrs, linestyle=style, marker=marker, label=item["display_name"], alpha=0.85)

    ax.set_title("Inference Throughput vs. Batch Size", fontsize=13, fontweight="bold")
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Throughput (samples/sec)", fontsize=11)
    ax.set_xticks(batch_ints)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)

    thr_path = os.path.join(output_dir, "throughput_scaling.png")
    fig.tight_layout()
    fig.savefig(thr_path)
    plt.close(fig)
    print(f"[PLOT] Throughput scaling chart saved to: {thr_path}")


if __name__ == "__main__":
    generate_all_plots()
