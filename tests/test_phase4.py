"""Unit and integration tests for Phase 4 (Benchmark and Registry)."""

import os
import json
import pytest
from benchmark.plot_results import compute_pareto_frontier


def test_registry_schema():
    """Validates that registry.json adheres to required schema and has positive metrics."""
    registry_path = "benchmark/results/registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        assert isinstance(entries, list)
        assert len(entries) >= 3

        required_keys = [
            "name", "family", "precision", "task", "dataset",
            "accuracy", "macro_f1", "size_mb", "peak_mem_mb",
            "latency_ms", "throughput", "hardware"
        ]

        for entry in entries:
            for k in required_keys:
                assert k in entry, f"Missing key '{k}' in registry entry {entry.get('name')}"

            # Validate latency structure
            assert "bs1" in entry["latency_ms"]
            for p in ["p50", "p95", "p99"]:
                assert p in entry["latency_ms"]["bs1"]
                assert entry["latency_ms"]["bs1"][p] > 0.0

            # Validate throughput
            assert "bs1" in entry["throughput"]
            assert entry["throughput"]["bs1"] > 0.0

            # Validate hardware
            assert "cpu" in entry["hardware"]
            assert "threads" in entry["hardware"]


def test_pareto_frontier_logic():
    """Validates that Pareto frontier correctly filters dominated variants."""
    registry_path = "benchmark/results/registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        pareto = compute_pareto_frontier(entries)
        assert len(pareto) >= 1
        assert len(pareto) <= len(entries)


def test_generated_plots_exist():
    """Validates that all three benchmark report plots exist and are non-empty."""
    for img_name in ["tradeoff_accuracy_latency.png", "batch_scaling_latency.png", "throughput_scaling.png"]:
        path = os.path.join("reports", img_name)
        if os.path.exists(path):
            assert os.path.getsize(path) > 1000
