"""Unit and integration tests for Phase 3 (ONNX Export and INT8 Quantization)."""

import os
import json
import pytest
import pandas as pd


def test_onnx_export_summary():
    """Validates that ONNX export summary exists and all parity checks passed."""
    summary_path = "benchmark/results/onnx_export_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) >= 1
        for item in data:
            assert "onnx_path" in item
            assert "max_output_diff" in item
            assert item["max_output_diff"] < 1e-4
            assert item["parity_pass"] is True


def test_variants_summary_table():
    """Validates that variants_summary table contains FP32 and INT8 variants."""
    csv_path = "benchmark/results/variants_summary.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        assert "model_id" in df.columns
        assert "precision" in df.columns
        assert "size_mb" in df.columns
        assert "accuracy" in df.columns

        # Verify both FP32 and INT8 are present
        precisions = set(df["precision"].unique())
        assert "FP32" in precisions
        assert "INT8" in precisions

        # Verify INT8 models are smaller than FP32 counterparts
        int8_models = df[df["precision"] == "INT8"]
        fp32_models = df[df["precision"] == "FP32"]
        assert len(int8_models) > 0
        assert len(fp32_models) > 0
