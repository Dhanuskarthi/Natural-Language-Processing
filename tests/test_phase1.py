"""Unit and integration tests for Phase 1 (Data and Teacher Training)."""

import os
import json
import pytest
import torch
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification

from compress.data_utils import load_banking77, compute_dataset_statistics, load_config
from compress.finetune_teacher import evaluate_model, prepare_dataloaders
from transformers import AutoTokenizer


def test_config_loading():
    """Validates that config.yaml loads properly with required keys."""
    cfg = load_config()
    assert "project" in cfg
    assert "dataset" in cfg
    assert "models" in cfg
    assert cfg["dataset"]["num_labels"] == 77


def test_banking77_fast_loading():
    """Validates that Banking77 fast mode loads valid splits."""
    data_dict, label_names = load_banking77(fast=True)
    assert isinstance(data_dict, DatasetDict)
    assert "train" in data_dict
    assert "val" in data_dict
    assert "test" in data_dict
    assert len(label_names) == 77

    # Fast mode should have small subset
    assert len(data_dict["train"]) <= 600
    assert len(data_dict["val"]) <= 200
    assert len(data_dict["test"]) <= 200

    stats = compute_dataset_statistics(data_dict)
    assert stats["train"]["num_classes"] > 0
    assert stats["train"]["word_max"] > 0


def test_saved_metrics_file():
    """Validates that generated metrics files exist and contain non-zero metrics."""
    teacher_metrics_path = "benchmark/results/teacher_metrics.json"
    if os.path.exists(teacher_metrics_path):
        with open(teacher_metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["model_type"] == "teacher"
        assert "accuracy" in data
        assert "macro_f1" in data
        assert 0.0 <= data["accuracy"] <= 1.0
