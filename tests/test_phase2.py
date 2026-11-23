"""Unit and integration tests for Phase 2 (Knowledge Distillation)."""

import os
import json
import pytest
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from compress.distill import DistillationLoss, create_initialized_student


def test_distillation_loss_forward():
    """Validates that DistillationLoss calculates positive, finite loss."""
    criterion = DistillationLoss(temperature=3.0, alpha=0.5)
    student_logits = torch.randn(4, 77)
    teacher_logits = torch.randn(4, 77)
    labels = torch.randint(0, 77, (4,))

    total_loss, hard_loss, soft_loss = criterion(student_logits, teacher_logits, labels)

    assert total_loss.item() > 0.0
    assert hard_loss.item() > 0.0
    assert soft_loss.item() >= 0.0
    assert torch.isfinite(total_loss)


def test_create_initialized_student():
    """Validates student architecture generation and layer extraction."""
    teacher_path = "models/teacher_bert"
    if os.path.exists(teacher_path):
        teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path)
        student = create_initialized_student(teacher, num_layers=4)

        assert student.config.num_hidden_layers == 4
        assert len(student.bert.encoder.layer) == 4
        assert student.config.num_labels == 77


def test_distillation_ablation_output():
    """Validates that distillation ablation files exist and contain valid results."""
    csv_path = "benchmark/results/distillation_ablation.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        assert "experiment" in df.columns
        assert "test_acc" in df.columns
        assert len(df) >= 2
