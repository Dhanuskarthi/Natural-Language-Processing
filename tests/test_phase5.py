"""Unit and integration tests for Phase 5 (Model Selector and Routing Engine)."""

import os
import pytest
from selector.task_classifier import TaskClassifier
from selector.constraint_parser import parse_constraints, CONSTRAINT_TEST_PHRASES
from selector.scoring import filter_candidates, score_candidates
from selector.pareto import is_dominated, filter_pareto_frontier
from selector.route import select, load_registry


def test_task_classifier_accuracy():
    """Validates that TaskClassifier achieves high accuracy on test set."""
    clf = TaskClassifier()
    metrics = clf.train_and_evaluate()
    assert metrics["test_accuracy"] >= 0.85
    assert len(metrics["classes"]) == 5

    task, conf = clf.predict("classify banking customer queries for our chatbot")
    assert task == "intent_classification"
    assert conf > 0.5


def test_constraint_parser_suite():
    """Validates that constraint parser passes at least 30 distinct phrases."""
    assert len(CONSTRAINT_TEST_PHRASES) >= 30
    passed = test_phrase_suite()
    assert passed >= 30


def test_selector_banking_query_under_20ms():
    """Validates 'classify banking queries, CPU only, under 20 ms'.

    Requirement: Returns a sensible model with reasons, and a tight limit
    never returns the heavy teacher model.
    """
    res = select("classify banking queries, CPU only, under 20 ms", preset="fastest")
    assert res["status"] == "success"
    assert res["detected_task"] == "intent_classification"
    assert res["constraints"]["hardware"] == "CPU"
    assert res["constraints"]["max_latency_ms"] == 20.0

    # Ensure top models are under 20ms
    assert len(res["top_models"]) > 0
    top_model = res["top_models"][0]
    assert top_model["latency_ms"]["bs1"]["p95"] <= 20.0

    # Strict check: NEVER return teacher model (p95 > 190ms)
    returned_names = [m["name"] for m in res["top_models"]]
    assert "teacher_bert_pytorch_fp32" not in returned_names
    assert "teacher_bert_onnx_fp32" not in returned_names


def test_selector_tight_limit_no_fit():
    """Validates that an impossible limit (under 5 ms on CPU) returns no_fit."""
    res = select("classify banking queries on CPU under 5 ms")
    assert res["status"] == "no_fit"
    assert "No model fits" in res["explanation"]
    assert len(res["top_models"]) == 0
