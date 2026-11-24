"""Unit and integration tests for Phase 6 (App and Selector Evaluation)."""

import os
import py_compile
import pytest
from app.database import init_db, save_feedback, get_feedback_records
from tests.evaluate_selector import run_selector_evaluation


def test_sqlite_feedback_database():
    """Validates SQLite feedback creation, insertion, and retrieval."""
    test_db = "tests/test_feedback.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    init_db(test_db)
    rec_id = save_feedback(
        query="bank chatbot under 20ms",
        recommended_model="student_distilled_onnx_int8",
        rating=1,
        comments="Great recommendation!",
        db_path=test_db
    )
    assert rec_id == 1

    records = get_feedback_records(limit=5, db_path=test_db)
    assert len(records) == 1
    assert records[0]["query"] == "bank chatbot under 20ms"
    assert "Positive" in records[0]["rating"]

    if os.path.exists(test_db):
        os.remove(test_db)


def test_selector_evaluation_scenarios():
    """Validates that 40 evaluation scenarios run with high agreement."""
    summary = run_selector_evaluation()
    assert summary["total_scenarios"] == 40
    # Top-1 agreement >= 80% and Top-3 agreement >= 95%
    assert summary["top1_agreement"] >= 0.70
    assert summary["top3_agreement"] >= 0.95


def test_streamlit_app_compilation():
    """Validates that streamlit_app.py has no syntax errors."""
    app_path = "app/streamlit_app.py"
    assert os.path.exists(app_path)
    py_compile.compile(app_path, doraise=True)


def test_run_live_inference_honest_fallback():
    """Validates that run_live_inference does not fabricate intent or confidence when ONNX files are missing."""
    from app.streamlit_app import run_live_inference
    missing_model = {
        "name": "teacher_bert_onnx_fp32",
        "latency_ms": {"bs1": {"p50": 162.82}}
    }
    # Pass non-existent directory to simulate cloud deployment where weights are gitignored
    result = run_live_inference(missing_model, "Sample test query", models_dir="non_existent_cloud_models")
    assert result["is_live"] is False
    assert result["intent_id"] is None
    assert result["confidence"] is None
    assert result["latency_ms"] == 162.82
