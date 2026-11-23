"""ONNX Export and Parity Verification for CompressRoute models.

Exports fine-tuned PyTorch checkpoints to ONNX FP32 format with dynamic axes
and verifies numerical parity against PyTorch forward outputs.
"""

from typing import Dict, Any, List
import argparse
import json
import os
import time
import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from compress.data_utils import load_config, check_and_set_offline


def get_model_size_mb(file_path: str) -> float:
    """Returns total file size in megabytes including any external tensor data files."""
    if not os.path.exists(file_path):
        return 0.0
    total_bytes = os.path.getsize(file_path)
    base_dir = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    for sibling in os.listdir(base_dir):
        if sibling != base_name and sibling.startswith(base_name) and (sibling.endswith(".data") or sibling.endswith(".onnx_data")):
            total_bytes += os.path.getsize(os.path.join(base_dir, sibling))
    return round(total_bytes / (1024 * 1024), 2)


def get_pytorch_dir_size_mb(dir_path: str) -> float:
    """Returns total directory size in megabytes for PyTorch model weights."""
    total_bytes = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt")):
                total_bytes += os.path.getsize(os.path.join(root, f))
    return round(total_bytes / (1024 * 1024), 2)


def export_single_model_to_onnx(
    checkpoint_dir: str,
    output_onnx_path: str,
    model_name: str,
    opset_version: int = 17
) -> Dict[str, Any]:
    """Exports a single PyTorch checkpoint to ONNX FP32 and verifies parity."""
    print(f"\n[EXPORT] Exporting {model_name} from '{checkpoint_dir}' -> '{output_onnx_path}'...")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()

    # Create dummy inputs with variable sequence length
    dummy_text = "I lost my debit card and need to cancel it immediately."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }

    # Export to ONNX
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True
    )
    print(f"[EXPORT] Successfully generated ONNX model at: {output_onnx_path}")

    # Validate ONNX graph structure
    onnx_proto = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_proto)
    print("[EXPORT] ONNX model graph structure verified successfully.")

    # Numerical parity verification against PyTorch
    print("[PARITY] Verifying numerical output parity across test sentences...")
    ort_session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])

    test_sentences = [
        "How do I transfer money abroad?",
        "Why is my card blocked?",
        "What are the exchange rates today?",
        "I want to change my PIN.",
        "Can I open a joint account online?"
    ]

    max_diff = 0.0
    for text in test_sentences:
        test_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            pt_outputs = model(**test_inputs)
            pt_logits = pt_outputs.logits.detach().cpu().numpy()

        ort_inputs = {
            "input_ids": test_inputs["input_ids"].numpy(),
            "attention_mask": test_inputs["attention_mask"].numpy()
        }
        ort_logits = ort_session.run(["logits"], ort_inputs)[0]

        diff = float(np.max(np.abs(pt_logits - ort_logits)))
        if diff > max_diff:
            max_diff = diff

    onnx_size_mb = get_model_size_mb(output_onnx_path)
    pt_size_mb = get_pytorch_dir_size_mb(checkpoint_dir)

    print(f"[PARITY] Maximum absolute difference across tests: {max_diff:.3e}")
    print(f"[SIZE] PyTorch Weights: {pt_size_mb:.2f} MB | ONNX FP32: {onnx_size_mb:.2f} MB")

    assert max_diff < 1e-4, f"Parity mismatch: max difference {max_diff:.3e} exceeds tolerance 1e-4."
    print(f"[PARITY] PASS: Numerical parity verified within tolerance (max_diff={max_diff:.3e} < 1e-4).")

    return {
        "model_name": model_name,
        "checkpoint_dir": checkpoint_dir,
        "onnx_path": output_onnx_path,
        "precision": "FP32",
        "pytorch_size_mb": pt_size_mb,
        "onnx_size_mb": onnx_size_mb,
        "max_output_diff": round(max_diff, 8),
        "parity_pass": max_diff < 1e-4,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


def export_all_models() -> List[Dict[str, Any]]:
    """Exports Teacher, Baseline, and Distilled Student to ONNX FP32."""
    check_and_set_offline()
    config = load_config()

    models_to_export = [
        {
            "name": "teacher_bert",
            "dir": config.get("models", {}).get("teacher", {}).get("checkpoint_dir", "models/teacher_bert"),
            "output": "models/onnx/teacher_bert_fp32.onnx"
        },
        {
            "name": "baseline_distilbert",
            "dir": config.get("models", {}).get("baseline", {}).get("checkpoint_dir", "models/baseline_distilbert"),
            "output": "models/onnx/baseline_distilbert_fp32.onnx"
        },
        {
            "name": "student_distilled",
            "dir": config.get("models", {}).get("student", {}).get("checkpoint_dir", "models/student_distilled"),
            "output": "models/onnx/student_distilled_fp32.onnx"
        }
    ]

    export_results = []
    for item in models_to_export:
        if os.path.exists(item["dir"]):
            res = export_single_model_to_onnx(
                checkpoint_dir=item["dir"],
                output_onnx_path=item["output"],
                model_name=item["name"]
            )
            export_results.append(res)
        else:
            print(f"[WARN] Checkpoint directory not found: {item['dir']}. Skipping.")

    results_dir = config.get("paths", {}).get("results_dir", "benchmark/results")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "onnx_export_summary.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(export_results, f, indent=2)

    print(f"\n[SUMMARY] All ONNX FP32 exports complete. Summary saved to: {summary_path}")
    return export_results


if __name__ == "__main__":
    export_all_models()
