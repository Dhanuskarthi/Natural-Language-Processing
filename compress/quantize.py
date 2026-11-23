"""INT8 Dynamic Quantization and Evaluation for CompressRoute models.

Quantizes ONNX FP32 graphs to INT8 using ONNX Runtime quantization,
evaluates all variants on the Banking77 test split, and generates a
complete comparison table of sizes, compression ratios, and accuracies.
"""

from typing import Dict, Any, List
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from compress.data_utils import load_config, load_banking77, check_and_set_offline
from compress.export_onnx import get_model_size_mb, get_pytorch_dir_size_mb


def evaluate_onnx_model(
    onnx_path: str,
    tokenizer: Any,
    test_texts: List[str],
    test_labels: List[int],
    batch_size: int = 32
) -> Dict[str, float]:
    """Runs inference with ONNX Runtime and computes accuracy and macro-F1."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    all_preds = []

    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="np"
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        logits = session.run(["logits"], ort_inputs)[0]
        preds = np.argmax(logits, axis=-1)
        all_preds.extend(preds)

    all_preds = np.array(all_preds)
    test_labels_arr = np.array(test_labels)

    acc = float(accuracy_score(test_labels_arr, all_preds))
    macro_f1 = float(f1_score(test_labels_arr, all_preds, average="macro", zero_division=0))

    return {
        "accuracy": round(acc, 5),
        "macro_f1": round(macro_f1, 5),
        "test_samples": len(test_labels)
    }


def evaluate_pytorch_model(
    checkpoint_dir: str,
    test_texts: List[str],
    test_labels: List[int],
    batch_size: int = 32
) -> Dict[str, float]:
    """Runs inference with PyTorch and computes accuracy and macro-F1."""
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    test_labels_arr = np.array(test_labels)

    acc = float(accuracy_score(test_labels_arr, all_preds))
    macro_f1 = float(f1_score(test_labels_arr, all_preds, average="macro", zero_division=0))

    return {
        "accuracy": round(acc, 5),
        "macro_f1": round(macro_f1, 5),
        "test_samples": len(test_labels)
    }


def quantize_single_onnx_model(
    input_fp32_path: str,
    output_int8_path: str,
    model_name: str
) -> Dict[str, Any]:
    """Applies dynamic INT8 quantization to an ONNX FP32 model graph."""
    print(f"\n[QUANTIZE] Applying dynamic INT8 quantization to {model_name}...")
    print(f"  Input  : {input_fp32_path}")
    print(f"  Output : {output_int8_path}")

    os.makedirs(os.path.dirname(output_int8_path), exist_ok=True)
    start_time = time.time()

    quantize_dynamic(
        model_input=input_fp32_path,
        model_output=output_int8_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"]
    )

    duration = time.time() - start_time
    fp32_size = get_model_size_mb(input_fp32_path)
    int8_size = get_model_size_mb(output_int8_path)
    compression_ratio = round(fp32_size / max(int8_size, 0.01), 2)

    print(f"[QUANTIZE] Completed in {duration:.2f}s.")
    print(f"[QUANTIZE] FP32 Size: {fp32_size:.2f} MB -> INT8 Size: {int8_size:.2f} MB (Compression: {compression_ratio}x)")

    return {
        "model_name": model_name,
        "input_fp32_path": input_fp32_path,
        "output_int8_path": output_int8_path,
        "fp32_size_mb": fp32_size,
        "int8_size_mb": int8_size,
        "compression_ratio": compression_ratio,
        "quant_time_sec": round(duration, 2)
    }


def run_quantization_and_evaluation(fast: bool = False) -> List[Dict[str, Any]]:
    """Runs INT8 quantization on all exported ONNX models and evaluates all variants."""
    check_and_set_offline()
    config = load_config()
    seed = config.get("project", {}).get("seed", 42)

    # Load test split
    print(f"Loading Banking77 test split for evaluation (fast={fast})...")
    dataset_dict, _ = load_banking77(fast=fast, seed=seed, config=config)
    test_texts = list(dataset_dict["test"]["text"])
    test_labels = list(dataset_dict["test"]["label"])

    models_config = [
        {
            "name": "teacher_bert",
            "family": "BERT-base",
            "layers": 12,
            "pt_dir": "models/teacher_bert",
            "fp32_onnx": "models/onnx/teacher_bert_fp32.onnx",
            "int8_onnx": "models/onnx/teacher_bert_int8.onnx"
        },
        {
            "name": "baseline_distilbert",
            "family": "DistilBERT",
            "layers": 6,
            "pt_dir": "models/baseline_distilbert",
            "fp32_onnx": "models/onnx/baseline_distilbert_fp32.onnx",
            "int8_onnx": "models/onnx/baseline_distilbert_int8.onnx"
        },
        {
            "name": "student_distilled",
            "family": "Student-4L",
            "layers": 4,
            "pt_dir": "models/student_distilled",
            "fp32_onnx": "models/onnx/student_distilled_fp32.onnx",
            "int8_onnx": "models/onnx/student_distilled_int8.onnx"
        }
    ]

    all_variant_results: List[Dict[str, Any]] = []

    # First, perform quantization on all ONNX FP32 models
    for m in models_config:
        if os.path.exists(m["fp32_onnx"]):
            quantize_single_onnx_model(
                input_fp32_path=m["fp32_onnx"],
                output_int8_path=m["int8_onnx"],
                model_name=m["name"]
            )

    teacher_pt_size = get_pytorch_dir_size_mb("models/teacher_bert") if os.path.exists("models/teacher_bert") else 418.0

    # Evaluate each precision format
    for m in models_config:
        tokenizer = AutoTokenizer.from_pretrained(m["pt_dir"]) if os.path.exists(m["pt_dir"]) else None

        # 1. PyTorch FP32
        if os.path.exists(m["pt_dir"]):
            print(f"\n[EVAL] Evaluating {m['name']} (PyTorch FP32)...")
            pt_metrics = evaluate_pytorch_model(m["pt_dir"], test_texts, test_labels)
            pt_size = get_pytorch_dir_size_mb(m["pt_dir"])
            all_variant_results.append({
                "model_id": f"{m['name']}_pytorch_fp32",
                "display_name": f"{m['family']} (PyTorch FP32)",
                "family": m["family"],
                "format": "PyTorch",
                "precision": "FP32",
                "layers": m["layers"],
                "size_mb": pt_size,
                "compression_vs_teacher": round(teacher_pt_size / max(pt_size, 0.01), 2),
                "accuracy": pt_metrics["accuracy"],
                "macro_f1": pt_metrics["macro_f1"],
                "test_samples": pt_metrics["test_samples"]
            })

        # 2. ONNX FP32
        if os.path.exists(m["fp32_onnx"]) and tokenizer is not None:
            print(f"[EVAL] Evaluating {m['name']} (ONNX FP32)...")
            fp32_metrics = evaluate_onnx_model(m["fp32_onnx"], tokenizer, test_texts, test_labels)
            fp32_size = get_model_size_mb(m["fp32_onnx"])
            all_variant_results.append({
                "model_id": f"{m['name']}_onnx_fp32",
                "display_name": f"{m['family']} (ONNX FP32)",
                "family": m["family"],
                "format": "ONNX",
                "precision": "FP32",
                "layers": m["layers"],
                "size_mb": fp32_size,
                "compression_vs_teacher": round(teacher_pt_size / max(fp32_size, 0.01), 2),
                "accuracy": fp32_metrics["accuracy"],
                "macro_f1": fp32_metrics["macro_f1"],
                "test_samples": fp32_metrics["test_samples"]
            })

        # 3. ONNX INT8
        if os.path.exists(m["int8_onnx"]) and tokenizer is not None:
            print(f"[EVAL] Evaluating {m['name']} (ONNX INT8)...")
            int8_metrics = evaluate_onnx_model(m["int8_onnx"], tokenizer, test_texts, test_labels)
            int8_size = get_model_size_mb(m["int8_onnx"])
            all_variant_results.append({
                "model_id": f"{m['name']}_onnx_int8",
                "display_name": f"{m['family']} (ONNX INT8)",
                "family": m["family"],
                "format": "ONNX",
                "precision": "INT8",
                "layers": m["layers"],
                "size_mb": int8_size,
                "compression_vs_teacher": round(teacher_pt_size / max(int8_size, 0.01), 2),
                "accuracy": int8_metrics["accuracy"],
                "macro_f1": int8_metrics["macro_f1"],
                "test_samples": int8_metrics["test_samples"]
            })

    # Save summary table
    results_dir = config.get("paths", {}).get("results_dir", "benchmark/results")
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(all_variant_results)
    csv_path = os.path.join(results_dir, "variants_summary.csv")
    json_path = os.path.join(results_dir, "variants_summary.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_variant_results, f, indent=2)

    print("\n" + "=" * 90)
    print("PHASE 3: COMPREHENSIVE MODEL VARIANTS & QUANTIZATION MATRIX")
    print("=" * 90)
    columns_to_show = ["model_id", "family", "format", "precision", "layers", "size_mb", "compression_vs_teacher", "accuracy", "macro_f1"]
    print(df[columns_to_show].to_string(index=False))
    print("=" * 90)
    print(f"Variant matrix saved to: {csv_path} and {json_path}")

    return all_variant_results


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models to INT8 and evaluate all variants.")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode using fast test split.")
    args = parser.parse_args()

    run_quantization_and_evaluation(fast=args.fast)


if __name__ == "__main__":
    main()
