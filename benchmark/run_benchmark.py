"""Benchmark runner across PyTorch, ONNX FP32, and ONNX INT8 variants.

Measures p50, p95, p99 latency, throughput, file size, and peak memory (psutil)
across multiple batch sizes and sequence lengths with fixed thread allocation.
Generates benchmark/results/registry.json adhering to standard schema.
"""

from typing import Dict, Any, List, Tuple
import argparse
import gc
import json
import os
import time
import numpy as np
import onnxruntime as ort
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from compress.data_utils import load_config, check_and_set_offline
from benchmark.metrics import (
    get_hardware_info,
    get_peak_memory_mb,
    measure_inference_latency,
)
from compress.export_onnx import get_model_size_mb, get_pytorch_dir_size_mb


def create_dummy_batch(
    tokenizer: Any,
    batch_size: int,
    seq_length: int,
    is_numpy: bool = False
) -> Dict[str, Any]:
    """Generates synthetic input tokens of specified batch size and sequence length."""
    input_ids = np.random.randint(100, 20000, size=(batch_size, seq_length), dtype=np.int64)
    attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)

    # Place CLS (101) and SEP (102) tokens
    input_ids[:, 0] = 101
    input_ids[:, -1] = 102

    if is_numpy:
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
    }


def benchmark_pytorch_variant(
    checkpoint_dir: str,
    batch_sizes: List[int],
    seq_lengths: List[int],
    warmup: int,
    timed: int,
    num_threads: int = 4
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], float]:
    """Runs latency and memory benchmarks for a PyTorch FP32 checkpoint."""
    torch.set_num_threads(num_threads)
    device = torch.device("cpu")

    gc.collect()
    mem_before = get_peak_memory_mb()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    mem_after = get_peak_memory_mb()
    peak_mem = max(mem_after, mem_before)

    latencies: Dict[str, Dict[str, float]] = {}
    throughputs: Dict[str, float] = {}

    for bs in batch_sizes:
        for seq in seq_lengths:
            key_full = f"bs{bs}_seq{seq}"
            batch = create_dummy_batch(tokenizer, bs, seq, is_numpy=False)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            def run_step():
                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)

            lat_metrics, thr = measure_inference_latency(
                run_step, batch_size=bs, warmup_runs=warmup, timed_runs=timed
            )
            latencies[key_full] = {
                "p50": lat_metrics["p50"],
                "p95": lat_metrics["p95"],
                "p99": lat_metrics["p99"]
            }
            throughputs[key_full] = thr

        # Standard shorthand for default sequence length 32
        latencies[f"bs{bs}"] = latencies[f"bs{bs}_seq32"]
        throughputs[f"bs{bs}"] = throughputs[f"bs{bs}_seq32"]

    del model
    gc.collect()
    return latencies, throughputs, peak_mem


def benchmark_onnx_variant(
    onnx_path: str,
    tokenizer_dir: str,
    batch_sizes: List[int],
    seq_lengths: List[int],
    warmup: int,
    timed: int,
    num_threads: int = 4
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], float]:
    """Runs latency and memory benchmarks for an ONNX Runtime model."""
    gc.collect()
    mem_before = get_peak_memory_mb()

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_threads
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    mem_after = get_peak_memory_mb()
    peak_mem = max(mem_after, mem_before)

    latencies: Dict[str, Dict[str, float]] = {}
    throughputs: Dict[str, float] = {}

    for bs in batch_sizes:
        for seq in seq_lengths:
            key_full = f"bs{bs}_seq{seq}"
            batch = create_dummy_batch(tokenizer, bs, seq, is_numpy=True)
            ort_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"]
            }

            def run_step():
                _ = session.run(["logits"], ort_inputs)

            lat_metrics, thr = measure_inference_latency(
                run_step, batch_size=bs, warmup_runs=warmup, timed_runs=timed
            )
            latencies[key_full] = {
                "p50": lat_metrics["p50"],
                "p95": lat_metrics["p95"],
                "p99": lat_metrics["p99"]
            }
            throughputs[key_full] = thr

        latencies[f"bs{bs}"] = latencies[f"bs{bs}_seq32"]
        throughputs[f"bs{bs}"] = throughputs[f"bs{bs}_seq32"]

    del session
    gc.collect()
    return latencies, throughputs, peak_mem


def run_comprehensive_benchmark(fast: bool = False) -> List[Dict[str, Any]]:
    """Runs the benchmark across all models and saves registry.json."""
    check_and_set_offline()
    config = load_config()

    num_threads = config.get("benchmark", {}).get("threads", 4)
    hardware_meta = get_hardware_info(num_threads=num_threads)

    if fast:
        batch_sizes = [1, 8, 32]
        seq_lengths = [32, 128]
        warmup_runs = 5
        timed_runs = 25
    else:
        batch_sizes = config.get("benchmark", {}).get("batch_sizes", [1, 8, 32, 64])
        seq_lengths = config.get("benchmark", {}).get("sequence_lengths", [32, 128])
        warmup_runs = config.get("benchmark", {}).get("warmup_runs", 20)
        timed_runs = config.get("benchmark", {}).get("timed_runs", 100)

    print("=" * 80)
    print(f"COMPRESSROUTE SYSTEM BENCHMARK (Fast Mode: {fast}, Threads: {num_threads})")
    print(f"CPU: {hardware_meta['cpu']} | Cores: {hardware_meta['physical_cores']} Phys, {hardware_meta['logical_cores']} Log")
    print(f"Batch Sizes: {batch_sizes} | Sequence Lengths: {seq_lengths} | Runs: {timed_runs}")
    print("=" * 80)

    # Load evaluated accuracy from variants_summary.json
    results_dir = config.get("paths", {}).get("results_dir", "benchmark/results")
    variants_summary_path = os.path.join(results_dir, "variants_summary.json")

    acc_lookup = {}
    if os.path.exists(variants_summary_path):
        with open(variants_summary_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                acc_lookup[item["model_id"]] = {
                    "accuracy": item["accuracy"],
                    "macro_f1": item["macro_f1"]
                }

    model_targets = [
        # 1. Teacher BERT
        {
            "id": "teacher_bert_pytorch_fp32",
            "name": "BERT-base Teacher (PyTorch FP32)",
            "family": "BERT-base",
            "precision": "FP32",
            "type": "pytorch",
            "path": "models/teacher_bert",
            "tok_dir": "models/teacher_bert"
        },
        {
            "id": "teacher_bert_onnx_fp32",
            "name": "BERT-base Teacher (ONNX FP32)",
            "family": "BERT-base",
            "precision": "FP32",
            "type": "onnx",
            "path": "models/onnx/teacher_bert_fp32.onnx",
            "tok_dir": "models/teacher_bert"
        },
        {
            "id": "teacher_bert_onnx_int8",
            "name": "BERT-base Teacher (ONNX INT8)",
            "family": "BERT-base",
            "precision": "INT8",
            "type": "onnx",
            "path": "models/onnx/teacher_bert_int8.onnx",
            "tok_dir": "models/teacher_bert"
        },
        # 2. Baseline DistilBERT
        {
            "id": "baseline_distilbert_pytorch_fp32",
            "name": "DistilBERT Baseline (PyTorch FP32)",
            "family": "DistilBERT",
            "precision": "FP32",
            "type": "pytorch",
            "path": "models/baseline_distilbert",
            "tok_dir": "models/baseline_distilbert"
        },
        {
            "id": "baseline_distilbert_onnx_fp32",
            "name": "DistilBERT Baseline (ONNX FP32)",
            "family": "DistilBERT",
            "precision": "FP32",
            "type": "onnx",
            "path": "models/onnx/baseline_distilbert_fp32.onnx",
            "tok_dir": "models/baseline_distilbert"
        },
        {
            "id": "baseline_distilbert_onnx_int8",
            "name": "DistilBERT Baseline (ONNX INT8)",
            "family": "DistilBERT",
            "precision": "INT8",
            "type": "onnx",
            "path": "models/onnx/baseline_distilbert_int8.onnx",
            "tok_dir": "models/baseline_distilbert"
        },
        # 3. Student Distilled
        {
            "id": "student_distilled_pytorch_fp32",
            "name": "Distilled Student 4L (PyTorch FP32)",
            "family": "Student-4L",
            "precision": "FP32",
            "type": "pytorch",
            "path": "models/student_distilled",
            "tok_dir": "models/student_distilled"
        },
        {
            "id": "student_distilled_onnx_fp32",
            "name": "Distilled Student 4L (ONNX FP32)",
            "family": "Student-4L",
            "precision": "FP32",
            "type": "onnx",
            "path": "models/onnx/student_distilled_fp32.onnx",
            "tok_dir": "models/student_distilled"
        },
        {
            "id": "student_distilled_onnx_int8",
            "name": "Distilled Student 4L (ONNX INT8)",
            "family": "Student-4L",
            "precision": "INT8",
            "type": "onnx",
            "path": "models/onnx/student_distilled_int8.onnx",
            "tok_dir": "models/student_distilled"
        }
    ]

    registry_entries: List[Dict[str, Any]] = []

    for tgt in model_targets:
        if not os.path.exists(tgt["path"]):
            print(f"[SKIP] Target not found: {tgt['path']}")
            continue

        print(f"\n--> Benchmarking: {tgt['name']}...")
        if tgt["type"] == "pytorch":
            latencies, throughputs, peak_mem = benchmark_pytorch_variant(
                checkpoint_dir=tgt["path"],
                batch_sizes=batch_sizes,
                seq_lengths=seq_lengths,
                warmup=warmup_runs,
                timed=timed_runs,
                num_threads=num_threads
            )
            size_mb = get_pytorch_dir_size_mb(tgt["path"])
        else:
            latencies, throughputs, peak_mem = benchmark_onnx_variant(
                onnx_path=tgt["path"],
                tokenizer_dir=tgt["tok_dir"],
                batch_sizes=batch_sizes,
                seq_lengths=seq_lengths,
                warmup=warmup_runs,
                timed=timed_runs,
                num_threads=num_threads
            )
            size_mb = get_model_size_mb(tgt["path"])

        acc_info = acc_lookup.get(tgt["id"], {"accuracy": 0.0, "macro_f1": 0.0})

        entry = {
            "name": tgt["id"],
            "display_name": tgt["name"],
            "family": tgt["family"],
            "precision": tgt["precision"],
            "task": "intent_classification",
            "dataset": "banking77",
            "accuracy": acc_info["accuracy"],
            "macro_f1": acc_info["macro_f1"],
            "size_mb": size_mb,
            "peak_mem_mb": peak_mem,
            "latency_ms": latencies,
            "throughput": throughputs,
            "hardware": hardware_meta
        }

        registry_entries.append(entry)
        p50_bs1 = latencies["bs1"]["p50"]
        p95_bs1 = latencies["bs1"]["p95"]
        thr_bs1 = throughputs["bs1"]
        print(f"    Size: {size_mb} MB | Peak RSS: {peak_mem} MB | Latency (BS1): p50={p50_bs1}ms, p95={p95_bs1}ms | Thr: {thr_bs1} smp/s")

    # Save to registry.json
    registry_path = os.path.join(results_dir, "registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry_entries, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE: {len(registry_entries)} variants registered in {registry_path}")
    print("=" * 80)

    return registry_entries


def main():
    parser = argparse.ArgumentParser(description="Run standardized benchmark across all model variants.")
    parser.add_argument("--fast", action="store_true", help="Fast benchmark mode for rapid verification.")
    args = parser.parse_args()

    run_comprehensive_benchmark(fast=args.fast)


if __name__ == "__main__":
    main()
