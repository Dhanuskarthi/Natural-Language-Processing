"""Standardized hardware inspection and performance metrics profiling for CompressRoute.

Profiles latency distributions (p50, p95, p99), inference throughput (samples/sec),
peak process memory footprint (RSS via psutil), and captures CPU system metadata.
"""

from typing import Dict, Any, Callable, List, Tuple
import os
import platform
import time
import numpy as np
import psutil
import torch


def get_hardware_info(num_threads: int = 4) -> Dict[str, Any]:
    """Captures CPU, memory, OS, and thread settings of the benchmarking machine."""
    cpu_name = platform.processor() or "Unknown CPU"
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        pass

    return {
        "cpu": cpu_name,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "threads": num_threads,
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "python_version": platform.python_version()
    }


def get_peak_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in megabytes."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def measure_inference_latency(
    run_fn: Callable[[], Any],
    batch_size: int,
    warmup_runs: int = 20,
    timed_runs: int = 100
) -> Tuple[Dict[str, float], float]:
    """Measures latency distribution (p50, p95, p99) and throughput for an inference function.

    Args:
        run_fn: Zero-argument callable executing one inference batch.
        batch_size: Number of sequences in the batch.
        warmup_runs: Iterations to discard for cache/engine warming.
        timed_runs: Number of timed iterations to collect.

    Returns:
        Tuple of (dict with p50, p95, p99 in ms, throughput in samples/sec).
    """
    # 1. Warm-up runs
    for _ in range(warmup_runs):
        run_fn()

    # 2. Timed runs
    latencies_ms: List[float] = []
    start_total = time.perf_counter()

    for _ in range(timed_runs):
        t0 = time.perf_counter()
        run_fn()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    total_time_sec = time.perf_counter() - start_total
    total_samples = batch_size * timed_runs
    throughput = total_samples / max(total_time_sec, 1e-6)

    latency_percentiles = {
        "p50": round(float(np.percentile(latencies_ms, 50)), 2),
        "p95": round(float(np.percentile(latencies_ms, 95)), 2),
        "p99": round(float(np.percentile(latencies_ms, 99)), 2),
        "mean": round(float(np.mean(latencies_ms)), 2),
        "std": round(float(np.std(latencies_ms)), 2)
    }

    return latency_percentiles, round(throughput, 2)
