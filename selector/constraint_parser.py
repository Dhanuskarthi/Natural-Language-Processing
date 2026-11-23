"""Constraint Parser for natural language queries in CompressRoute.

Extracts hardware bounds (CPU, GPU, Edge), maximum latency SLA (ms),
maximum memory limit (MB), and batch size preferences using robust regex rules.
"""

from typing import Dict, Any, Optional
import re


def parse_hardware(text: str) -> Optional[str]:
    """Extracts hardware target: 'CPU', 'GPU', 'Edge', or None."""
    lower = text.lower()

    if re.search(r"\b(edge|mobile|phone|embedded|iot|raspberry\s*pi|on-device|ondevice)\b", lower):
        return "Edge"
    if re.search(r"\b(gpu|cuda|nvidia|v100|t4|a100|rtx)\b", lower):
        return "GPU"
    if re.search(r"\b(cpu|cpu\s*only|x86|intel|amd|server\s*cpu)\b", lower):
        return "CPU"

    return None


def parse_latency_limit(text: str) -> Optional[float]:
    """Extracts latency limit in milliseconds."""
    lower = text.lower()

    # Pattern: under 20 ms, <20ms, <= 25 milliseconds, max 30ms, sub-15ms
    ms_patterns = [
        r"(?:under|below|less\s+than|<|<=|max|maximum|at\s+most|sub-?)\s*(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)\b",
        r"(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)\s*(?:limit|latency|sla|max|or\s+less)?\b",
        r"latency\s*(?:under|below|<|<=|of|at\s+most)?\s*(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)\b"
    ]

    for pat in ms_patterns:
        m = re.search(pat, lower)
        if m:
            return float(m.group(1))

    # Pattern with seconds: under 0.05s, < 0.1 sec
    sec_patterns = [
        r"(?:under|below|less\s+than|<|<=|max|sub-?)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)\b",
    ]
    for pat in sec_patterns:
        m = re.search(pat, lower)
        if m:
            val = float(m.group(1))
            # Convert to ms
            return round(val * 1000.0, 1)

    return None


def parse_memory_limit(text: str) -> Optional[float]:
    """Extracts maximum memory / model size limit in megabytes."""
    lower = text.lower()

    # Pattern in MB
    mb_patterns = [
        r"(?:under|below|less\s+than|<|<=|max|maximum|at\s+most|within)\s*(\d+(?:\.\d+)?)\s*(?:mb|megabytes?|m)\b",
        r"(\d+(?:\.\d+)?)\s*(?:mb|megabytes?)\s*(?:memory|ram|size|limit|max)?\b",
        r"memory\s*(?:under|below|<|<=|of|at\s+most)?\s*(\d+(?:\.\d+)?)\s*(?:mb|megabytes?)\b"
    ]
    for pat in mb_patterns:
        m = re.search(pat, lower)
        if m:
            return float(m.group(1))

    # Pattern in GB: under 1GB -> 1024 MB
    gb_patterns = [
        r"(?:under|below|less\s+than|<|<=|max|within)\s*(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?)\b",
    ]
    for pat in gb_patterns:
        m = re.search(pat, lower)
        if m:
            val = float(m.group(1))
            return round(val * 1024.0, 1)

    return None


def parse_batch_size(text: str) -> int:
    """Extracts desired batch size, defaulting to 1 for interactive inference."""
    lower = text.lower()
    m = re.search(r"\b(?:batch\s*size|bs|batch)\s*(?:=|:|\s)?\s*(\d+)\b", lower)
    if m:
        return int(m.group(1))
    return 1


def parse_constraints(text: str) -> Dict[str, Any]:
    """Parses all operational constraints from user text.

    Args:
        text: Natural language user query.

    Returns:
        Dictionary containing hardware, max_latency_ms, max_memory_mb, and batch_size.
    """
    return {
        "raw_text": text,
        "hardware": parse_hardware(text),
        "max_latency_ms": parse_latency_limit(text),
        "max_memory_mb": parse_memory_limit(text),
        "batch_size": parse_batch_size(text)
    }


# Suite of 35 test phrases covering diverse syntactic formulations
CONSTRAINT_TEST_PHRASES = [
    ("bank chatbot, CPU only, under 20 ms", {"hardware": "CPU", "max_latency_ms": 20.0, "max_memory_mb": None}),
    ("deploy on mobile edge with < 150 MB memory", {"hardware": "Edge", "max_latency_ms": None, "max_memory_mb": 150.0}),
    ("server CPU inference with latency below 50ms and max 200mb", {"hardware": "CPU", "max_latency_ms": 50.0, "max_memory_mb": 200.0}),
    ("GPU server under 15ms latency", {"hardware": "GPU", "max_latency_ms": 15.0, "max_memory_mb": None}),
    ("embedded phone app, sub-25ms, memory < 120 mb", {"hardware": "Edge", "max_latency_ms": 25.0, "max_memory_mb": 120.0}),
    ("CPU chatbot under 30 milliseconds", {"hardware": "CPU", "max_latency_ms": 30.0, "max_memory_mb": None}),
    ("raspberry pi device under 100 MB", {"hardware": "Edge", "max_latency_ms": None, "max_memory_mb": 100.0}),
    ("classify banking queries, max 40ms on x86 cpu", {"hardware": "CPU", "max_latency_ms": 40.0, "max_memory_mb": None}),
    ("cloud GPU inference under 10ms", {"hardware": "GPU", "max_latency_ms": 10.0, "max_memory_mb": None}),
    ("memory under 180MB on edge hardware", {"hardware": "Edge", "max_latency_ms": None, "max_memory_mb": 180.0}),
    ("ultra low latency sub-15ms for banking teller app", {"hardware": None, "max_latency_ms": 15.0, "max_memory_mb": None}),
    ("mobile banking widget with < 200 megabytes footprint", {"hardware": "Edge", "max_latency_ms": None, "max_memory_mb": 200.0}),
    ("standard server cpu, latency <= 35ms", {"hardware": "CPU", "max_latency_ms": 35.0, "max_memory_mb": None}),
    ("on cuda gpu under 0.02 seconds", {"hardware": "GPU", "max_latency_ms": 20.0, "max_memory_mb": None}),
    ("iot device with max 130mb ram and under 50ms", {"hardware": "Edge", "max_latency_ms": 50.0, "max_memory_mb": 130.0}),
    ("fastest cpu model under 25ms", {"hardware": "CPU", "max_latency_ms": 25.0, "max_memory_mb": None}),
    ("embedded system with less than 150 mb", {"hardware": "Edge", "max_latency_ms": None, "max_memory_mb": 150.0}),
    ("nvidia v100 under 5ms latency", {"hardware": "GPU", "max_latency_ms": 5.0, "max_memory_mb": None}),
    ("classify intents with batch size 8 under 45ms", {"hardware": None, "max_latency_ms": 45.0, "max_memory_mb": None}),
    ("lightweight model under 125 MB on cpu", {"hardware": "CPU", "max_latency_ms": None, "max_memory_mb": 125.0}),
    ("phone client under 20ms and memory < 140mb", {"hardware": "Edge", "max_latency_ms": 20.0, "max_memory_mb": 140.0}),
    ("intel cpu server with latency under 80 ms", {"hardware": "CPU", "max_latency_ms": 80.0, "max_memory_mb": None}),
    ("edge hardware under 0.03 sec and < 200mb", {"hardware": "Edge", "max_latency_ms": 30.0, "max_memory_mb": 200.0}),
    ("chatbot running on cpu with max 300 mb memory", {"hardware": "CPU", "max_latency_ms": None, "max_memory_mb": 300.0}),
    ("real-time customer service under 18ms", {"hardware": None, "max_latency_ms": 18.0, "max_memory_mb": None}),
    ("offline mobile model with max 115mb size", {"hardware": "Edge", "max_latency_ms": None, "max_memory_mb": 115.0}),
    ("cloud cpu deployment under 60 milliseconds", {"hardware": "CPU", "max_latency_ms": 60.0, "max_memory_mb": None}),
    ("amd server cpu with latency < 22ms", {"hardware": "CPU", "max_latency_ms": 22.0, "max_memory_mb": None}),
    ("on-device assistant with under 160 MB ram", {"hardware": "Edge", "max_latency_ms": None, "max_memory_mb": 160.0}),
    ("high throughput cpu service with latency < 30ms", {"hardware": "CPU", "max_latency_ms": 30.0, "max_memory_mb": None}),
    ("edge phone under 15ms latency limit", {"hardware": "Edge", "max_latency_ms": 15.0, "max_memory_mb": None}),
    ("gpu cluster with latency under 8ms", {"hardware": "GPU", "max_latency_ms": 8.0, "max_memory_mb": None}),
    ("microservice on cpu, max memory 250 mb", {"hardware": "CPU", "max_latency_ms": None, "max_memory_mb": 250.0}),
    ("classify banking queries, CPU only, under 20 ms", {"hardware": "CPU", "max_latency_ms": 20.0, "max_memory_mb": None}),
    ("sub-10ms latency on gpu with bs 16", {"hardware": "GPU", "max_latency_ms": 10.0, "max_memory_mb": None}),
]


def test_phrase_suite() -> int:
    """Executes verification test on all 35 phrases and reports pass rate."""
    passed = 0
    for phrase, expected in CONSTRAINT_TEST_PHRASES:
        res = parse_constraints(phrase)
        hw_match = res["hardware"] == expected["hardware"]
        lat_match = res["max_latency_ms"] == expected["max_latency_ms"]
        mem_match = res["max_memory_mb"] == expected["max_memory_mb"]

        if hw_match and lat_match and mem_match:
            passed += 1
        else:
            print(f"[FAIL] '{phrase}' -> Got: {res}, Expected: {expected}")

    print(f"[CONSTRAINT PARSER] Verified {passed}/{len(CONSTRAINT_TEST_PHRASES)} test phrases ({passed/len(CONSTRAINT_TEST_PHRASES)*100:.1f}% pass rate).")
    return passed


if __name__ == "__main__":
    test_phrase_suite()
