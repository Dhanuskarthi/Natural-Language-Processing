"""Data utility functions for loading, inspecting, and splitting Banking77.

Provides deterministic train/val/test splits, statistical profiling
(class distribution, sequence length percentiles), and a --fast subset option.
"""

from typing import Dict, List, Tuple, Any
import os
import socket
import yaml
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict


def check_and_set_offline() -> bool:
    """Checks if network is available; if not, enables offline mode to bypass retries."""
    try:
        # Quick 0.5s socket check
        socket.setdefaulttimeout(0.5)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("huggingface.co", 443))
        return True
    except Exception:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return False


# Initialize offline mode if network is blocked
check_and_set_offline()


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration dictionary from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _stratified_subset(dataset: Dataset, target_size: int, seed: int = 42) -> Dataset:
    """Extracts a stratified subset while preserving all columns."""
    if len(dataset) <= target_size:
        return dataset
    # Hugging Face train_test_split with stratify_by_column
    split = dataset.train_test_split(
        train_size=target_size,
        seed=seed,
        stratify_by_column="label"
    )
    return split["train"]


def load_banking77(
    fast: bool = False,
    seed: int = 42,
    config: Dict[str, Any] = None
) -> Tuple[DatasetDict, List[str]]:
    """Loads Banking77 dataset with deterministic train/val/test splits.

    Args:
        fast: If True, returns a small stratified subset for fast CPU testing.
        seed: Random seed for reproducibility.
        config: Optional configuration dictionary.

    Returns:
        DatasetDict with keys ('train', 'val', 'test') and list of label names.
    """
    check_and_set_offline()

    if config is None:
        config = load_config()

    dataset_name = config.get("dataset", {}).get("name", "PolyAI/banking77")
    if "/" not in dataset_name:
        dataset_name = "PolyAI/banking77"

    try:
        raw_dataset = load_dataset(dataset_name)
    except Exception:
        raw_dataset = load_dataset("banking77")

    # Extract label names
    label_feature = raw_dataset["train"].features["label"]
    label_names = label_feature.names if hasattr(label_feature, "names") else [str(i) for i in range(77)]

    # The canonical dataset has 'train' (10,003) and 'test' (3,080)
    train_full = raw_dataset["train"]
    test_set = raw_dataset["test"]

    # Deterministic validation split from train_full (10% of train = ~1,000 samples)
    split_train_val = train_full.train_test_split(
        test_size=0.10,
        seed=seed,
        stratify_by_column="label"
    )
    train_set = split_train_val["train"]
    val_set = split_train_val["test"]

    if fast:
        fast_size = config.get("dataset", {}).get("fast_subset_size", 500)
        fast_val_size = max(77, int(fast_size * 0.2))
        fast_test_size = max(77, int(fast_size * 0.2))

        train_set = _stratified_subset(train_set, fast_size, seed=seed)
        val_set = _stratified_subset(val_set, fast_val_size, seed=seed)
        test_set = _stratified_subset(test_set, fast_test_size, seed=seed)

    dataset_dict = DatasetDict({
        "train": train_set,
        "val": val_set,
        "test": test_set
    })

    return dataset_dict, label_names


def compute_dataset_statistics(dataset_dict: DatasetDict) -> Dict[str, Any]:
    """Computes class distribution and sequence length percentiles.

    Args:
        dataset_dict: DatasetDict containing 'train', 'val', 'test'.

    Returns:
        Dictionary of summary statistics.
    """
    stats = {}
    for split in ["train", "val", "test"]:
        split_data = dataset_dict[split]
        texts = split_data["text"]
        labels = split_data["label"]

        word_counts = [len(text.split()) for text in texts]
        char_counts = [len(text) for text in texts]

        label_series = pd.Series(labels)
        class_counts = label_series.value_counts()

        stats[split] = {
            "num_samples": len(texts),
            "num_classes": len(class_counts),
            "class_count_min": int(class_counts.min()),
            "class_count_max": int(class_counts.max()),
            "class_count_mean": float(class_counts.mean()),
            "word_p50": float(np.percentile(word_counts, 50)),
            "word_p95": float(np.percentile(word_counts, 95)),
            "word_max": int(np.max(word_counts)),
            "char_p50": float(np.percentile(char_counts, 50)),
            "char_p95": float(np.percentile(char_counts, 95)),
        }

    return stats


def print_dataset_statistics(stats: Dict[str, Any]) -> None:
    """Prints formatted dataset statistics table to stdout."""
    print("=" * 70)
    print("Banking77 Dataset Statistics Summary")
    print("=" * 70)
    for split, s in stats.items():
        print(f"Split: [{split.upper()}] (Samples: {s['num_samples']}, Classes: {s['num_classes']})")
        print(f"  Class Distribution : Min={s['class_count_min']}, Max={s['class_count_max']}, Mean={s['class_count_mean']:.1f}")
        print(f"  Word Length        : p50={s['word_p50']:.1f}, p95={s['word_p95']:.1f}, Max={s['word_max']}")
        print(f"  Character Length   : p50={s['char_p50']:.1f}, p95={s['char_p95']:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    print("Loading Banking77 and calculating split statistics...")
    data_dict, labels = load_banking77(fast=False)
    summary_stats = compute_dataset_statistics(data_dict)
    print_dataset_statistics(summary_stats)
