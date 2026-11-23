"""Fine-tunes BERT-base (Teacher) and DistilBERT (Baseline) on Banking77.

Supports both full GPU training and a --fast CPU mode for quick verification.
Saves checkpoints to models/ and evaluation metrics to benchmark/results/.
"""

from typing import Dict, Any, Tuple
import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score

from compress.data_utils import load_config, load_banking77, check_and_set_offline


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BankingDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for tokenized Banking77 splits."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: list[int]):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self) -> int:
        return len(self.labels)


def prepare_dataloaders(
    dataset_dict: Dict[str, Any],
    tokenizer: Any,
    batch_size: int,
    max_length: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tokenizes splits and returns train, val, and test DataLoaders."""
    loaders = {}
    for split in ["train", "val", "test"]:
        texts = list(dataset_dict[split]["text"])
        labels = list(dataset_dict[split]["label"])

        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        ds = BankingDataset(encodings, labels)
        shuffle = (split == "train")
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return loaders["train"], loaders["val"], loaders["test"]


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates model performance and returns accuracy and macro-F1."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    avg_loss = float(total_loss / len(all_labels))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "loss": avg_loss,
        "total_samples": len(all_labels)
    }


def train_single_model(
    model_type: str,
    fast: bool = False,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Fine-tunes a single model (teacher or baseline) and saves outputs.

    Args:
        model_type: 'teacher' or 'baseline'.
        fast: If True, executes minimal training for verification.
        config: Optional configuration dictionary.

    Returns:
        Dictionary of final test metrics.
    """
    check_and_set_offline()

    if config is None:
        config = load_config()

    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{model_type.upper()}] Initializing training on device: {device} (Fast Mode: {fast})")

    # Resolve model config
    model_info = config.get("models", {}).get(model_type, {})
    model_name = model_info.get("name", "bert-base-uncased" if model_type == "teacher" else "distilbert-base-uncased")
    checkpoint_dir = model_info.get("checkpoint_dir", f"models/{model_type}_model")
    num_labels = config.get("dataset", {}).get("num_labels", 77)

    # Resolve training hyperparameters
    if fast:
        train_cfg = config.get("training", {}).get("fast", {})
        num_epochs = train_cfg.get("num_epochs", 1)
        batch_size = train_cfg.get("batch_size", 16)
        learning_rate = 5e-5
    else:
        train_cfg = config.get("training", {})
        num_epochs = train_cfg.get("num_epochs", 3)
        batch_size = train_cfg.get("batch_size", 32)
        learning_rate = float(train_cfg.get("learning_rate", 2e-5))

    max_length = config.get("dataset", {}).get("max_seq_length", 64)

    # Load dataset
    print(f"[{model_type.upper()}] Loading Banking77 split (fast={fast})...")
    dataset_dict, _ = load_banking77(fast=fast, seed=seed, config=config)

    # Initialize Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    ).to(device)

    train_loader, val_loader, test_loader = prepare_dataloaders(
        dataset_dict, tokenizer, batch_size, max_length=max_length
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state_dict = None
    start_time = time.time()

    print(f"[{model_type.upper()}] Starting training for {num_epochs} epoch(s)...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, device)

        print(
            f"  Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
            f"Val Macro-F1: {val_metrics['macro_f1'] * 100:.2f}%"
        )

        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best checkpoint before final test
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    train_duration = time.time() - start_time
    print(f"[{model_type.upper()}] Training completed in {train_duration:.2f}s. Evaluating on Test Split...")

    test_metrics = evaluate_model(model, test_loader, device)
    print(
        f"[{model_type.upper()}] Test Results -> "
        f"Accuracy: {test_metrics['accuracy'] * 100:.2f}% | "
        f"Macro-F1: {test_metrics['macro_f1'] * 100:.2f}%"
    )

    # Save model locally (models/ is gitignored)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"[{model_type.upper()}] Model and tokenizer saved to: {checkpoint_dir}")

    # Save metrics JSON to benchmark/results/
    results_dir = config.get("paths", {}).get("results_dir", "benchmark/results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, f"{model_type}_metrics.json")

    metrics_payload = {
        "model_type": model_type,
        "base_model": model_name,
        "checkpoint_dir": checkpoint_dir,
        "precision": "FP32",
        "fast_mode": fast,
        "device": str(device),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_time_sec": round(train_duration, 2),
        "test_samples": test_metrics["total_samples"],
        "accuracy": round(test_metrics["accuracy"], 5),
        "macro_f1": round(test_metrics["macro_f1"], 5),
        "test_loss": round(test_metrics["loss"], 5),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[{model_type.upper()}] Metrics saved to: {metrics_path}")

    # Check: Verify model reloads and yields identical accuracy
    print(f"[{model_type.upper()}] Verifying checkpoint reload parity...")
    reloaded_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    reload_metrics = evaluate_model(reloaded_model, test_loader, device)

    acc_diff = abs(reload_metrics["accuracy"] - test_metrics["accuracy"])
    f1_diff = abs(reload_metrics["macro_f1"] - test_metrics["macro_f1"])

    assert acc_diff < 1e-4, f"Reload accuracy mismatch: {reload_metrics['accuracy']} vs {test_metrics['accuracy']}"
    assert f1_diff < 1e-4, f"Reload Macro-F1 mismatch: {reload_metrics['macro_f1']} vs {test_metrics['macro_f1']}"
    print(f"[{model_type.upper()}] PARITY VERIFIED: Reloaded model accuracy ({reload_metrics['accuracy']:.4f}) matches logged test metric exactly.")

    return metrics_payload


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT teacher and DistilBERT baseline on Banking77.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="all",
        choices=["teacher", "baseline", "all"],
        help="Model architecture to train (teacher, baseline, or all)."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast mode (1 epoch on subset) for quick CPU verification."
    )
    args = parser.parse_args()

    config = load_config()

    if args.model_type in ["teacher", "all"]:
        train_single_model("teacher", fast=args.fast, config=config)

    if args.model_type in ["baseline", "all"]:
        train_single_model("baseline", fast=args.fast, config=config)


if __name__ == "__main__":
    main()
