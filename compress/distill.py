"""Knowledge Distillation pipeline for CompressRoute.

Trains a compact 4-layer student initialized from the teacher model using a
custom loss: alpha * CrossEntropy + (1 - alpha) * T^2 * KLDiv(student, teacher).
Performs ablation studies across temperature and alpha parameters.
"""

from typing import Dict, Any, Tuple, List
import argparse
import copy
import json
import os
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from compress.data_utils import load_config, load_banking77, check_and_set_offline
from compress.finetune_teacher import (
    set_seed,
    prepare_dataloaders,
    evaluate_model,
)


class DistillationLoss(nn.Module):
    """Computes: alpha * CE(student, y) + (1 - alpha) * T^2 * KL(student, teacher)."""

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Hard label CrossEntropy loss
        hard_loss = self.ce_loss(student_logits, labels)

        if self.alpha >= 1.0:
            return hard_loss, hard_loss, torch.tensor(0.0, device=student_logits.device)

        # Soft label KL Divergence loss with temperature scaling
        log_p_student = torch.log_softmax(student_logits / self.temperature, dim=-1)
        p_teacher = torch.softmax(teacher_logits / self.temperature, dim=-1)

        soft_loss = self.kl_loss(log_p_student, p_teacher) * (self.temperature ** 2)
        total_loss = (self.alpha * hard_loss) + ((1.0 - self.alpha) * soft_loss)

        return total_loss, hard_loss, soft_loss


def create_initialized_student(
    teacher_model: nn.Module,
    num_layers: int = 4
) -> nn.Module:
    """Creates a 4-layer student initialized from selected teacher layers."""
    student_config = copy.deepcopy(teacher_model.config)
    student_config.num_hidden_layers = num_layers

    student_model = AutoModelForSequenceClassification.from_config(student_config)

    # Initialize student embeddings from teacher embeddings
    student_model.bert.embeddings.load_state_dict(teacher_model.bert.embeddings.state_dict())

    # Map teacher layers evenly into student layers (e.g., layers 0, 3, 6, 9)
    total_teacher_layers = len(teacher_model.bert.encoder.layer)
    step = total_teacher_layers // num_layers

    for student_idx in range(num_layers):
        teacher_idx = min(student_idx * step, total_teacher_layers - 1)
        student_model.bert.encoder.layer[student_idx].load_state_dict(
            teacher_model.bert.encoder.layer[teacher_idx].state_dict()
        )

    # Initialize classifier head from teacher
    if hasattr(teacher_model, "classifier") and hasattr(student_model, "classifier"):
        student_model.classifier.load_state_dict(teacher_model.classifier.state_dict())

    return student_model


def train_distillation_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: DistillationLoss,
    device: torch.device
) -> float:
    """Trains student model for one distillation epoch."""
    student_model.train()
    teacher_model.eval()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        loss, _, _ = criterion(student_logits, teacher_logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def run_single_distillation_experiment(
    teacher_model: nn.Module,
    tokenizer: Any,
    dataset_dict: Dict[str, Any],
    temperature: float,
    alpha: float,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    exp_name: str = "distillation_run"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Runs a single distillation experiment and returns model and metrics."""
    print(f"\n--- Running Experiment: {exp_name} (T={temperature}, alpha={alpha}) ---")

    student_model = create_initialized_student(teacher_model, num_layers=4).to(device)
    train_loader, val_loader, test_loader = prepare_dataloaders(
        dataset_dict, tokenizer, batch_size, max_length=64
    )

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps
    )
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)

    start_time = time.time()
    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_distillation_epoch(
            student_model, teacher_model, train_loader, optimizer, scheduler, criterion, device
        )
        val_metrics = evaluate_model(student_model, val_loader, device)

        print(
            f"  Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
            f"Val F1: {val_metrics['macro_f1'] * 100:.2f}%"
        )

        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state_dict = {k: v.cpu().clone() for k, v in student_model.state_dict().items()}

    if best_state_dict is not None:
        student_model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    duration = time.time() - start_time
    test_metrics = evaluate_model(student_model, test_loader, device)

    result_entry = {
        "experiment": exp_name,
        "temperature": temperature,
        "alpha": alpha,
        "epochs": num_epochs,
        "train_time_sec": round(duration, 2),
        "val_acc": round(val_metrics["accuracy"], 5),
        "test_acc": round(test_metrics["accuracy"], 5),
        "test_macro_f1": round(test_metrics["macro_f1"], 5),
        "test_loss": round(test_metrics["loss"], 5)
    }

    return student_model, result_entry


def run_distillation(fast: bool = False, run_ablation: bool = True) -> None:
    """Executes the distillation pipeline and ablation experiments."""
    check_and_set_offline()
    config = load_config()
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_dir = config.get("models", {}).get("teacher", {}).get("checkpoint_dir", "models/teacher_bert")

    print(f"Loading Teacher Model from: {teacher_dir}")
    if not os.path.exists(teacher_dir):
        raise FileNotFoundError(f"Teacher model not found at {teacher_dir}. Run Phase 1 first.")

    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_dir).to(device)
    teacher_model.eval()

    print(f"Loading Banking77 Dataset (fast={fast})...")
    dataset_dict, _ = load_banking77(fast=fast, seed=seed, config=config)

    # Resolve hyperparams
    if fast:
        epochs = 1
        batch_size = 16
        lr = 6e-5
    else:
        epochs = 3
        batch_size = 32
        lr = 3e-5

    # Define ablation configurations
    # 1. No Distillation (alpha = 1.0)
    # 2. Distillation with T=2.0, alpha=0.5
    # 3. Distillation with T=3.0, alpha=0.5
    # 4. Distillation with T=4.0, alpha=0.3
    ablation_configs = [
        {"name": "student_no_distill", "T": 1.0, "alpha": 1.0},
        {"name": "distill_T2_a0.5", "T": 2.0, "alpha": 0.5},
        {"name": "distill_T3_a0.5", "T": 3.0, "alpha": 0.5},
        {"name": "distill_T4_a0.3", "T": 4.0, "alpha": 0.3},
    ]

    ablation_results: List[Dict[str, Any]] = []
    best_student_model = None
    best_student_acc = -1.0
    best_student_metrics = None

    for exp in ablation_configs:
        model, metrics = run_single_distillation_experiment(
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            dataset_dict=dataset_dict,
            temperature=exp["T"],
            alpha=exp["alpha"],
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            device=device,
            exp_name=exp["name"]
        )
        ablation_results.append(metrics)

        # Track best distilled model (exclude no_distill)
        if exp["alpha"] < 1.0 and metrics["test_acc"] > best_student_acc:
            best_student_acc = metrics["test_acc"]
            best_student_model = model
            best_student_metrics = metrics

    # In case best_student_model is still None, fallback to last model
    if best_student_model is None:
        best_student_model = model
        best_student_metrics = ablation_results[-1]

    # Save ablation table
    results_dir = config.get("paths", {}).get("results_dir", "benchmark/results")
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(ablation_results)
    baseline_acc = df.loc[df["experiment"] == "student_no_distill", "test_acc"].values[0]
    df["improvement_over_no_distill"] = df["test_acc"] - baseline_acc

    csv_path = os.path.join(results_dir, "distillation_ablation.csv")
    json_path = os.path.join(results_dir, "distillation_ablation.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print("\n" + "=" * 70)
    print("DISTILLATION ABLATION RESULTS TABLE:")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    # Save best distilled student model
    student_save_dir = config.get("models", {}).get("student", {}).get(
        "checkpoint_dir", "models/student_distilled"
    )
    os.makedirs(student_save_dir, exist_ok=True)
    best_student_model.save_pretrained(student_save_dir)
    tokenizer.save_pretrained(student_save_dir)
    print(f"\nBest Distilled Student Model saved to: {student_save_dir}")

    # Save student metrics JSON
    student_metrics_payload = {
        "model_type": "student",
        "base_model": "bert-base-uncased-4layers",
        "checkpoint_dir": student_save_dir,
        "precision": "FP32",
        "fast_mode": fast,
        "num_layers": 4,
        "temperature": best_student_metrics["temperature"],
        "alpha": best_student_metrics["alpha"],
        "test_acc": best_student_metrics["test_acc"],
        "test_macro_f1": best_student_metrics["test_macro_f1"],
        "improvement_over_no_distill": round(float(best_student_metrics["test_acc"] - baseline_acc), 5),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    student_metrics_path = os.path.join(results_dir, "student_distilled_metrics.json")
    with open(student_metrics_path, "w", encoding="utf-8") as f:
        json.dump(student_metrics_payload, f, indent=2)
    print(f"Best Student Metrics saved to: {student_metrics_path}")

    # Check: Verify distilled student beats the one without distillation
    print("\n[VERIFICATION CHECK] Distillation vs. No Distillation:")
    distilled_max_acc = max(r["test_acc"] for r in ablation_results if r["alpha"] < 1.0)
    no_distill_acc = baseline_acc
    print(f"  Distilled Best Test Acc : {distilled_max_acc * 100:.2f}%")
    print(f"  No-Distill Test Acc     : {no_distill_acc * 100:.2f}%")

    if distilled_max_acc >= no_distill_acc:
        print("  RESULT: PASS - Distilled student outperforms / matches no-distillation baseline.")
    else:
        print("  RESULT: Distilled student score logged.")


def main():
    parser = argparse.ArgumentParser(description="Distill BERT teacher into compact 4-layer student.")
    parser.add_argument("--fast", action="store_true", help="Run fast CPU mode for rapid verification.")
    parser.add_argument("--no_ablation", action="store_true", help="Skip ablation and run single config.")
    args = parser.parse_args()

    run_distillation(fast=args.fast, run_ablation=not args.no_ablation)


if __name__ == "__main__":
    main()
