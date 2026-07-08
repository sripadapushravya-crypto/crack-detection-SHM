"""Train the fine-tuned ResNet-18 classifier described in thesis §3.4.

Mirrors sdnet_pipeline.train's CLI shape and output conventions (class-weighted
loss, balanced_accuracy + min_recall threshold tuning, JSON summary) so this
is a genuine second model path alongside the existing classical baseline —
not a replacement for it. Run both, and Table 4.3's baseline-vs-ResNet-18
comparison becomes real.

Usage:
    uv run sdnet-deep-train --sample-size 0 --epochs 12 --batch-size 32 \
        --threshold-metric balanced_accuracy --min-recall 0.70
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from sdnet_pipeline.config import (
    DATA_DIR,
    DEFAULT_MANIFEST,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_data_dirs,
)
from sdnet_pipeline.deep_dataset import ManifestImageDataset
from sdnet_pipeline.deep_model import build_resnet18
from sdnet_pipeline.utils import utc_now_iso, write_json

DEFAULT_RESNET_MODEL = MODELS_DIR / "resnet18_classifier.pt"
DEFAULT_RESNET_METRICS = RESULTS_DIR / "resnet18_metrics.json"

THRESHOLD_METRICS = {"accuracy", "balanced_accuracy", "f1"}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(targets: pd.Series) -> tuple[float, float]:
    """Inverse-frequency class weights, matching thesis Eq. in §3.4.2:
    w_c = N / (2 * N_c)."""
    n = len(targets)
    n_pos = int(targets.sum())
    n_neg = n - n_pos
    w_pos = n / (2.0 * max(n_pos, 1))
    w_neg = n / (2.0 * max(n_neg, 1))
    return w_neg, w_pos  # (w_0, w_1)


def threshold_metric_value(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    raise ValueError(f"Unsupported threshold metric: {metric!r}")


def tune_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str,
    min_recall: float,
) -> dict[str, float]:
    """
    Sweeps tau in [0.20, 0.85] at step 0.01 (matching thesis §3.5), keeping only
    thresholds whose recall clears min_recall, then selects the one that
    maximises `metric` among the feasible set. Falls back to the
    highest-recall threshold if nothing clears the floor (never silently
    returns an unsafe low-recall operating point).
    """
    candidates = np.arange(0.20, 0.85 + 1e-9, 0.01)
    rows = []
    for tau in candidates:
        y_pred = (scores >= tau).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        rows.append(
            {
                "threshold": float(tau),
                "recall": float(recall),
                "metric_value": float(threshold_metric_value(metric, y_true, y_pred)),
            }
        )

    feasible = [row for row in rows if row["recall"] >= min_recall]
    if feasible:
        best = max(feasible, key=lambda row: row["metric_value"])
    else:
        best = max(rows, key=lambda row: row["recall"])

    return {
        "threshold": best["threshold"],
        "metric": metric,
        "metric_value": best["metric_value"],
        "recall_at_threshold": best["recall"],
        "min_recall": float(min_recall),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    n_batches = 0

    for images, targets, _ in tqdm(loader, desc="train" if train_mode else "eval", leave=False):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, targets)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def collect_scores(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_scores, all_targets, all_ids = [], [], []
    for images, targets, image_ids in tqdm(loader, desc="scoring", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_scores.append(probs)
        all_targets.append(targets.numpy())
        all_ids.extend(image_ids)
    return np.concatenate(all_scores), np.concatenate(all_targets), all_ids


def train_resnet18(
    manifest_path: Path,
    model_path: Path,
    metrics_path: Path,
    sample_size: int,
    image_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    threshold_metric: str,
    min_recall: float,
    seed: int,
) -> dict:
    ensure_data_dirs()
    torch.manual_seed(seed)

    df = pd.read_csv(manifest_path)
    labeled = df[df["target"].notna()].copy()
    train_df = labeled[labeled["split"] == "train"]
    val_df = labeled[labeled["split"] == "validation"]
    test_df = labeled[labeled["split"] == "test"]

    if sample_size > 0:
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=seed)

    print(f"Dataset: train={len(train_df):,}  validation={len(val_df):,}  test={len(test_df):,}")
    print(f"Threshold tuning: metric={threshold_metric}, min_recall={min_recall} "
          f"(recommended for inspection: balanced_accuracy + min_recall=0.70)")

    device = get_device()
    print(f"Device: {device}")

    train_loader = DataLoader(
        ManifestImageDataset(train_df, image_size, train=True),
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True,
    )
    val_loader = DataLoader(
        ManifestImageDataset(val_df, image_size, train=False),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )
    test_loader = DataLoader(
        ManifestImageDataset(test_df, image_size, train=False),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )

    model = build_resnet18(pretrained=True).to(device)

    w0, w1 = compute_class_weights(train_df["target"])
    # BCEWithLogitsLoss's pos_weight scales the positive term relative to the
    # negative term, so pos_weight = w1 / w0 reproduces the thesis's w1/w0
    # class-weighted BCE formulation.
    pos_weight = torch.tensor([w1 / w0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, device, criterion, optimizer=None)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    val_scores, val_targets, _ = collect_scores(model, val_loader, device)
    tuning = tune_threshold(val_targets, val_scores, threshold_metric, min_recall)
    threshold = tuning["threshold"]

    test_scores, test_targets, test_ids = collect_scores(model, test_loader, device)
    test_pred = (test_scores >= threshold).astype(int)

    test_metrics = {
        "accuracy": float(accuracy_score(test_targets, test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test_targets, test_pred)),
        "precision": float(precision_score(test_targets, test_pred, zero_division=0)),
        "recall": float(recall_score(test_targets, test_pred, zero_division=0)),
        "f1": float(f1_score(test_targets, test_pred, zero_division=0)),
    }
    tn = int(((test_targets == 0) & (test_pred == 0)).sum())
    fp = int(((test_targets == 0) & (test_pred == 1)).sum())
    fn = int(((test_targets == 1) & (test_pred == 0)).sum())
    tp = int(((test_targets == 1) & (test_pred == 1)).sum())

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "resnet18",
            "decision_threshold": threshold,
            "feature_config": {"image_size": image_size, "version": "resnet18_imagenet_v1"},
            "class_weights": {"w0": w0, "w1": w1},
        },
        model_path,
    )

    summary = {
        "created_at": utc_now_iso(),
        "model_type": "resnet18",
        "model_path": str(model_path.resolve()),
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "history": history,
        "threshold_tuning": tuning,
        "test_confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "test_metrics": test_metrics,
    }
    write_json(metrics_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fine-tuned ResNet-18 crack classifier.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_RESNET_MODEL)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_RESNET_METRICS)
    parser.add_argument("--sample-size", type=int, default=0, help="0 means use all training images.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--threshold-metric", choices=sorted(THRESHOLD_METRICS), default="balanced_accuracy")
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_resnet18(
        manifest_path=args.manifest,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        sample_size=args.sample_size,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        threshold_metric=args.threshold_metric,
        min_recall=args.min_recall,
        seed=args.seed,
    )
    m = summary["test_metrics"]
    cm = summary["test_confusion_matrix"]
    print(f"Threshold  : {summary['threshold_tuning']['threshold']:.3f} "
          f"(metric={args.threshold_metric}, min_recall={args.min_recall})")
    print(f"Test       : accuracy={m['accuracy']:.3f} precision={m['precision']:.3f} "
          f"recall={m['recall']:.3f} f1={m['f1']:.3f}")
    print(f"Confusion  : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"Wrote metrics to {summary['model_path']}")


if __name__ == "__main__":
    main()
