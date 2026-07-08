"""Run the trained ResNet-18 classifier across the manifest and write
predictions in the same schema as sdnet_pipeline.inference, so the existing
FastAPI backend and React frontend can serve ResNet-18 results without any
API or UI changes.

Also writes Grad-CAM heatmaps for cracked predictions. These are a distinct
artifact from the CLAHE+Frangi crack-likelihood heatmaps written by
localization.py — one explains the classifier's decision (Grad-CAM, over
ResNet-18's layer4), the other estimates *where* the crack pixels are
(Frangi/Sauvola, unsupervised). Keeping them in separate output folders
avoids the two ever being confused with each other, which was one of the
inconsistencies flagged in the thesis review.

Usage:
    uv run sdnet-deep-infer --limit 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from sdnet_pipeline.config import (
    DEFAULT_MANIFEST,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_data_dirs,
)
from sdnet_pipeline.deep_dataset import build_transforms
from sdnet_pipeline.deep_model import GradCAM, build_resnet18
from sdnet_pipeline.deep_train import DEFAULT_RESNET_MODEL, get_device
from sdnet_pipeline.utils import utc_now_iso, write_json

DEFAULT_RESNET_PREDICTIONS = RESULTS_DIR / "resnet18_predictions.csv"
DEFAULT_RESNET_SUMMARY = RESULTS_DIR / "resnet18_summary.json"
GRADCAM_DIR = RESULTS_DIR / "localization" / "gradcam"


def load_bundle(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    model = build_resnet18(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def save_gradcam_heatmap(image_path: str, cam: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        base = ImageOps.exif_transpose(image).convert("RGB")

    heat = (cam * 255).astype(np.uint8)
    # Deep blue -> bright red, matching the perceptually uniform five-stop
    # colour layout described in thesis §3.10.
    heat_rgb = np.zeros((*heat.shape, 3), dtype=np.uint8)
    heat_rgb[..., 0] = heat
    heat_rgb[..., 2] = 255 - heat
    heat_image = Image.fromarray(heat_rgb, mode="RGB").resize(base.size)

    blended = Image.blend(base, heat_image, alpha=0.45)
    blended.save(out_path, quality=90)


def run_inference(
    manifest_path: Path,
    model_path: Path,
    predictions_path: Path,
    summary_path: Path,
    limit: int,
    save_gradcam: bool,
    gradcam_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    ensure_data_dirs()
    device = get_device()
    model, checkpoint = load_bundle(model_path, device)

    image_size = int(checkpoint.get("feature_config", {}).get("image_size", 224))
    decision_threshold = float(checkpoint.get("decision_threshold", 0.5))
    transform = build_transforms(image_size, train=False)

    print(f"Model              : resnet18")
    print(f"Feature config     : image_size={image_size}")
    print(f"Decision threshold : {decision_threshold:.3f}")

    df = pd.read_csv(manifest_path)
    if limit > 0:
        df = df.head(limit).copy()

    gradcam = GradCAM(model) if save_gradcam else None

    probs, gradcam_paths = [], []
    for path in tqdm(df["path"].tolist(), desc="ResNet-18 inference"):
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

        if save_gradcam:
            tensor.requires_grad_(False)
            with torch.no_grad():
                logit = model(tensor)
                prob = torch.sigmoid(logit).item()
        else:
            with torch.no_grad():
                logit = model(tensor)
                prob = torch.sigmoid(logit).item()
        probs.append(prob)

        if save_gradcam and prob >= decision_threshold:
            cam = gradcam(tensor)
            image_id = Path(path).stem
            out_path = gradcam_dir / f"{image_id}.jpg"
            save_gradcam_heatmap(path, cam, out_path)
            gradcam_paths.append(str(out_path.resolve()))
        else:
            gradcam_paths.append("")

    probs = np.array(probs)
    predicted = (probs >= decision_threshold).astype(int)
    labels = {0: "non_cracked", 1: "cracked"}

    result = df.copy()
    result["predicted_target"] = predicted
    result["predicted_label"] = [labels[int(v)] for v in predicted]
    result["crack_probability"] = probs
    result["confidence"] = [p if pred == 1 else 1.0 - p for pred, p in zip(predicted, probs)]
    if save_gradcam:
        result["gradcam_path"] = gradcam_paths

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(predictions_path, index=False)

    labeled = result[result["target"].notna()].copy()
    metrics = None
    if not labeled.empty and labeled["target"].nunique() == 2:
        y_true = labeled["target"].astype(int)
        y_pred = labeled["predicted_target"].astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        metrics["confusion_matrix"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    summary = {
        "created_at": utc_now_iso(),
        "rows": int(len(result)),
        "model_path": str(model_path.resolve()),
        "predictions_path": str(predictions_path.resolve()),
        "decision_threshold": decision_threshold,
        "model_type": "resnet18",
        "feature_config": checkpoint.get("feature_config", {}),
        "predicted_labels": result["predicted_label"].value_counts().to_dict(),
        "actual_labels": result["label"].fillna("unknown").value_counts().to_dict(),
        "surfaces": result["surface"].fillna("unknown").value_counts().to_dict(),
        "average_crack_probability": float(result["crack_probability"].mean()),
        "metrics_on_labeled_data": metrics,
        "scope_note": (
            "predicted_labels/actual_labels/metrics_on_labeled_data reflect "
            "whatever subset of the manifest was processed this run (see "
            "'rows' and --limit). Cross-check against the split column before "
            "quoting these numbers as test-only or corpus-wide."
        ),
    }
    write_json(summary_path, summary)
    return result, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ResNet-18 inference across manifest images.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_RESNET_MODEL)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_RESNET_PREDICTIONS)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_RESNET_SUMMARY)
    parser.add_argument("--limit", type=int, default=0, help="0 processes every manifest image.")
    parser.add_argument("--save-gradcam", action="store_true", help="Write Grad-CAM heatmaps for cracked predictions.")
    parser.add_argument("--gradcam-dir", type=Path, default=GRADCAM_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, summary = run_inference(
        args.manifest,
        args.model_path,
        args.predictions_path,
        args.summary_path,
        limit=args.limit,
        save_gradcam=args.save_gradcam,
        gradcam_dir=args.gradcam_dir,
    )
    print(f"Wrote {len(result):,} predictions to {args.predictions_path}")
    print(f"Predicted : {summary['predicted_labels']}")
    print(f"Actual    : {summary['actual_labels']}")
    if summary.get("metrics_on_labeled_data"):
        m = summary["metrics_on_labeled_data"]
        print(
            f"Metrics   : accuracy={m['accuracy']:.3f} balanced_accuracy={m['balanced_accuracy']:.3f} "
            f"precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}"
        )
        cm = m["confusion_matrix"]
        print(f"Confusion : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")


if __name__ == "__main__":
    main()
