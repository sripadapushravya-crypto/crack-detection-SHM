from __future__ import annotations

import gc
import json
import math
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image

from sdnet_pipeline.config import (
    DEFAULT_LOCALIZATIONS,
    DEFAULT_MANIFEST,
    DEFAULT_METHODOLOGY,
    DEFAULT_METRICS,
    DEFAULT_MODEL,
    DEFAULT_PREDICTIONS,
    DEFAULT_SUMMARY,
    MODELS_DIR,
    PROJECTS_DIR,
)
from sdnet_pipeline.features import extract_features
from sdnet_pipeline.localization import analyze_image
from sdnet_pipeline.methodology import build_methodology_payload, write_methodology_summary
from sdnet_pipeline.utils import read_json, utc_now_iso, write_json

app = FastAPI(
    title="SDNET Crack Detection API",
    description="Local API for SDNET2018 crack detection pipeline outputs.",
    version="0.2.0",
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(key): clean_value(value) for key, value in row.items()} for row in df.to_dict("records")]


def to_native(value: Any) -> Any:
    """Recursively convert numpy / pandas scalar types to native Python types so
    the payload is JSON-serialisable (numpy float32/float64/int64 otherwise raise
    'Object of type float32 is not JSON serializable')."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if isinstance(value, np.generic):
        native = value.item()
        if isinstance(native, float) and math.isnan(native):
            return None
        return native
    if isinstance(value, np.ndarray):
        return [to_native(v) for v in value.tolist()]
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def safe_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-")
    return token or "image"


def resolve_project_path(path_str: str) -> Path:
    p = Path(str(path_str))

    if p.exists():
        return p

    project_root = Path(__file__).resolve().parents[2]
    text = str(path_str).replace("\\", "/")

    marker = "data/raw/"
    if marker in text:
        relative = text.split(marker, 1)[1]
        candidate = project_root / "data" / "raw" / relative
        if candidate.exists():
            return candidate

    marker = "data/results/"
    if marker in text:
        relative = text.split(marker, 1)[1]
        candidate = project_root / "data" / "results" / relative
        if candidate.exists():
            return candidate

    marker = "data/projects/"
    if marker in text:
        relative = text.split(marker, 1)[1]
        candidate = project_root / "data" / "projects" / relative
        if candidate.exists():
            return candidate

    return p


# Base URL for externally hosted image assets (e.g. a Hugging Face dataset repo
# like https://huggingface.co/datasets/<user>/<repo>/resolve/main). When set,
# dataset images/artifacts not present on disk are served by redirecting to this
# host, so the large SDNET assets never need to ship inside the container.
ASSET_BASE_URL = os.getenv("ASSET_BASE_URL", "").rstrip("/")


def resolve_asset(path_str: str) -> tuple[str, Any]:
    """Resolve a stored artifact path to a local file or a remote URL.

    Returns ("file", Path) when the asset exists locally (dev machine or files
    shipped with the deploy), or ("url", str) when it must be fetched from
    ASSET_BASE_URL. The remote path mirrors the local data/ tree: a stored path
    ending in data/raw/D/CD/x.jpg maps to <ASSET_BASE_URL>/raw/D/CD/x.jpg, and
    data/results/localization/overlays/x.jpg to <ASSET_BASE_URL>/results/....
    """
    local = resolve_project_path(path_str)
    if local.exists():
        return ("file", local)

    if ASSET_BASE_URL:
        text = str(path_str).replace("\\", "/")
        for marker in ("data/raw/", "data/results/", "data/projects/"):
            if marker in text:
                relative = marker[len("data/") :] + text.split(marker, 1)[1]
                return ("url", f"{ASSET_BASE_URL}/{relative}")

    return ("file", local)


def load_predictions() -> pd.DataFrame:
    if not DEFAULT_PREDICTIONS.exists():
        raise HTTPException(
            status_code=404,
            detail="Predictions not found. Run ./scripts/run_pipeline.sh first.",
        )

    predictions = pd.read_csv(DEFAULT_PREDICTIONS)

    if DEFAULT_LOCALIZATIONS.exists():
        localizations = pd.read_csv(DEFAULT_LOCALIZATIONS)
        duplicate_columns = {
            column
            for column in localizations.columns
            if column in predictions.columns and column != "image_id"
        }
        localizations = localizations.drop(columns=sorted(duplicate_columns), errors="ignore")
        predictions = predictions.merge(localizations, on="image_id", how="left")

    return predictions


def load_localizations() -> pd.DataFrame:
    if not DEFAULT_LOCALIZATIONS.exists():
        raise HTTPException(
            status_code=404,
            detail="Localizations not found. Run uv run sdnet-localize first.",
        )
    return pd.read_csv(DEFAULT_LOCALIZATIONS)


def load_manifest() -> pd.DataFrame:
    if not DEFAULT_MANIFEST.exists():
        raise HTTPException(
            status_code=404,
            detail="Manifest not found. Run ./scripts/run_pipeline.sh first.",
        )
    return pd.read_csv(DEFAULT_MANIFEST)


# --- ResNet-18 upload classifier (primary path) ---------------------------------
# Loaded lazily and cached so torch/model weights are only read once per process.
DEFAULT_RESNET_MODEL = MODELS_DIR / "resnet18_classifier.pt"

_RESNET_STATE: dict[str, Any] = {"model": None, "device": None, "threshold": 0.5, "image_size": 224}


def _load_resnet_model() -> dict[str, Any]:
    if _RESNET_STATE["model"] is not None:
        return _RESNET_STATE

    if not DEFAULT_RESNET_MODEL.exists():
        raise HTTPException(
            status_code=404,
            detail="ResNet-18 model not found. Train it with 'uv run sdnet-deep-train' first.",
        )

    import torch
    from sdnet_pipeline.deep_model import build_resnet18

    # One thread keeps per-thread allocator buffers small on constrained instances.
    torch.set_num_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(DEFAULT_RESNET_MODEL, map_location=device)
    model = build_resnet18(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _RESNET_STATE["model"] = model
    _RESNET_STATE["device"] = device
    _RESNET_STATE["threshold"] = float(checkpoint.get("decision_threshold", 0.5))
    _RESNET_STATE["image_size"] = int(
        checkpoint.get("feature_config", {}).get("image_size", 224)
    )
    return _RESNET_STATE


def classify_uploaded_image_resnet(image_path: Path, image_id: str) -> dict[str, Any]:
    import torch
    from sdnet_pipeline.deep_dataset import build_transforms

    state = _load_resnet_model()
    model = state["model"]
    device = state["device"]
    threshold = state["threshold"]
    image_size = state["image_size"]
    transform = build_transforms(image_size, train=False)

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.inference_mode():
        logit = model(tensor)
        crack_probability = float(torch.sigmoid(logit).item())

    del tensor, logit

    predicted_target = int(crack_probability >= threshold)
    predicted_label = "cracked" if predicted_target == 1 else "non_cracked"

    return {
        "image_id": image_id,
        "path": str(image_path.resolve()),
        "relative_path": image_path.name,
        "label": None,
        "target": None,
        "surface": "uploaded",
        "source_folder": "uploaded",
        "width": int(width),
        "height": int(height),
        "predicted_target": predicted_target,
        "predicted_label": predicted_label,
        "crack_probability": crack_probability,
        "confidence": crack_probability if predicted_target == 1 else 1.0 - crack_probability,
        "model_type": "resnet18",
    }


def load_model_bundle() -> dict[str, Any]:
    if not DEFAULT_MODEL.exists():
        raise HTTPException(status_code=404, detail="Model not found. Run the data pipeline first.")
    return joblib.load(DEFAULT_MODEL)


def classify_uploaded_image(image_path: Path, image_id: str, bundle: dict[str, Any]) -> dict[str, Any]:
    model = bundle["model"]
    image_size = int(bundle.get("feature_config", {}).get("image_size", 224))
    decision_threshold = float(bundle.get("decision_threshold", 0.5))
    labels = bundle.get("labels", {0: "non_cracked", 1: "cracked"})

    feature_row = extract_features(image_path, image_size=image_size)
    crack_probability = float(model.predict_proba([feature_row])[0][1])
    predicted_target = int(crack_probability >= decision_threshold)
    predicted_label = labels.get(predicted_target, labels.get(str(predicted_target), "unknown"))

    with Image.open(image_path) as image:
        width, height = image.size

    return {
        "image_id": image_id,
        "path": str(image_path.resolve()),
        "relative_path": image_path.name,
        "label": None,
        "target": None,
        "surface": "uploaded",
        "source_folder": "uploaded",
        "width": int(width),
        "height": int(height),
        "predicted_target": predicted_target,
        "predicted_label": predicted_label,
        "crack_probability": crack_probability,
        "confidence": crack_probability if predicted_target == 1 else 1.0 - crack_probability,
    }


def enrich_metric_measurements(record: dict[str, Any], scale_mm_per_px: float | None) -> dict[str, Any]:
    record["scale_mm_per_px"] = scale_mm_per_px

    crack_length_px = record.get("crack_length_px")
    mean_width_px = record.get("mean_width_px")
    max_width_px = record.get("max_width_px")

    if scale_mm_per_px is not None:
        record["crack_length_mm"] = (
            round(float(crack_length_px) * scale_mm_per_px, 6)
            if crack_length_px is not None
            else None
        )
        record["mean_width_mm"] = (
            round(float(mean_width_px) * scale_mm_per_px, 6)
            if mean_width_px is not None
            else None
        )
        record["max_width_mm"] = (
            round(float(max_width_px) * scale_mm_per_px, 6)
            if max_width_px is not None
            else None
        )
        record["severity_basis"] = "metric_calibrated"
    else:
        record["crack_length_mm"] = None
        record["mean_width_mm"] = None
        record["max_width_mm"] = None
        record["severity_basis"] = record.get("severity_basis") or "pixel_estimate"

    return record


def project_path(project_id: str) -> Path:
    path = PROJECTS_DIR / project_id
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown project_id: {project_id}")
    return path


def project_json_path(project_id: str) -> Path:
    return project_path(project_id) / "project.json"


def read_project(project_id: str) -> dict[str, Any]:
    payload = read_json(project_json_path(project_id))
    if not payload:
        raise HTTPException(status_code=404, detail=f"Project metadata not found: {project_id}")
    return payload


def summarize_project(records: list[dict[str, Any]]) -> dict[str, Any]:
    cracked = [record for record in records if record.get("predicted_label") == "cracked"]
    non_cracked = [record for record in records if record.get("predicted_label") == "non_cracked"]
    localized = [record for record in cracked if record.get("overlay_path")]

    severity: dict[str, int] = {}
    for record in localized:
        label = str(record.get("severity_label") or "unknown")
        severity[label] = severity.get(label, 0) + 1

    localized_with_mm = [record for record in localized if record.get("max_width_mm") is not None]

    return {
        "image_count": len(records),
        "predicted_cracked": len(cracked),
        "predicted_non_cracked": len(non_cracked),
        "localized_cracks": len(localized),
        "severity_labels": severity,
        "average_crack_probability": (
            sum(float(record.get("crack_probability") or 0.0) for record in records) / len(records)
            if records
            else 0.0
        ),
        "total_crack_area_px": int(sum(int(record.get("crack_area_px") or 0) for record in localized)),
        "total_crack_length_px": float(sum(float(record.get("crack_length_px") or 0.0) for record in localized)),
        "average_mean_width_px": (
            sum(float(record.get("mean_width_px") or 0.0) for record in localized) / len(localized)
            if localized
            else 0.0
        ),
        "average_max_width_px": (
            sum(float(record.get("max_width_px") or 0.0) for record in localized) / len(localized)
            if localized
            else 0.0
        ),
        "total_crack_length_mm": (
            round(sum(float(record.get("crack_length_mm") or 0.0) for record in localized_with_mm), 6)
            if localized_with_mm
            else None
        ),
        "average_mean_width_mm": (
            round(
                sum(float(record.get("mean_width_mm") or 0.0) for record in localized_with_mm)
                / len(localized_with_mm),
                6,
            )
            if localized_with_mm
            else None
        ),
        "average_max_width_mm": (
            round(
                sum(float(record.get("max_width_mm") or 0.0) for record in localized_with_mm)
                / len(localized_with_mm),
                6,
            )
            if localized_with_mm
            else None
        ),
        "calibrated_images": len(localized_with_mm),
        "segmentation_source": "heuristic_clahe_frangi_morphology",
        "measurement_method": "mask_skeleton_distance_transform",
    }


@app.get("/", tags=["System"])
def root() -> dict[str, Any]:
    return {
        "name": "SDNET Crack Detection API",
        "status": "running",
        "version": app.version,
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/projects")
def list_projects() -> dict[str, Any]:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    projects: list[dict[str, Any]] = []

    for path in sorted(PROJECTS_DIR.iterdir(), reverse=True):
        metadata_path = path / "project.json"
        if metadata_path.exists():
            payload = read_json(metadata_path)
            projects.append(
                {
                    "project_id": payload.get("project_id"),
                    "name": payload.get("name"),
                    "created_at": payload.get("created_at"),
                    "summary": payload.get("summary", {}),
                }
            )

    return {"projects": projects}


@app.post("/api/projects")
async def create_project(
    name: str = Form("Concrete Inspection Project"),
    scale_mm_per_px: float | None = Form(None),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    if scale_mm_per_px is not None and scale_mm_per_px <= 0:
        raise HTTPException(status_code=400, detail="scale_mm_per_px must be greater than 0.")

    # Warm the ResNet-18 model up front so a missing/broken model fails fast
    # (before any files are written) with a clear error.
    _load_resnet_model()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    project_id = f"project_{timestamp}_{uuid.uuid4().hex[:8]}"
    base_dir = PROJECTS_DIR / project_id
    uploads_dir = base_dir / "uploads"
    results_dir = base_dir / "results"
    localization_dir = results_dir / "localization"

    uploads_dir.mkdir(parents=True, exist_ok=True)
    localization_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    localization_records: list[dict[str, Any]] = []
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for index, upload in enumerate(files, start=1):
        original_name = upload.filename or f"upload_{index}.jpg"
        suffix = Path(original_name).suffix.lower()

        if suffix not in allowed_suffixes:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {original_name}")

        image_id = f"{project_id}_{index:04d}"
        destination = uploads_dir / f"{image_id}_{safe_token(Path(original_name).stem)}{suffix}"

        with destination.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)

        try:
            record = classify_uploaded_image_resnet(destination, image_id=image_id)
            record["original_filename"] = original_name
            record["scale_mm_per_px"] = scale_mm_per_px

            if record["predicted_label"] == "cracked":
                localization = analyze_image(
                    pd.Series(record),
                    output_dir=localization_dir,
                    min_object_size=64,
                    max_components=12,
                    max_polygon_points=160,
                    min_component_length=18,
                    min_elongation=1.8,
                    scale_mm_per_px=scale_mm_per_px,
                )

                localization = enrich_metric_measurements(dict(localization), scale_mm_per_px)
                record.update(localization)
                localization_records.append(localization)
            else:
                record = enrich_metric_measurements(record, scale_mm_per_px)

            records.append(record)
            gc.collect()  # return large localization arrays to the OS between images

        except Exception as exc:
            records.append(
                {
                    "image_id": image_id,
                    "path": str(destination.resolve()),
                    "relative_path": destination.name,
                    "original_filename": original_name,
                    "scale_mm_per_px": scale_mm_per_px,
                    "error": str(exc),
                }
            )

    predictions_path = results_dir / "predictions.csv"
    localizations_path = results_dir / "localizations.csv"

    pd.DataFrame(records).to_csv(predictions_path, index=False)
    pd.DataFrame(localization_records).to_csv(localizations_path, index=False)

    project = {
        "project_id": project_id,
        "name": name,
        "created_at": utc_now_iso(),
        "project_dir": str(base_dir.resolve()),
        "uploads_dir": str(uploads_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "predictions_path": str(predictions_path.resolve()),
        "localizations_path": str(localizations_path.resolve()),
        "scale_mm_per_px": scale_mm_per_px,
        "summary": summarize_project(records),
        "records": records,
    }

    project = to_native(project)
    write_json(base_dir / "project.json", project)
    return project


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    return read_project(project_id)


@app.get("/api/projects/{project_id}/images/{image_id}/{artifact}")
def project_image_artifact(project_id: str, image_id: str, artifact: str) -> FileResponse:
    if artifact not in {"original", "overlay", "heatmap", "mask"}:
        raise HTTPException(status_code=404, detail=f"Unsupported artifact: {artifact}")

    project = read_project(project_id)
    records = project.get("records", [])
    match = next((record for record in records if record.get("image_id") == image_id), None)

    if not match:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")

    path_key = "path" if artifact == "original" else f"{artifact}_path"
    artifact_path = match.get(path_key)

    if not artifact_path:
        raise HTTPException(status_code=404, detail=f"{artifact} not available for image_id: {image_id}")

    path = resolve_project_path(artifact_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{artifact} file no longer exists: {path}")

    return FileResponse(path)


@app.get("/api/status")
def status() -> dict[str, Any]:
    artifacts = {
        "manifest": DEFAULT_MANIFEST,
        "metrics": DEFAULT_METRICS,
        "predictions": DEFAULT_PREDICTIONS,
        "localizations": DEFAULT_LOCALIZATIONS,
        "methodology": DEFAULT_METHODOLOGY,
        "summary": DEFAULT_SUMMARY,
    }

    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "modified": path.stat().st_mtime if path.exists() else None,
        }
        for name, path in artifacts.items()
    }


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    methodology = read_json(DEFAULT_METHODOLOGY) or build_methodology_payload()
    output = {
        "summary": read_json(DEFAULT_SUMMARY),
        "metrics": read_json(DEFAULT_METRICS),
        "manifest": read_json(DEFAULT_MANIFEST.with_suffix(".summary.json")),
        "methodology": methodology,
        "status": status(),
    }
    return output


@app.get("/api/methodology")
def methodology() -> dict[str, Any]:
    payload = read_json(DEFAULT_METHODOLOGY)
    if payload:
        return payload
    return write_methodology_summary(DEFAULT_METHODOLOGY)


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    payload = read_json(DEFAULT_METRICS)
    if not payload:
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    return payload


@app.get("/api/predictions")
def predictions(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    surface: str | None = None,
    predicted_label: str | None = None,
    actual_label: str | None = None,
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    sort_by: str = Query("confidence", pattern="^(confidence|crack_probability|image_id)$"),
    direction: str = Query("desc", pattern="^(asc|desc)$"),
) -> dict[str, Any]:
    df = load_predictions()

    if surface:
        df = df[df["surface"] == surface]
    if predicted_label:
        df = df[df["predicted_label"] == predicted_label]
    if actual_label:
        df = df[df["label"] == actual_label]
    if min_confidence is not None:
        df = df[df["confidence"] >= min_confidence]

    total = len(df)
    ascending = direction == "asc"

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    page = df.iloc[offset : offset + limit].copy()
    return {
        "total": int(total),
        "offset": int(offset),
        "limit": int(limit),
        "records": clean_records(page),
    }


@app.get("/api/options")
def options() -> dict[str, list[str]]:
    df = load_predictions()
    return {
        "surfaces": sorted(value for value in df["surface"].dropna().unique().tolist()),
        "predicted_labels": sorted(value for value in df["predicted_label"].dropna().unique().tolist()),
        "actual_labels": sorted(value for value in df["label"].dropna().unique().tolist()),
        "severity_labels": sorted(
            value for value in df.get("severity_label", pd.Series(dtype=str)).dropna().unique().tolist()
        ),
    }


@app.get("/api/predictions/{image_id}/image")
def prediction_image(image_id: str):
    df = load_predictions()
    match = df[df["image_id"] == image_id]

    if match.empty:
        manifest = load_manifest()
        match = manifest[manifest["image_id"] == image_id]

    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")

    kind, target = resolve_asset(match.iloc[0]["path"])
    if kind == "url":
        return RedirectResponse(target)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Image file no longer exists: {target}")

    return FileResponse(target)


@app.get("/api/predictions/{image_id}/localization")
def prediction_localization(image_id: str) -> dict[str, Any]:
    df = load_localizations()
    match = df[df["image_id"] == image_id]

    if match.empty:
        raise HTTPException(status_code=404, detail=f"No localization found for image_id: {image_id}")

    record = clean_records(match.head(1))[0]
    polygons_json = record.get("polygons_json")

    if polygons_json:
        try:
            record["polygons"] = json.loads(polygons_json)
        except Exception:
            record["polygons"] = []

    return record


@app.get("/api/predictions/{image_id}/{artifact}")
def prediction_artifact(image_id: str, artifact: str):
    if artifact not in {"overlay", "heatmap", "mask"}:
        raise HTTPException(status_code=404, detail=f"Unsupported artifact: {artifact}")

    df = load_localizations()
    match = df[df["image_id"] == image_id]

    if match.empty:
        raise HTTPException(status_code=404, detail=f"No localization found for image_id: {image_id}")

    path_column = f"{artifact}_path"
    kind, target = resolve_asset(match.iloc[0].get(path_column, ""))
    if kind == "url":
        return RedirectResponse(target)

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"{artifact} file no longer exists: {target}")

    return FileResponse(target)
