# Improved Prediction and Methodology Execution Guide

## Purpose

Use this guide to run the improved SDNET2018 pipeline, tune prediction behavior, generate crack localization artifacts, and view the updated methodology dashboard.

The pipeline now produces:

- classification metrics
- full-dataset predictions
- crack polygon overlays
- estimated crack masks
- heatmaps
- area, length, mean width, max width, and severity
- methodology summary and Performance Radar metadata

## What Changed

The solution now includes:

- richer image features:
  - HOG texture and shape features
  - local binary pattern features
  - intensity histograms
  - edge histograms
  - dark-edge density
- stronger local model options:
  - `extra_trees`
  - `random_forest`
  - `sgd`
- validation-based decision threshold tuning:
  - `accuracy`
  - `balanced_accuracy`
  - `f1`
  - `precision`
  - `recall`
- crack localization for predicted cracked images:
  - estimated binary masks
  - contour-following polygons
  - heatmaps
  - skeleton length
  - mean and max width
  - severity scoring
- methodology artifact:
  - `data/results/methodology_summary.json`
- frontend Performance Radar:
  - ResNet-50
  - VGG-16
  - EfficientNet-B0

## Important Measurement Note

SDNET2018 has image-level labels but no pixel masks. Crack masks, polygons, and measurements are heuristic estimates unless supervised masks are added later.

When no calibration scale is supplied, measurements are reported in pixels.

## Before Running

Open a terminal:

```bash
cd \Users\shrav\OneDrive\Documents\code_codex/SDNET
```

Install dependencies:

```bash
./scripts/bootstrap.sh
```

Confirm data exists:

```bash
find data/raw/sdnet2018 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
```

For the full SDNET2018 dataset, the expected image count is approximately:

```text
56092
```

## Recommended Full Run

Use this for full Kaggle processing:

```bash
./scripts/run_pipeline.sh \
  --dataset-dir data/raw/sdnet2018 \
  --sample-size 0 \
  --model-type extra_trees \
  --threshold-metric accuracy \
  --min-recall 0.0 \
  --n-estimators 350 \
  --max-depth 0 \
  --image-size 224 \
  --localization-limit 0
```

This runs:

```text
manifest -> training -> inference -> localization -> methodology summary
```

## Option 1: Maximize Overall Accuracy

Use this when the goal is to reduce false positives and increase overall correct predictions:

```bash
./scripts/run_pipeline.sh \
  --dataset-dir data/raw/sdnet2018 \
  --sample-size 0 \
  --model-type extra_trees \
  --threshold-metric accuracy \
  --min-recall 0.0 \
  --n-estimators 350 \
  --max-depth 0 \
  --image-size 224 \
  --localization-limit 0
```

Expected behavior:

- fewer false positive cracked predictions
- higher overall accuracy
- possible reduction in cracked-class recall

## Option 2: Balanced Inspection Mode

Use this when cracked and non-cracked classes both matter:

```bash
./scripts/run_pipeline.sh \
  --dataset-dir data/raw/sdnet2018 \
  --sample-size 0 \
  --model-type extra_trees \
  --threshold-metric balanced_accuracy \
  --min-recall 0.50 \
  --n-estimators 350 \
  --max-depth 0 \
  --image-size 224 \
  --localization-limit 0
```

Expected behavior:

- better balance between recall and precision
- more crack-sensitive than pure accuracy mode
- useful for inspection triage

## Option 3: F1-Oriented Mode

Use this for a single compromise metric between precision and recall:

```bash
./scripts/run_pipeline.sh \
  --dataset-dir data/raw/sdnet2018 \
  --sample-size 0 \
  --model-type extra_trees \
  --threshold-metric f1 \
  --min-recall 0.45 \
  --n-estimators 350 \
  --max-depth 0 \
  --image-size 224 \
  --localization-limit 0
```

## Faster Trial Run

Use this before a full run:

```bash
./scripts/run_pipeline.sh \
  --dataset-dir data/raw/sdnet2018 \
  --sample-size 5000 \
  --inference-limit 2000 \
  --localization-limit 200 \
  --model-type extra_trees \
  --threshold-metric accuracy \
  --n-estimators 120 \
  --max-depth 0 \
  --image-size 224
```

## Manual Step-by-Step Run

```bash
uv run sdnet-build-manifest --dataset-dir data/raw/sdnet2018
uv run sdnet-train --sample-size 0 --model-type extra_trees --threshold-metric accuracy --n-estimators 350 --max-depth 0 --image-size 224
uv run sdnet-infer --limit 0
uv run sdnet-localize --limit 0
uv run sdnet-methodology-summary
```

Optional calibrated localization:

```bash
uv run sdnet-localize --limit 0 --scale-mm-per-px 0.05
uv run sdnet-methodology-summary
```

## Start the App

Backend:

```bash
./scripts/run_backend.sh
```

Frontend:

```bash
./scripts/run_frontend.sh
```

Open:

```text
http://localhost:5173
```

## Verify Results

Summary:

```bash
python3 -m json.tool data/results/summary.json | sed -n '1,160p'
```

Metrics:

```bash
python3 -m json.tool data/results/metrics.json | sed -n '1,220p'
```

Methodology:

```bash
python3 -m json.tool data/results/methodology_summary.json | sed -n '1,220p'
```

Prediction counts:

```bash
uv run python - <<'PY'
import pandas as pd

df = pd.read_csv("data/results/predictions.csv")
print("Rows:", len(df))
print("\nActual labels:")
print(df["label"].value_counts())
print("\nPredicted labels:")
print(df["predicted_label"].value_counts())
print("\nSurface counts:")
print(df["surface"].value_counts())
print("\nAverage crack probability:", round(df["crack_probability"].mean(), 4))
PY
```

Localization quality sample:

```bash
uv run python - <<'PY'
import pandas as pd

df = pd.read_csv("data/results/localizations.csv")
cols = [
    "image_id",
    "crack_area_pct",
    "crack_length_px",
    "mean_width_px",
    "max_width_px",
    "severity_label",
    "severity_basis",
    "segmentation_source",
]
print(df[cols].head(20).to_string(index=False))
PY
```

## Compare New Prediction Counts

The earlier baseline over-predicted cracks:

```text
Predicted cracked: 25,723
Predicted non-cracked: 30,369
Actual cracked: 8,484
Actual non-cracked: 47,608
```

After a new run:

```bash
uv run python - <<'PY'
import json

summary = json.load(open("data/results/summary.json"))
print("Predicted:", summary["predicted_labels"])
print("Actual:", summary["actual_labels"])
print("Metrics:", summary["metrics_on_labeled_data"])
print("Decision threshold:", summary["decision_threshold"])
PY
```

## What to Review in the Dashboard

1. Distribution of predicted versus actual labels.
2. Confusion matrix.
3. Crack recall and precision.
4. Prediction confidence.
5. Overlay quality.
6. Mask quality.
7. Heatmap alignment.
8. Crack area and length values.
9. Mean and max width values.
10. Severity labels.
11. Performance Radar section.

## Notes

- `--sample-size 0` means train on all labeled rows.
- `--localization-limit 0` means process every predicted cracked image.
- `--threshold-metric accuracy` usually reduces false positives.
- `--threshold-metric balanced_accuracy` is better when both classes matter.
- `--min-recall` prevents threshold tuning from becoming too conservative.
- `--n-estimators 350` is a good local default; use `500` when runtime is acceptable.
- Pixel measurements are estimates unless scale calibration is provided.
