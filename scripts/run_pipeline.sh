#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DATASET_DIR="data/raw/sdnet2018"
SAMPLE_SIZE=3000
INFERENCE_LIMIT=0
LOCALIZATION_LIMIT=0
MODEL_TYPE="extra_trees"
THRESHOLD_METRIC="accuracy"
MIN_RECALL="0.0"
N_ESTIMATORS="350"
MAX_DEPTH="0"
IMAGE_SIZE="224"
SKIP_LOCALIZATION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      echo "Usage: $0 [--dataset-dir PATH] [--sample-size N] [--inference-limit N] [--localization-limit N] [--skip-localization] [--model-type sgd|extra_trees|random_forest] [--threshold-metric accuracy|balanced_accuracy|f1|precision|recall] [--min-recall N] [--n-estimators N] [--max-depth N] [--image-size N]"
      exit 0
      ;;
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --inference-limit)
      INFERENCE_LIMIT="$2"
      shift 2
      ;;
    --localization-limit)
      LOCALIZATION_LIMIT="$2"
      shift 2
      ;;
    --skip-localization)
      SKIP_LOCALIZATION=1
      shift
      ;;
    --model-type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --threshold-metric)
      THRESHOLD_METRIC="$2"
      shift 2
      ;;
    --min-recall)
      MIN_RECALL="$2"
      shift 2
      ;;
    --n-estimators)
      N_ESTIMATORS="$2"
      shift 2
      ;;
    --max-depth)
      MAX_DEPTH="$2"
      shift 2
      ;;
    --image-size)
      IMAGE_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--dataset-dir PATH] [--sample-size N] [--inference-limit N] [--localization-limit N] [--skip-localization] [--model-type sgd|extra_trees|random_forest] [--threshold-metric accuracy|balanced_accuracy|f1|precision|recall] [--min-recall N] [--n-estimators N] [--max-depth N] [--image-size N]"
      exit 1
      ;;
  esac
done

uv run sdnet-build-manifest --dataset-dir "$DATASET_DIR"
uv run sdnet-train \
  --sample-size "$SAMPLE_SIZE" \
  --model-type "$MODEL_TYPE" \
  --threshold-metric "$THRESHOLD_METRIC" \
  --min-recall "$MIN_RECALL" \
  --n-estimators "$N_ESTIMATORS" \
  --max-depth "$MAX_DEPTH" \
  --image-size "$IMAGE_SIZE"
uv run sdnet-infer --limit "$INFERENCE_LIMIT"
if [[ "$SKIP_LOCALIZATION" -eq 0 ]]; then
  uv run sdnet-localize --limit "$LOCALIZATION_LIMIT"
fi
uv run sdnet-methodology-summary

echo "Pipeline complete. Results are in data/results/."
