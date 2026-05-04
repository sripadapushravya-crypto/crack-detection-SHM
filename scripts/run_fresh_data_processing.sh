#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DATASET_DIR="data/raw/sdnet2018"
SAMPLE_SIZE="0"
MODEL_TYPE="extra_trees"
THRESHOLD_METRIC="accuracy"
MIN_RECALL="0.0"
N_ESTIMATORS="500"
MAX_DEPTH="0"
IMAGE_SIZE="224"
INFERENCE_LIMIT="0"
LOCALIZATION_LIMIT="0"
SKIP_LOCALIZATION="0"
DOWNLOAD_MODE="link"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE=""

usage() {
  cat <<EOF
Usage: $0 [options]

Fresh Kaggle-only data processing flow. This clears local data artifacts,
downloads the original SDNET2018 Kaggle dataset, trains the model, and runs
full inference. No synthetic/demo data is generated.

Options:
  --dataset-dir PATH                  Dataset destination path. Default: data/raw/sdnet2018
  --sample-size N                     Training sample size. Use 0 for all rows. Default: 0
  --model-type NAME                   sgd|extra_trees|random_forest. Default: extra_trees
  --threshold-metric NAME             accuracy|balanced_accuracy|f1|precision|recall. Default: accuracy
  --min-recall N                      Minimum recall during threshold tuning. Default: 0.0
  --n-estimators N                    Number of trees for ensemble models. Default: 500
  --max-depth N                       Max tree depth; 0 means unlimited. Default: 0
  --image-size N                      Feature extraction image size. Default: 224
  --inference-limit N                 Inference row limit. Use 0 for all rows. Default: 0
  --localization-limit N              Crack localization limit. Use 0 for every cracked prediction. Default: 0
  --skip-localization                 Skip polygon, heatmap, area, length, and severity processing.
  --download-mode link|copy           Link Kaggle cache or copy into data/raw. Default: link
  --log-file PATH                     Explicit log file path. Default: logs/fresh_data_processing_<timestamp>.log
  --help, -h                          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
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
    --inference-limit)
      INFERENCE_LIMIT="$2"
      shift 2
      ;;
    --localization-limit)
      LOCALIZATION_LIMIT="$2"
      shift 2
      ;;
    --skip-localization)
      SKIP_LOCALIZATION="1"
      shift
      ;;
    --download-mode)
      DOWNLOAD_MODE="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$DOWNLOAD_MODE" != "link" && "$DOWNLOAD_MODE" != "copy" ]]; then
  echo "Invalid --download-mode: $DOWNLOAD_MODE. Use link or copy."
  exit 1
fi

mkdir -p "$LOG_DIR"
if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="$LOG_DIR/fresh_data_processing_$(date +%Y%m%d_%H%M%S).log"
fi
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

exec > >(tee -a "$LOG_FILE") 2>&1

format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

run_step() {
  local title="$1"
  shift
  local start_ts
  local end_ts
  local elapsed
  start_ts="$(date +%s)"

  echo ""
  echo "============================================================"
  echo "PROCESS: $title"
  echo "STARTED: $(date)"
  echo "COMMAND: $*"
  echo "============================================================"

  "$@"

  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  echo "============================================================"
  echo "COMPLETED: $title"
  echo "TIME TAKEN: $(format_duration "$elapsed") ($elapsed seconds)"
  echo "FINISHED: $(date)"
  echo "============================================================"
}

echo "Fresh SDNET2018 Kaggle Data Processing"
echo "Repository: $ROOT_DIR"
echo "Log file: $LOG_FILE"
echo "No synthetic/demo data will be generated."
echo ""
echo "Configuration"
echo "  Dataset dir: $DATASET_DIR"
echo "  Sample size: $SAMPLE_SIZE"
echo "  Model type: $MODEL_TYPE"
echo "  Threshold metric: $THRESHOLD_METRIC"
echo "  Minimum recall: $MIN_RECALL"
echo "  Estimators: $N_ESTIMATORS"
echo "  Max depth: $MAX_DEPTH"
echo "  Image size: $IMAGE_SIZE"
echo "  Inference limit: $INFERENCE_LIMIT"
echo "  Localization limit: $LOCALIZATION_LIMIT"
echo "  Skip localization: $SKIP_LOCALIZATION"
echo "  Download mode: $DOWNLOAD_MODE"

run_step "Clear existing local data artifacts" \
  rm -rf data/raw data/interim data/processed data/models data/results

run_step "Create clean local data directories" \
  mkdir -p data/raw data/interim data/processed data/models data/results

run_step "Install and lock Python dependencies with uv" \
  uv sync

if [[ "$DOWNLOAD_MODE" == "copy" ]]; then
  run_step "Download original Kaggle SDNET2018 data and copy into data/raw" \
    uv run sdnet-download --destination "$DATASET_DIR" --copy --force
else
  run_step "Download original Kaggle SDNET2018 data and link into data/raw" \
    uv run sdnet-download --destination "$DATASET_DIR" --force
fi

run_step "Build full image manifest from Kaggle data" \
  uv run sdnet-build-manifest --dataset-dir "$DATASET_DIR"

run_step "Train improved crack detection model" \
  uv run sdnet-train \
    --sample-size "$SAMPLE_SIZE" \
    --model-type "$MODEL_TYPE" \
    --threshold-metric "$THRESHOLD_METRIC" \
    --min-recall "$MIN_RECALL" \
    --n-estimators "$N_ESTIMATORS" \
    --max-depth "$MAX_DEPTH" \
    --image-size "$IMAGE_SIZE"

run_step "Run full-dataset inference and write final results" \
  uv run sdnet-infer --limit "$INFERENCE_LIMIT"

if [[ "$SKIP_LOCALIZATION" == "0" ]]; then
  run_step "Mark predicted cracks with polygons and calculate area, length, severity, and heatmaps" \
    uv run sdnet-localize --limit "$LOCALIZATION_LIMIT"
fi

run_step "Write CrackNet methodology summary and model radar metadata" \
  uv run sdnet-methodology-summary

echo ""
echo "Fresh data processing complete."
echo "Results directory: $ROOT_DIR/data/results"
echo "Model directory: $ROOT_DIR/data/models"
echo "Localization directory: $ROOT_DIR/data/results/localization"
echo "Log file: $LOG_FILE"
