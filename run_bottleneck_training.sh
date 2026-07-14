#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="${1:-kiba}"
SEED="${2:-12}"
MODE="${3:-inductive}"

TRAIN_PATH="${ROOT_DIR}/Data/${DATASET}/${MODE}/seed${SEED}/source_train_${DATASET}${SEED}.csv"
VAL_PATH="${ROOT_DIR}/Data/${DATASET}/${MODE}/seed${SEED}/target_train_${DATASET}${SEED}.csv"
TEST_PATH="${ROOT_DIR}/Data/${DATASET}/${MODE}/seed${SEED}/target_test_${DATASET}${SEED}.csv"
TEACHER_PATH="${ROOT_DIR}/Data/${DATASET}/${MODE}/seed${SEED}/${DATASET}${SEED}_inductive_teacher_emb.parquet"
OUTPUT_DIR="${ROOT_DIR}/results/${DATASET}/${MODE}/seed${SEED}"

echo "Project root: ${ROOT_DIR}"
echo "Dataset: ${DATASET}"
echo "Seed: ${SEED}"
echo "Mode: ${MODE}"

python "${ROOT_DIR}/run_model_adaptive.py" \
  --train_path "${TRAIN_PATH}" \
  --val_path "${VAL_PATH}" \
  --test_path "${TEST_PATH}" \
  --teacher_path "${TEACHER_PATH}" \
  --seed "${SEED}" \
  --mode "${MODE}" \
  --adaptive \
  --selection_metric auroc \
  --output_dir "${OUTPUT_DIR}"
