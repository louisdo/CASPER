#!/usr/bin/env bash
set -euo pipefail

# Activate the dense virtual environment (created with: make venv-dense)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/.venv-dense/bin/activate"

OUT_FOLDER="/scratch/lamdo/CSpR/retrieval"
WORK_DIR="/scratch/lamdo/CSpR"

CUDA_DEVICE=2
BATCH_SIZE=4

datasets=(
    # scifact
    # scidocs
    # nfcorpus
    litsearch
    # acm_cr
    # doris_mae
    # relish
    # cfscube
)
models=(
    # "bge_m3"
    "qwen3_embedding"
)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        INDEX_FOLDER="$OUT_FOLDER/indexes/${dataset}__${model}"

        rm -rf "$INDEX_FOLDER"
        mkdir -p "$INDEX_FOLDER"

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
        python -m evaluation.dense.index \
            --dataset "$dataset" \
            --model_name "$model" \
            --index_path "$INDEX_FOLDER" \
            --work_dir "$WORK_DIR" \
            --batch_size "$BATCH_SIZE"

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
        python -m evaluation.dense.eval \
            --model_name "$model" \
            --index_folder "$OUT_FOLDER/indexes" \
            --work_dir "$WORK_DIR" \
            --dataset "$dataset" \
            --batch_size "$BATCH_SIZE"
    done
done
