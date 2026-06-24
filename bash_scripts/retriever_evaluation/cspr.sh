#!/usr/bin/env bash
set -euo pipefail

# Run from the cspr env (where pyserini + the trained model deps live):
#   UV_PROJECT_ENVIRONMENT=.venv-cspr uv run --group cspr bash bash_scripts/retriever_evaluation/cspr.sh
# pyserini needs a JVM; the system OpenJDK is used by default.
export JAVA_HOME="/home/lamdo/miniconda3/envs/venv310/lib/jvm"

OUT_FOLDER="/scratch/lamdo/CSpR/retrieval"
WORK_DIR="/scratch/lamdo/CSpR"

CUDA_DEVICE=0
BATCH_SIZE=32
DOC_MAX_LENGTH=256        # document encoding length (indexing)
QUERY_MAX_LENGTH=256       # query encoding length (eval) -- matches train q_max_length
QUANTIZATION_FACTOR=100   # MUST match between index and eval so weights share a scale
THREADS=8

# CSpR is a trained checkpoint directory (produced by train.py), not a short id.
MODEL_DIR="/scratch/lamdo/CSpR/finetuning/distilbert+15k+inplace+cspr/2026-06-23_12PM"
CONFIG_PATH="/home/lamdo/CSpR/train/conf/cspr.yaml"

# MODEL_DIR="lamdo/casper"
# CONFIG_PATH="/home/lamdo/CSpR/evaluation/cspr/conf/casper.yaml"

# short tag used only to name the index folder (index.py + eval.py must agree)
MODEL_TAG="cspr"

datasets=(
    # scifact
    # scidocs
    nfcorpus
    # litsearch
    # acm_cr
    # doris_mae
    # relish
    # cfscube
)

for dataset in "${datasets[@]}"; do
    INDEX_FOLDER="$OUT_FOLDER/indexes/${dataset}__${MODEL_TAG}"

    # rm -rf "$INDEX_FOLDER"
    # mkdir -p "$INDEX_FOLDER"

    # # 1) Encode the corpus -> Pyserini JsonVectorCollection at
    # #    $INDEX_FOLDER/collection/chunk.jsonl
    # CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
    # python -m evaluation.cspr.index \
    #     --dataset "$dataset" \
    #     --model_name "$MODEL_DIR" \
    #     --config_path "$CONFIG_PATH" \
    #     --index_path "$INDEX_FOLDER" \
    #     --work_dir "$WORK_DIR" \
    #     --batch_size "$BATCH_SIZE" \
    #     --max_length "$DOC_MAX_LENGTH" \
    #     --quantization_factor "$QUANTIZATION_FACTOR" \
    #     --top_k_token 400 --top_k_phrase 400

    # # 2) Build the Lucene impact index from the collection.
    # #    --input is the collection DIR, --index is a SEPARATE output dir.
    # python -m pyserini.index.lucene \
    #     --collection JsonVectorCollection \
    #     --input  "$INDEX_FOLDER/collection" \
    #     --index  "$INDEX_FOLDER/lucene" \
    #     --generator DefaultLuceneDocumentGenerator \
    #     --threads "$THREADS" \
    #     --impact --pretokenized

    # 3) Encode queries with the same model, search the impact index, score.
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
    python -m evaluation.cspr.eval \
        --dataset "$dataset" \
        --model_name "$MODEL_DIR" \
        --config_path "$CONFIG_PATH" \
        --model_tag "$MODEL_TAG" \
        --index_folder "$OUT_FOLDER/indexes" \
        --work_dir "$WORK_DIR" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$QUERY_MAX_LENGTH" \
        --quantization_factor "$QUANTIZATION_FACTOR" \
        --threads "$THREADS" \
        --top_k_token 400 --top_k_phrase 400
done
