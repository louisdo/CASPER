# index and eval


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
        DATASET_MODEL_FOLDER_NAME="${dataset}__${model}"
        INDEX_FOLDER="$OUT_FOLDER/indexes/$DATASET_MODEL_FOLDER_NAME"

        # index

        rm -r $INDEX_FOLDER
        mkdir $INDEX_FOLDER

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
        python -m evaluation.index_dense \
        --dataset $dataset \
        --model_name $model \
        --index_path $INDEX_FOLDER \
        --work_dir $WORK_DIR \
        --batch_size $BATCH_SIZE

        # evaluation
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
        python -m evaluation.eval_dense \
        --model_name $model \
        --index_folder "$OUT_FOLDER/indexes" \
        --work_dir $WORK_DIR \
        --dataset $dataset \
        --batch_size $BATCH_SIZE

    done
done