NUM_CHUNKS=1
OUT_FOLDER=/scratch/lamdo/beir_splade/

CUDA_DEVICE=2

datasets=(
    # scifact 
    # scidocs 
    # nfcorpus 
    # litsearch
    # acm_cr
    # doris_mae
    trec-covid
    # relish
    # cfscube
    # arguana 
    # fiqa
    # msmarco

    # doris_mae_taxoindex
    # cfscube_taxoindex
)
models=(
    "specter2"
    "e5_base"
)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        DATASET_MODEL_FOLDER_NAME="${dataset}__${model}"
        COLLECTION_FOLDER="$OUT_FOLDER/collections/$DATASET_MODEL_FOLDER_NAME"
        INDEX_FOLDER="$OUT_FOLDER/indexes/$DATASET_MODEL_FOLDER_NAME"

        rm -r $COLLECTION_FOLDER
        rm -r $INDEX_FOLDER
        
        mkdir $COLLECTION_FOLDER
        mkdir $INDEX_FOLDER

        # create representation
        for chunk_idx in $(seq 0 $((${NUM_CHUNKS} - 1))); do
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
            python index_dense.py \
            --dataset $dataset \
            --model_name $model \
            --index_path $INDEX_FOLDER \
            --num_chunks $NUM_CHUNKS \
            --chunk_idx $chunk_idx

        done

        # # do indexing
        # python -m pyserini.index.faiss --input $COLLECTION_FOLDER --output $INDEX_FOLDER --hnsw

        rm -r $COLLECTION_FOLDER
    done
done