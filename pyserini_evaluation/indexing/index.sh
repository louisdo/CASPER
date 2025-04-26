NUM_CHUNKS=4
OUT_FOLDER=/scratch/lamdo/beir_splade/

STORE_DOCUMENTS_IN_RAW=0


datasets=(
    scifact 
    # scidocs 
    # nfcorpus 
    # trec-covid
    # doris_mae
    # cfscube
    # acm_cr
    # arguana 
    # fiqa
    # msmarco
)
models=(
    # "phrase_splade_27"
    # "phrase_splade_33"
    phrase_splade_38
    # "eru_kg"
    # "splade_maxsim_100k_lowregv6"
    # normal_splade_pretrains2orc
    # original_spladev2_max
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
        for chunk_idx in $(seq 0 ${NUM_CHUNKS-1}); do
            CUDA_VISIBLE_DEVICES=1 \
            python index.py \
            --dataset $dataset \
            --model_name $model \
            --outfolder /scratch/lamdo/beir_splade/ \
            --num_chunks $NUM_CHUNKS \
            --chunk_idx $chunk_idx \
            --remove_collections_folder 1 \
            --store_documents_in_raw $STORE_DOCUMENTS_IN_RAW
        done

        # do indexing
        python -m pyserini.index.lucene \
        --collection JsonVectorCollection \
        --input $COLLECTION_FOLDER \
        --index $INDEX_FOLDER \
        --generator DefaultLuceneDocumentGenerator \
        --threads 8 \
        --impact --pretokenized \
        --storePositions --storeDocvectors --storeRaw

        rm -r $COLLECTION_FOLDER
    done
done