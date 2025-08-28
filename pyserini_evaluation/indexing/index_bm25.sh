NUM_CHUNKS=1
OUT_FOLDER=/scratch/lamdo/beir_splade/


datasets=(
    # scifact 
    # scidocs 
    # nfcorpus 
    # litsearch
    # acm_cr
    # doris_mae
    # trec-covid
    # cfscube
    # relish
    # arguana 
    # fiqa
    # msmarco

    # doris_mae_taxoindex
    # cfscube_taxoindex

    irb
)

for dataset in "${datasets[@]}"; do
    DATASET_MODEL_FOLDER_NAME="${dataset}__bm25"
    COLLECTION_FOLDER="$OUT_FOLDER/collections/$DATASET_MODEL_FOLDER_NAME"
    INDEX_FOLDER="$OUT_FOLDER/indexes/$DATASET_MODEL_FOLDER_NAME"

    rm -r $COLLECTION_FOLDER
    rm -r $INDEX_FOLDER
    
    mkdir $COLLECTION_FOLDER
    mkdir $INDEX_FOLDER

    # create representation
    for chunk_idx in $(seq 0 $((${NUM_CHUNKS} - 1))); do
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
        python index_bm25.py \
        --dataset $dataset \
        --outfolder /scratch/lamdo/beir_splade/ \
        --num_chunks $NUM_CHUNKS \
        --chunk_idx $chunk_idx \
        --remove_collections_folder 1 \
        --store_documents_in_raw $STORE_DOCUMENTS_IN_RAW
    done

    # do indexing
    python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input $COLLECTION_FOLDER \
    --index $INDEX_FOLDER \
    --generator DefaultLuceneDocumentGenerator \
    --threads 8 --storePositions --storeDocvectors --storeRaw

done