NUM_CHUNKS=1
OUT_FOLDER=/scratch/lamdo/beir_splade/

STORE_DOCUMENTS_IN_RAW=0
ADD_BM25=0
MASK_SPECIAL_TOKENS=0

CUDA_DEVICE=2

datasets=(
    scifact 
    scidocs 
    nfcorpus 
    litsearch
    acm_cr
    doris_mae
    trec-covid
    cfscube
    relish
    # arguana 
    # fiqa
    # msmarco

    # doris_mae_taxoindex
    # cfscube_taxoindex
)
models=(
    original_spladev2
    # "phrase_splade_27"
    # "phrase_splade_33"
    # splade_addedword_1
    # phrase_splade_55
    # phrase_splade_93
    # phrase_splade_88
    # phrase_splade_89
    # splade_normal_150k_lowreg
    # "eru_kg"
    # "splade_maxsim_150k_lowregv6"
    # normal_splade_pretrains2orc
    # original_spladev2_max
    # splade_max_1
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
            python index.py \
            --dataset $dataset \
            --model_name $model \
            --outfolder /scratch/lamdo/beir_splade/ \
            --num_chunks $NUM_CHUNKS \
            --chunk_idx $chunk_idx \
            --remove_collections_folder 1 \
            --store_documents_in_raw $STORE_DOCUMENTS_IN_RAW \
            --add_bm25 $ADD_BM25 \
            --mask_special_tokens $MASK_SPECIAL_TOKENS
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