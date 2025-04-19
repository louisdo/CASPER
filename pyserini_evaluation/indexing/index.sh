datasets=(scifact scidocs nfcorpus trec-covid doris_mae cfscube)
models=(
    # "phrase_splade_27"
    "phrase_splade_24"
    # "eru_kg"
    # "splade_maxsim_150k_lowregv4"
    # normal_splade_pretrains2orc
    # original_spladev2_max
)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
    echo "Processing: $dataset - $model"

    CUDA_VISIBLE_DEVICES=1 \
    python index.py \
    --dataset $dataset \
    --model_name $model \
    --outfolder /scratch/lamdo/beir_splade/ \
    --chunking_size 200000 \
    --remove_collections_folder 1 \
    --store_documents_in_raw 1
    done
done