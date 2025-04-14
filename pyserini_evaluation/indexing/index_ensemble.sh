datasets=(scifact scidocs nfcorpus)
phrase_splade_model=phrase_splade
normal_splade_model=eru_kg

for dataset in "${datasets[@]}"; do
    echo "Processing: $dataset - $phrase_splade_model + $normal_splade_model"

    CUDA_VISIBLE_DEVICES=0 \
    python index_ensemble.py \
    --dataset $dataset \
    --phrase_splade_model_name $phrase_splade_model \
    --normal_splade_model_name $normal_splade_model \
    --outfolder /scratch/lamdo/beir_splade/ \
    --chunking_size 200000 \
    --remove_collections_folder 1 \
    --store_documents_in_raw 1
done