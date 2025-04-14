datasets=(scifact scidocs nfcorpus)
phrase_splade_model="phrase_splade"
normal_splade_model="eru_kg"

for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0 \
    python eval_ensemble.py \
    --phrase_splade_model_name $phrase_splade_model \
    --normal_splade_model_name $normal_splade_model \
    --index_folder /scratch/lamdo/beir_splade/indexes/ \
    --dataset $dataset > results/$phrase_splade_model+$normal_splade_model-$dataset.txt
done