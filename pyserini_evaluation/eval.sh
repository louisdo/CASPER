datasets=(cfscube)
models=(
    # "phrase_splade"
    # "eru_kg"
    # "splade_maxsim_150k_lowregv4"
    # normal_splade_pretrains2orc
    original_spladev2_max
)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do

        CUDA_VISIBLE_DEVICES=2 \
        python eval.py \
        --splade_model_name $model \
        --index_folder /scratch/lamdo/beir_splade/indexes/ \
        --dataset $dataset > results/$model-$dataset.txt \
        --save_metadata_for_debugging 1 \
        --add_onehot 0
    done
done