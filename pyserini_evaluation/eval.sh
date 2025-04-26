# datasets=(
#     # scifact 
#     scidocs 
#     # nfcorpus 
#     # trec-covid 
#     # doris_mae 
#     # cfscube
#     # acm_cr
# )

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
    # trec_dl_2019
    # trec_dl_2020
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

        CUDA_VISIBLE_DEVICES=0 \
        python eval.py \
        --splade_model_name $model \
        --index_folder /scratch/lamdo/beir_splade/indexes/ \
        --dataset $dataset > results/$model-$dataset.txt \
        --save_metadata_for_debugging 1 \
        --add_onehot 0
    done
done