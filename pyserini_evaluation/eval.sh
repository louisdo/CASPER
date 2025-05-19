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
    scidocs 
    nfcorpus  
    # trec-covid 
    # doris_mae 
    # cfscube
    # acm_cr
    arguana 
    fiqa
    # msmarco
    # trec_dl_2019
    # trec_dl_2020
)
models=(
    # "phrase_splade_27"
    # "phrase_splade_33"
    phrase_splade_53
    # splade_addedword_1
    # "eru_kg"
    # "splade_maxsim_150k_lowregv6"
    # normal_splade_pretrains2orc
    # original_spladev2_max
    # splade_max_1
)
weight_tokens=( 1 )
weight_phrases=( 0.1 )

for wtoken in "${weight_tokens[@]}"; do 
    for wphrase in "${weight_phrases[@]}"; do 
        for dataset in "${datasets[@]}"; do
            for model in "${models[@]}"; do

                CUDA_VISIBLE_DEVICES=2 \
                python eval.py \
                --splade_model_name $model \
                --index_folder /scratch/lamdo/beir_splade/indexes/ \
                --dataset $dataset \
                --save_metadata_for_debugging 1 \
                --weight_tokens $wtoken --weight_phrases $wphrase \
                --add_onehot 0 > results/$model-$dataset-t${wtoken}p${wphrase}.txt
            done
        done
    done
done