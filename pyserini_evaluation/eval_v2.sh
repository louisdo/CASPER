NUM_CHUNKS=1
INDEX_FOLDER=/scratch/lamdo/beir_splade/indexes/
BM25_MODELS_FOLDER=/scratch/lamdo/beir_splade/bm25_models/
ADD_BM25=1
MASK_SPECIAL_TOKENS=0

CUDA_DEVICE=0

datasets=(
    # scifact  
    # scidocs 
    # nfcorpus  
    # doris_mae 
    # trec-covid 
    # cfscube
    # acm_cr 
    # arguana 
    # fiqa
    # msmarco
    # trec_dl_2019
    # trec_dl_2020
    # litsearch
    # relish


    # doris_mae_taxoindex
    cfscube_taxoindex
)
models=(
    # "phrase_splade_27"
    # "phrase_splade_33"
    # phrase_splade_73
    # phrase_splade_74
    # phrase_splade_75
    # phrase_splade_76
    phrase_splade_83
    # splade_normal_150k_lowreg
    # splade_addedword_2
    # "eru_kg"
    # "splade_maxsim_150k_lowregv6"
    # normal_splade_pretrains2orc
    # original_spladev2_max
    # splade_max_1
)
weight_tokens=( 1 )
weight_phrases=( 
    # 0
    # 0.05
    # 0.1
    # 0.2
    # 0.25  
    # 0.3
    0.5 
    # 0.75
    # 1
)

for wtoken in "${weight_tokens[@]}"; do 
    for wphrase in "${weight_phrases[@]}"; do 
        for dataset in "${datasets[@]}"; do
            for model in "${models[@]}"; do

                for chunk_idx in $(seq 0 $((${NUM_CHUNKS} - 1))); do
                    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
                    python eval_v2.py \
                    --splade_model_name $model \
                    --index_folder $INDEX_FOLDER \
                    --dataset $dataset \
                    --save_metadata_for_debugging 1 \
                    --weight_tokens $wtoken --weight_phrases $wphrase \
                    --add_onehot 0 \
                    --num_chunks $NUM_CHUNKS --chunk_idx $chunk_idx \
                    --mode predict \
                    --add_bm25 $ADD_BM25 \
                    --bm25_models_folder $BM25_MODELS_FOLDER
                done

                CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
                python eval_v2.py \
                --splade_model_name $model \
                --index_folder $INDEX_FOLDER \
                --dataset $dataset \
                --save_metadata_for_debugging 1 \
                --weight_tokens $wtoken --weight_phrases $wphrase \
                --add_onehot 0 \
                --num_chunks $NUM_CHUNKS --chunk_idx $chunk_idx \
                --add_bm25 $ADD_BM25 \
                --bm25_models_folder $BM25_MODELS_FOLDER \
                --mode eval #> results/$model-$dataset-t${wtoken}p${wphrase}.txt
            done
        done
    done
done