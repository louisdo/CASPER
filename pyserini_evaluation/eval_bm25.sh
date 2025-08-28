INDEX_FOLDER=/scratch/lamdo/beir_splade/indexes/

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
    # cfscube_taxoindex
    irb
)

for wtoken in "${weight_tokens[@]}"; do 
    for wphrase in "${weight_phrases[@]}"; do 
        for dataset in "${datasets[@]}"; do
            for model in "${models[@]}"; do

                for chunk_idx in $(seq 0 $((${NUM_CHUNKS} - 1))); do
                    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
                    python eval_bm25.py \
                    --index_folder $INDEX_FOLDER \
                    --dataset $dataset > results/bm25-$dataset.txt
                done
            done
        done
    done
done