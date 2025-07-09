INDEX_FOLDER=/scratch/lamdo/beir_splade/indexes/

CUDA_DEVICE=1

datasets=(
    # scifact   
    # scidocs 
    # nfcorpus  
    # doris_mae 
    trec-covid 
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
)
models=(
    "specter2"
    "e5_base"
)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_dense.py \
        --model_name $model \
        --index_folder $INDEX_FOLDER \
        --dataset $dataset > results/$model-$dataset.txt
    done
done