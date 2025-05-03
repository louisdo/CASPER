datasets=(
    # scifact 
    # scidocs 
    # nfcorpus 
    # trec-covid 
    doris_mae 
    # acm_cr
    # cfscube
    # arguana 
    # fiqa
    # msmarco
    # trec_dl_2019
    # trec_dl_2020
)

models=(
    combined_cc+cocit+kp1m+query+title
    # original_colbertv2
)

experiment_path="/scratch/lamdo/colbert/experiments"
eval_data_folder="/home/lamdo/ColBERT/data"

cuda_device=2

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        CUDA_VISIBLE_DEVICES=$cuda_device python index.py \
        --model_name $model \
        --dataset_name $dataset \
        --experiment_path=$experiment_path


        CUDA_VISIBLE_DEVICES=$cuda_device python eval.py \
        --model_name $model \
        --dataset_name $dataset \
        --experiment_path $experiment_path \
        --eval_data_folder $eval_data_folder > results/$model-$dataset.txt
    done
done