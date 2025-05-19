models=(
    phrase_splade_39
)
datasets=(
    "semeval" "inspec" "nus" "krapivin" #"kp20k" 
    # kp20k
)

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        HF_HUB_ETAG_TIMEOUT=200 CUDA_VISIBLE_DEVICES=1 python keyphrase_generation_batch.py --dataset_name $dataset --splade_model_name $model
    done
done