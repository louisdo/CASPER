datasets=(
    "semeval" "inspec" "nus" "krapivin" "kp20k" 
    # kp20k
)

for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python keyphrase_generation_batch.py --dataset_name $dataset
done