datasets=(
    scifact 
    scidocs 
    nfcorpus 
    trec-covid 
    doris_mae 
    cfscube
    acm_cr
    # arguana 
    # fiqa
    # msmarco
    # trec_dl_2019
    # trec_dl_2020
)

for dataset in "${datasets[@]}"; do
    python convert_eval_datasets_to_colbert_format.py --dataset_name $dataset --output_folder ../data/ --source_folder /home/lamdo/splade/data
done
