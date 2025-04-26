for dataset in nfcorpus
do
    CUDA_VISIBLE_DEVICES="1" \
    SPLADE_CONFIG_FULLPATH="/scratch/lamdo/splade_maxsim_ckpts/splade_normal/debug/checkpoint/config.yaml" \
    python -m splade.beir_eval \
      +beir.dataset=$dataset \
      +beir.dataset_path=data/beir \
      config.index_retrieve_batch_size=100
done