SPLADE_CONFIG_NAME="config_splade_maxsim" \
CUDA_VISIBLE_DEVICES=1 \
python3 -m splade.train



SPLADE_CONFIG_NAME="config_splade_normal_ensembledistil" \
CUDA_VISIBLE_DEVICES=0 \
python3 -m splade.train


SPLADE_CONFIG_NAME="config_phrase_splade" \
CUDA_VISIBLE_DEVICES=2 \
python3 -m splade.train