SPLADE_CONFIG_NAME="config_phrase_splade" \
CUDA_VISIBLE_DEVICES=0 \
python3 -m splade.train


SPLADE_CONFIG_NAME="config_splade_max" \
CUDA_VISIBLE_DEVICES=0 \
python3 -m splade.train


SPLADE_CONFIG_NAME="config_addedword_splade" \
CUDA_VISIBLE_DEVICES=0 \
python3 -m splade.train