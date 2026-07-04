# S2ORC
CUDA_VISIBLE_DEVICES=0 python -m train.cspr.continuous_pretraining.continuous_pretraining \
    --base-model-name distilbert-base-uncased \
    --keyphrase-vocab /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
    --max-keyphrase 15000 \
    --mode inplace --truncation-side right --cache-dir /scratch/lamdo/hf_cache \
    --s2orc-sample-size 10000000 \
    --output-dir /scratch/lamdo/CSpR/continuous_pretraining/distilbert+15k+inplace+s2orc \
    --keyphrase-mlm-probability 0.5 \
    --mlm-probability 0.15 \
    --per-device-train-batch-size 64 \
    --learning-rate 5e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --max-steps 100000 \
    --dataloader-num-workers 4 --bf16



CUDA_VISIBLE_DEVICES=0 python -m train.cspr.continuous_pretraining.continuous_pretraining \
    --base-model-name distilbert-base-uncased \
    --keyphrase-vocab /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
    --max-keyphrase 15000 \
    --mode inplace --keep-surface --truncation-side right --cache-dir /scratch/lamdo/hf_cache \
    --s2orc-sample-size 10000000 \
    --output-dir /scratch/lamdo/CSpR/continuous_pretraining/distilbert+15k+inplace+keepsurface+s2orc \
    --keyphrase-mlm-probability 0.5 \
    --mlm-probability 0.15 \
    --per-device-train-batch-size 64 \
    --learning-rate 5e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --max-steps 100000 \
    --dataloader-num-workers 4 --bf16
