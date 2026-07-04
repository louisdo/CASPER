







CUDA_VISIBLE_DEVICES=0,1,2 python train/cspr/continuous_pretraining/continuous_pretraining.py \
--base-model-name Alibaba-NLP/gte-en-mlm-base \
--keyphrase-vocab /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
--train-file /scratch/lamdo/CSpR/continuous_pretraining/scirepeval/collections_processed_with_negatives.txt \
--output-dir /scratch/lamdo/CSpR/continuous_pretraining/gte-en-mlm-base+15k+ul/ \
--max-keyphrase 15000 \
--max-seq-length 512 \
--per-device-train-batch-size 8 \
--keyphrase-mlm-probability 0.5 \
--mlm-probability 0.15 \
--bf16 --unlikelihood --ul-alpha 1.0 \
--train-only-keyphrase-embeddings --trust-remote-code




# S2ORC
CUDA_VISIBLE_DEVICES=0 python -m train.cspr.continuous_pretraining.continuous_pretraining \
    --base-model-name Alibaba-NLP/gte-en-mlm-base \
    --keyphrase-vocab /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
    --max-keyphrase 15000 \
    --dataset s2orc --mode inplace --truncation-side right --cache-dir /scratch/lamdo/hf_cache \
    --s2orc-sample-size 10000000 \
    --output-dir /scratch/lamdo/CSpR/continuous_pretraining/gte-en-mlm-base+15k+inplace+s2orc \
    --keyphrase-mlm-probability 0.5 \
    --mlm-probability 0.15 \
    --per-device-train-batch-size 64 \
    --learning-rate 5e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --max-steps 100000 \
    --dataloader-num-workers 4 --bf16 --trust-remote-code



CUDA_VISIBLE_DEVICES=0 python -m train.cspr.continuous_pretraining.continuous_pretraining \
    --base-model-name Alibaba-NLP/gte-en-mlm-base \
    --keyphrase-vocab /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
    --max-keyphrase 15000 \
    --mode inplace --keep-surface --truncation-side right --cache-dir /scratch/lamdo/hf_cache \
    --s2orc-sample-size 10000000 \
    --output-dir /scratch/lamdo/CSpR/continuous_pretraining/gte-en-mlm-base+15k+inplace+keepsurface+s2orc \
    --keyphrase-mlm-probability 0.5 \
    --mlm-probability 0.15 \
    --per-device-train-batch-size 64 \
    --learning-rate 5e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --max-steps 100000 \
    --dataloader-num-workers 4 --bf16 --trust-remote-code


# large
CUDA_VISIBLE_DEVICES=0 python -m train.cspr.continuous_pretraining.continuous_pretraining \
    --base-model-name Alibaba-NLP/gte-en-mlm-large \
    --keyphrase-vocab /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
    --max-keyphrase 15000 \
    --mode inplace --keep-surface --truncation-side right --cache-dir /scratch/lamdo/hf_cache \
    --s2orc-sample-size 10000000 \
    --output-dir /scratch/lamdo/CSpR/continuous_pretraining/gte-en-mlm-large+15k+inplace+keepsurface+s2orc \
    --keyphrase-mlm-probability 0.5 \
    --mlm-probability 0.15 \
    --per-device-train-batch-size 32 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --max-steps 100000 \
    --dataloader-num-workers 4 --bf16 --trust-remote-code