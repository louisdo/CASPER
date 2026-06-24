







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