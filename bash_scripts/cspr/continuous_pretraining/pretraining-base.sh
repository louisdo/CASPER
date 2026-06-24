# the script for preparing data and pretraining from distilbert-base-uncased

python train/cspr/continuous_pretraining/pretraining_data_processing.py \
--keyphrase-vocab /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
--s2orc-data /scratch/lamdo/CSpR/continuous_pretraining/scirepeval/collections.txt \
--output /scratch/lamdo/CSpR/continuous_pretraining/scirepeval/collections_processed.txt \
--max_keyphrase 15000


CUDA_VISIBLE_DEVICES=2 python train/cspr/continuous_pretraining/continuous_pretraining.py \
--base-model-name distilbert/distilbert-base-uncased \
--keyphrase-vocab /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
--train-file /scratch/lamdo/CSpR/continuous_pretraining/scirepeval/collections_processed.txt \
--output-dir /scratch/lamdo/CSpR/continuous_pretraining/distilbert+15k/ \
--max-keyphrase 15000 \
--max-seq-length 512 \
--per-device-train-batch-size 16 \
--keyphrase-mlm-probability 0.5 \
--mlm-probability 0.15 \
--bf16


##### unlikelihood training

python train/cspr/continuous_pretraining/pretraining_data_processing.py \
--keyphrase-vocab /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
--s2orc-data /scratch/lamdo/CSpR/continuous_pretraining/scirepeval/collections.txt \
--output /scratch/lamdo/CSpR/continuous_pretraining/scirepeval/collections_processed_with_negatives.txt \
--max_keyphrase 15000 --add-negatives


CUDA_VISIBLE_DEVICES=0,1,2 python train/cspr/continuous_pretraining/continuous_pretraining.py \
--base-model-name distilbert/distilbert-base-uncased \
--keyphrase-vocab /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
--train-file /scratch/lamdo/CSpR/continuous_pretraining/scirepeval/collections_processed_with_negatives.txt \
--output-dir /scratch/lamdo/CSpR/continuous_pretraining/distilbert+15k+ul/ \
--max-keyphrase 15000 \
--max-seq-length 512 \
--per-device-train-batch-size 16 \
--keyphrase-mlm-probability 0.5 \
--mlm-probability 0.15 \
--bf16 --unlikelihood --ul-alpha 1.0



# inplace (similar to CASPER), i.e., put keyphrase in place instead of appending at the end
python train/cspr/continuous_pretraining/pretraining_data_processing.py \
--keyphrase-vocab /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
--s2orc-data /scratch/lamdo/CSpR/continuous_pretraining/scirepeval/collections.txt \
--output /scratch/lamdo/CSpR/continuous_pretraining/scirepeval/collections_processed_inplace.txt \
--max_keyphrase 15000 --mode "inplace"


CUDA_VISIBLE_DEVICES=0,1,2 python train/cspr/continuous_pretraining/continuous_pretraining.py \
--base-model-name distilbert/distilbert-base-uncased \
--keyphrase-vocab /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
--train-file /scratch/lamdo/CSpR/continuous_pretraining/scirepeval/collections_processed_inplace.txt \
--output-dir /scratch/lamdo/CSpR/continuous_pretraining/distilbert+15k+inplace/ \
--max-keyphrase 15000 \
--max-seq-length 512 \
--truncation-side right \
--per-device-train-batch-size 16 \
--keyphrase-mlm-probability 0.5 \
--mlm-probability 0.15 \
--dataloader-num-workers 4 \
--bf16