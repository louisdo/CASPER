CUDA_VISIBLE_DEVICES=0 python train/cspr/init_new_pmb.py \
--wiki-definition "/scratch/lamdo/CSpR/keyphrase_vocab/wiki_definition.json" \
--keyphrase-vocab "/scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json" \
--output-model-path "test_model/" \
--keyphrase-vocab-size 50000 \
--batch-size 64 \
--pooling mean