python train/cspr/keyphrase_vocab/keyphrase_vocab_build/greedy_vocab_builder.py \
  --input_folder /scratch/lamdo/CSpR/keyphrase_vocab/s2orc_phrase_vocab/ \
  --grouping_file /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/groups.json \
  --num_phrases 100000 \
  --phrase_min_frequency 5 \
  --output_file /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json.grouped


python train/cspr/keyphrase_vocab/keyphrase_vocab_build/keyphrase_vocab_filtering.py \
  --input /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json.grouped \
  --output /scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/keyphrase_vocab.json \
  --checkpoint /scratch/lamdo/CSpR/keyphrase_vocab/filtering_checkpoint.json \
  --concurrency 16
