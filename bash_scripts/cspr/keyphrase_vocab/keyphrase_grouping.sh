INPUT_FOLDER="/scratch/lamdo/CSpR/keyphrase_vocab/s2orc_phrase_vocab/"
OUTPUT_FILE="/scratch/lamdo/CSpR/keyphrase_vocab/keyphrase_grouping/groups.json"
THRESHOLD=0.95
FUZZY_THRESHOLD=95
DEVICE="cuda:2"
TOP_K=50
FREQUENCY_FILTER=5

# first stage
# python train/cspr/keyphrase_vocab/keyphrase_grouping/keyphrase_grouping_first_stage.py \
#   --input $INPUT_FOLDER \
#   --output "${OUTPUT_FILE}.first_stage" \
#   --threshold $THRESHOLD \
#   --device $DEVICE \
#   --top-k $TOP_K \
#   --frequency-filter $FREQUENCY_FILTER


# # second stage
python train/cspr/keyphrase_vocab/keyphrase_grouping/keyphrase_grouping_second_stage.py \
  --input "${OUTPUT_FILE}.first_stage" \
  --output $OUTPUT_FILE \
  --keyphrase-index $INPUT_FOLDER \
  --fuzzy-threshold $FUZZY_THRESHOLD \
  --concurrency 16
