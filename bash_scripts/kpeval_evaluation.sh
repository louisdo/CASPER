#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

# List of JSON files to process
# files=(
#   # "inspec--phrase_splade_33.json"
#   # "krapivin--phrase_splade_33.json"
#   # "nus--phrase_splade_33.json"
#   # "semeval--phrase_splade_33.json"
#   # # "kp20k--phrase_splade_33.json"
#   "semeval--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json"
#   "inspec--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json"
#   "nus--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json"
#   "krapivin--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json"
#   # # "kp20k--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty.json"
#   "semeval--autokeygen-1.json"
#   "inspec--autokeygen-1.json"
#   "nus--autokeygen-1.json"
#   "krapivin--autokeygen-1.json"
#   # # "kp20k--autokeygen-1.json"
#   "semeval--copyrnn-1.json"
#   "inspec--copyrnn-1.json"
#   "nus--copyrnn-1.json"
#   "krapivin--copyrnn-1.json"
#   # "kp20k--copyrnn-1.json"
#   # "semeval--uokg-1.json"
#   # "inspec--uokg-1.json"
#   # "nus--uokg-1.json"
#   # "krapivin--uokg-1.json"
#   # "kp20k--uokg-1.json"
#   # "semeval--tpg-1.json"
#   # "inspec--tpg-1.json"
#   # "inspec--tpg-2.json"
#   # "inspec--tpg-3.json"
#   # "kp20k--tpg-2.json"
#   # "krapivin--tpg-2.json"
#   # "krapivin--tpg-3.json"
#   # "nus--tpg-2.json"
#   # "nus--tpg-3.json"
#   # "semeval--tpg-2.json"
#   # "semeval--tpg-3.json"
#   # "semeval--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_10.json"
#   # "inspec--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_10.json"
#   # "nus--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_10.json"
#   # "krapivin--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_10.json"
#   # "kp20k--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_10.json"
#   # "semeval--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_50.json"
#   # "inspec--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_50.json"
#   # "nus--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_50.json"
#   # "krapivin--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_50.json"
#   # "kp20k--retrieval_based_ukg_custom_trained_combined_references_nounphrase_v6-1_position_penalty+length_penalty_neighborsize_50.json"
#   # "semeval--embedrank_sentence_transformers_all-MiniLM-L12-v2.json"
#   # "inspec--embedrank_sentence_transformers_all-MiniLM-L12-v2.json"
#   # "nus--embedrank_sentence_transformers_all-MiniLM-L12-v2.json"
#   # "krapivin--embedrank_sentence_transformers_all-MiniLM-L12-v2.json"
#   # "kp20k--embedrank_sentence_transformers_all-MiniLM-L12-v2.json"
# )

files=(
  "inspec--phrase_splade_38.json"
  # "krapivin--phrase_splade_38.json"
  # "nus--phrase_splade_38.json"
  # "semeval--phrase_splade_38.json"
)

# Process each file
for file in "${files[@]}"; do
  echo "Processing $file..."

  # input_file="/scratch/lamdo/phrase_splade_keyphrase_generation_results/$file"
  # input_file="/scratch/lamdo/keyphrase_generation_results/results_ongoing/$file" \

  input_file="/scratch/lamdo/phrase_splade_keyphrase_generation_results/$file" \
  output_dir="_gitig_samples/" \
  top_k=10 \
  python metrics/convert_splade_file.py

  python utils/phrase_splade_evaluation.py \
    --config-file kpeval/config.gin \
    --jsonl-file "_gitig_samples/_all_keyphrase_$file" \
    --metrics diversity,meteor,approximate_matching,exact_matching,rouge,bert_score,semantic_matching,unieval \
    --log-file-prefix _gitig_results/
done
