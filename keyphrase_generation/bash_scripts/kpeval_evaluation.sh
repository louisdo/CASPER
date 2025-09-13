#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

# List of JSON files to process
files=(
  "inspec--lamdo_casper.json"
  "krapivin--lamdo_casper.json"
  "nus--lamdo_casper.json"
  "semeval--lamdo_casper.json"
)


# Process each file
for file in "${files[@]}"; do
  echo "Processing $file..."

  input_file="_gitig_kp_output/$file" \
  output_dir="_gitig_kp_results/" \
  top_k=10 \
  python3.10 utils/convert_splade_file.py

  python3.10 utils/phrase_splade_evaluation.py \
    --config-file config.gin \
    --jsonl-file "_gitig_kp_results/_all_keyphrase_$file" \
    --metrics diversity,exact_matching,semantic_matching \
    --log-file-prefix _gitig_kp_results
done

