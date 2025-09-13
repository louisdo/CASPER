#!/bin/bash
# :TODO: Write README.md
# Instruct download BM25 model, download Data
# Keyphrase Evaluation


echo "Download output directories under $OUT_FOLDER..."

# set the OUT_FOLDER variable before running this script.
export OUT_FOLDER="your_full_path"
if [ -z "$OUT_FOLDER" ]; then
    echo "Error: OUT_FOLDER is not set. Please set the variable and try again."
    exit 1
fi

# Create necessary folders.
# so we create it here in case it doesn't exist.
echo "Creating output directories under $OUT_FOLDER..."
mkdir -p "$OUT_FOLDER/index"
mkdir -p "$OUT_FOLDER/collections"
mkdir -p "$OUT_FOLDER/indexes"
mkdir -p "pyserini_evaluation/results_ablationadjustingbeta"
mkdir -p "pyserini_evaluation/metadata"

# Run the indexing script.
echo "Running the indexing script..."
cd pyserini_evaluation/indexing
OUT_FOLDER=$OUT_FOLDER bash index.sh

# Change directory to pyserini_evaluation to run the evaluation script.
echo "Changing directory to pyserini_evaluation..."
cd ../ || exit #cd pyserini_evaluation

echo "Running the evaluation script..."
INDEX_FOLDER="$OUT_FOLDER/indexes" bash eval.sh

echo "Indexing and Evaluating finished."