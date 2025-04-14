# python build_corpus_from_eru_kg_dataset.py

import os, sys, json, hashlib
sys.path.append("../s2orc")
from argparse import ArgumentParser
from nounphrase_extractor import CandidateExtractorRegExpNLTK
from splade_inference import get_tokens_scores_of_docs_batch, init_splade_model, scores_candidates, SPLADE_MODEL, SPLADE_MODEL_NAME
from tqdm import tqdm
from collections import Counter

CANDEXT = CandidateExtractorRegExpNLTK([1,3])
BATCH_SIZE = 100


def hash_document(document: str) -> str:
    """
    Hashes a document using the SHA-256 algorithm.

    Args:
        document (str): The input document as a string.

    Returns:
        str: The hexadecimal representation of the hash.
    """
    # Encode the document to bytes
    document_bytes = document.encode('utf-8')
    
    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()
    
    # Update the hash object with the document bytes
    sha256_hash.update(document_bytes)
    
    # Return the hexadecimal digest of the hash
    return sha256_hash.hexdigest()


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, default="/scratch/lamdo/unArxive/keyphrase_informativeness_combined_references/triplets_hardneg")
    parser.add_argument("--top_k_candidates", type = str, default = 5)
    parser.add_argument("--output_file", type = str, default="./vocab/vocab_query.json")
    parser.add_argument("--doc_type", type= str, default = "query")

    args = parser.parse_args()

    input_folder = args.input_folder
    top_k_candidates = args.top_k_candidates
    doc_type = args.doc_type
    output_file = args.output_file

    assert doc_type in ["query", "doc"]

    init_splade_model()

    input_file = os.path.join(input_folder, "raw.tsv")

    vocab = Counter()
    docs = []
    visited = set([])
    line_index = 0 if doc_type == 'query' else 1
    with open(input_file) as f:
        for line in tqdm(f, desc = f"Reading {doc_type}"):
            splitted_line = line.split("\t")

            doc = splitted_line[line_index]
            doc_id = hash_document(doc)
            if doc_id not in visited:
                docs.append(doc)
                visited.add(doc_id)

    for i in tqdm(range(0, len(docs), BATCH_SIZE)):
        batch_text = docs[i:i+BATCH_SIZE]

        docs_tokens = SPLADE_MODEL[SPLADE_MODEL_NAME]["tokenizer"](
            batch_text, return_tensors="pt", max_length = 256, padding = True, truncation = True)
        
        batch_tokens_scores = get_tokens_scores_of_docs_batch(docs_tokens, model_name=SPLADE_MODEL_NAME)

        batch_candidates = [CANDEXT(text) for text in batch_text]

        for candidates, tokens_scores in zip(batch_candidates, batch_tokens_scores):
            scores = scores_candidates(candidates, tokens_scores, model_name = SPLADE_MODEL_NAME)
            candidates_scores = Counter({c:s for c,s in zip(candidates, scores)})
            top_candidates = candidates_scores.most_common(top_k_candidates)
            vocab.update([item[0] for item in top_candidates])

    
    with open(output_file, "w") as f:
        json.dump(vocab, f, indent = 4)

if __name__ == "__main__":
    main()