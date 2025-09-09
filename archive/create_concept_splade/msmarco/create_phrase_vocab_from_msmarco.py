import torch, os, sys, json, random
from argparse import ArgumentParser
from nounphrase_extractor import CandidateExtractorRegExpNLTK
from splade_inference import get_tokens_scores_of_docs_batch, init_splade_model, scores_candidates, SPLADE_MODEL
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

CANDEXT = CandidateExtractorRegExpNLTK([1,3])
SPLADE_MODEL_NAME = "naver/splade-cocondenser-ensembledistil"

def load_msmarco(path):
    with open(path) as f:
        error_count = 0
        res = []
        for line in f:
            splitted_line = line.split("\t")
            if len(splitted_line) != 2:
                error_count += 1
                continue
            docid, text = splitted_line

            to_append = {"title": "", "abstract": text}
            res.append(to_append)

    return res



def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type = str, required = True)
    parser.add_argument("--num_slices", type = int, default = 8)
    parser.add_argument("--current_slice_index", type = int, default = 0)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--top_k_candidates", type = int, default = 10)
    parser.add_argument("--output_folder", type = str, required = True)

    args = parser.parse_args()

    input_file = args.input_file
    num_slices = args.num_slices
    current_slice_index = args.current_slice_index
    seed = args.seed
    top_k_candidates = args.top_k_candidates
    output_folder = args.output_folder

    init_splade_model(SPLADE_MODEL_NAME)
    BATCH_SIZE = 100

    ds = load_msmarco(input_file)
    vocab = {}

    total_num_docs = len(ds)
    slice_step = int(total_num_docs / num_slices)

    # list_indices = get_sampled_indices(len(ds["train"]), num_samples = total_num_docs, seed = seed)

    list_indices = list(range(len(ds)))
    list_indices = list_indices[slice_step * current_slice_index: slice_step * (current_slice_index + 1)]
    print("First 5 indices:", list_indices[:5])

    for i in tqdm(range(0, len(list_indices), BATCH_SIZE)):
        batch_indices = list_indices[i: i + BATCH_SIZE]
        batch = [ds[j] for j in batch_indices]
        batch_text = [f"""{line['abstract'].lower()}""" for line in batch]

        docs_tokens = SPLADE_MODEL[SPLADE_MODEL_NAME]["tokenizer"](
            batch_text, return_tensors="pt", max_length = 256, padding = True, truncation = True)
        
        batch_tokens_scores = get_tokens_scores_of_docs_batch(docs_tokens, model_name=SPLADE_MODEL_NAME)

        batch_candidates = [CANDEXT(text) for text in batch_text]

        for idx, candidates, tokens_scores in zip(batch_indices, batch_candidates, batch_tokens_scores):

            scores = scores_candidates(candidates, tokens_scores, model_name = SPLADE_MODEL_NAME)
            candidates_scores = Counter({c:s for c,s in zip(candidates, scores)})
            top_candidates = candidates_scores.most_common(top_k_candidates)
            for c,s in top_candidates:
                if len(c) <= 2: continue
                if c not in vocab: vocab[c] = []
                vocab[c].append(idx)

    
    output_file = os.path.join(output_folder, f"{current_slice_index}.json")
    with open(output_file, "w") as f:
        json.dump(vocab, f, indent = 4)

if __name__ == "__main__":
    main()