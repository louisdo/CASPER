import torch, os, sys, json, random
sys.path.append("..")
import numpy as np
from argparse import ArgumentParser
from nounphrase_extractor import CandidateExtractorRegExpNLTK
from splade_inference import get_tokens_scores_of_docs_batch, init_splade_model, scores_candidates, SPLADE_MODEL, SPLADE_MODEL_NAME
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

CANDEXT = CandidateExtractorRegExpNLTK([1,3])

def main():
    parser = ArgumentParser()
    parser.add_argument("--path", type = str, required = True)
    parser.add_argument("--num_slices", type = int, default = 8)
    parser.add_argument("--current_slice_index", type = int, default = 0)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--top_k_candidates", type = int, default = 10)
    parser.add_argument("--output_folder", type = str, required = True)

    args = parser.parse_args()

    path = args.path
    num_slices = args.num_slices
    current_slice_index = args.current_slice_index
    seed = args.seed
    top_k_candidates = args.top_k_candidates
    output_folder = args.output_folder


    init_splade_model()
    BATCH_SIZE = 100

    ds = []
    with open(path) as f:
        for line in tqdm(f, desc = "Reading input"):
            jline = json.loads(line)
            jline["title"] = jline.get("title") if jline.get("title") else ""
            jline["abstract"] = jline.get("abstract") if jline.get("abstract") else ""
            ds.append(jline)
    list_indices = np.array_split(range(len(ds)), num_slices)[current_slice_index].tolist()

    vocab = {}
    for i in tqdm(range(0, len(list_indices), BATCH_SIZE)):
        batch_indices = list_indices[i: i + BATCH_SIZE]
        batch = [ds[j] for j in batch_indices]
        batch_text = [f"""{line['title'].lower()}. {line['abstract'].lower()}""" for line in batch]

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