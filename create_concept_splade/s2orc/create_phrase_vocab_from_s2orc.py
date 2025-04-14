import torch, os, sys, json, random
from argparse import ArgumentParser
from nounphrase_extractor import CandidateExtractorRegExpNLTK
from splade_inference import get_tokens_scores_of_docs_batch, init_splade_model, scores_candidates, SPLADE_MODEL, SPLADE_MODEL_NAME
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

CANDEXT = CandidateExtractorRegExpNLTK([1,3])


def get_sampled_indices(length_dataset, num_samples, seed):
    if os.path.exists(f"sampled_indices_seed{seed}.json"):
        with open(f"sampled_indices_seed{seed}.json") as f:
            res = json.load(f)
        return res
    
    random.seed(seed)
    res = random.sample(range(length_dataset), num_samples)
    with open(f"sampled_indices_seed{seed}.json", "w") as f:
        json.dump(res, f)
    return res


def main():
    parser = ArgumentParser()
    parser.add_argument("--num_slices", type = int, default = 8)
    parser.add_argument("--current_slice_index", type = int, default = 0)
    parser.add_argument("--total_num_docs", type = int, default = 10000000)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--top_k_candidates", type = int, default = 10)
    parser.add_argument("--output_folder", type = str, required = True)

    args = parser.parse_args()

    num_slices = args.num_slices
    current_slice_index = args.current_slice_index
    total_num_docs = args.total_num_docs
    seed = args.seed
    top_k_candidates = args.top_k_candidates
    output_folder = args.output_folder

    assert total_num_docs % num_slices == 0

    init_splade_model()
    BATCH_SIZE = 100

    ds = load_dataset("sentence-transformers/s2orc", "title-abstract-pair")
    vocab = Counter()

    slice_step = int(total_num_docs / num_slices)

    list_indices = get_sampled_indices(len(ds["train"]), num_samples = total_num_docs, seed = seed)

    list_indices = list_indices[slice_step * current_slice_index: slice_step * (current_slice_index + 1)]
    print("First 5 indices:", list_indices[:5])

    for i in tqdm(range(0, len(list_indices), BATCH_SIZE)):
        batch = [ds["train"][j] for j in list_indices[i: i + BATCH_SIZE]]
        batch_text = [f"""{line['title'].lower()}. {line['abstract'].lower()}""" for line in batch]

        docs_tokens = SPLADE_MODEL[SPLADE_MODEL_NAME]["tokenizer"](
            batch_text, return_tensors="pt", max_length = 256, padding = True, truncation = True)
        
        batch_tokens_scores = get_tokens_scores_of_docs_batch(docs_tokens, model_name=SPLADE_MODEL_NAME)

        batch_candidates = [CANDEXT(text) for text in batch_text]

        for candidates, tokens_scores in zip(batch_candidates, batch_tokens_scores):
            scores = scores_candidates(candidates, tokens_scores, model_name = SPLADE_MODEL_NAME)
            candidates_scores = Counter({c:s for c,s in zip(candidates, scores)})
            top_candidates = candidates_scores.most_common(top_k_candidates)
            vocab.update([item[0] for item in top_candidates])

    
    output_file = os.path.join(output_folder, f"{current_slice_index}.json")
    with open(output_file, "w") as f:
        json.dump(vocab, f, indent = 4)

if __name__ == "__main__":
    main()