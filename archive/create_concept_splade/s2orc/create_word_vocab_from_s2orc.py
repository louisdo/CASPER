import torch, os, sys, json, random, nltk, string
from argparse import ArgumentParser
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer

BATCH_SIZE = 1000

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
BERT_VOCAB = set(tokenizer.vocab.keys())

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
    parser.add_argument("--total_num_docs", type = int, default = 10000000)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--output_folder", type = str, required = True)

    args = parser.parse_args()

    total_num_docs = args.total_num_docs
    seed = args.seed
    output_folder = args.output_folder
    

    ds = load_dataset("sentence-transformers/s2orc", "title-abstract-pair")
    vocab = {}

    list_indices = get_sampled_indices(len(ds["train"]), num_samples = total_num_docs, seed = seed)

    for i in tqdm(range(0, len(list_indices), BATCH_SIZE)):
        batch_indices = list_indices[i: i + BATCH_SIZE]
        batch = [ds["train"][j] for j in batch_indices]
        batch_text = [f"""{line['title'].lower()} {line['abstract'].lower()}""" for line in batch]

        batch_words = [[tok.strip(string.punctuation) for tok in text.split(" ")] for text in batch_text]

        for idx, words in zip(batch_indices, batch_words):
            for w in words:
                if len(w) <= 2 or w in BERT_VOCAB: continue
                if w not in vocab: vocab[w] = []
                vocab[w].append(idx)

    
    output_file = os.path.join(output_folder, f"0.json")
    with open(output_file, "w") as f:
        json.dump(vocab, f, indent = 4)

if __name__ == "__main__":
    main()