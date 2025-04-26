# python kp_datasets.py --output_file /scratch/lamdo/splade_kp_datasets/kp20kbiomed/raw.tsv
# python kp_datasets.py --output_file /scratch/lamdo/phrase_splade_datasets/kp/raw.tsv 
# python kp_datasets.py --output_file /scratch/lamdo/phrase_splade_datasets/kp1m/raw.tsv --max_collections 1000000

import json, random
import pandas as pd
from copy import deepcopy
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm


def create_tsv(data, file_name):
    with open(file_name, "w") as f:
        for line in tqdm(data, desc = "Writing data"):
            to_write = "\t".join(line).strip()
            f.write(to_write + "\n")

def process_kp20k(max_documents = 1000000):
    df_kp20k = pd.read_json("hf://datasets/memray/kp20k/train.json", lines=True)
    df_kp20k = df_kp20k.sample(frac=1).reset_index(drop=True) # shuffling

    print("Done downloading")

    all_pairs = []
    records = df_kp20k.to_dict("records")

    for i, line in enumerate(tqdm(records, desc = "Reading kp20k")):
        if i == max_documents:
            break
        title = line["title"]
        abstract = line["abstract"]

        keyphrases = line["keywords"]

        text = f"{title}. {abstract}".lower().replace("\n", " ")

        query = ", ".join([kp.lower() for kp in keyphrases])

        all_pairs.append([query, text])

    all_triplets = []
    for i in tqdm(range(len(all_pairs)), desc = "creating triplets for kp20k"):
        while True:
            j = random.choice(range(len(all_pairs)))
            if j != i: break
        negative_doc = all_pairs[j][1]

        to_append = deepcopy(all_pairs[i])
        to_append += [negative_doc]

        all_triplets.append(to_append)

    return all_triplets


def process_kpbiomed(max_documents = 1000000):
    ds_kpbiomed = load_dataset("taln-ls2n/kpbiomed", "medium")

    all_pairs = []

    shuffled_kpbiomed = ds_kpbiomed["train"].shuffle(seed = 42)
    for i, line in enumerate(tqdm(shuffled_kpbiomed, desc = "Reading kpbiomed")):
        if i == max_documents: break
        title = line["title"]
        abstract = line["abstract"]

        keyphrases = line["keyphrases"]

        text = f"{title}. {abstract}".lower().replace("\n", " ")

        query = ", ".join([kp.lower() for kp in keyphrases])

        all_pairs.append([query, text])

    all_triplets = []
    for i in tqdm(range(len(all_pairs)), desc = "creating triplets for kpbiomed"):
        while True:
            j = random.choice(range(len(all_pairs)))
            if j != i: break
        negative_doc = all_pairs[j][1]

        to_append = deepcopy(all_pairs[i])
        to_append += [negative_doc]

        all_triplets.append(to_append)

    return all_triplets




def main():
    parser = ArgumentParser()
    parser.add_argument("--max_collections", type = int, default = 500000)
    parser.add_argument("--output_file", type = str, required=True)

    args = parser.parse_args()

    max_collections = args.max_collections
    output_file = args.output_file

    triplets_kp20k = process_kp20k(max_documents=int(max_collections / 2))
    triplets_kpbiomed = process_kpbiomed(max_documents = int(max_collections / 2))

    final_dataset = triplets_kp20k + triplets_kpbiomed
    random.shuffle(final_dataset)

    print("Number of datapoints", len(final_dataset))

    create_tsv(data = final_dataset, file_name=output_file)


if __name__ == "__main__":
    main()