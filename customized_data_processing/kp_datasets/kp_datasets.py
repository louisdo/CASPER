# python kp_datasets.py --output_file /scratch/lamdo/splade_kp_datasets/kp20kbiomed/raw.tsv

import json, random
import pandas as pd
from copy import deepcopy
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm


def create_tsv(data, file_name):
    try:
        # Create a DataFrame from the data
        df = pd.DataFrame(data, columns=['query', 'pos', 'neg'])
        
        # Save the DataFrame to a TSV file
        df.to_csv(file_name, sep='\t', index=False, header = False, escapechar='\\')
        
        print(f"TSV file '{file_name}' has been created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_kp20k(max_documents = 1000000):
    df_kp20k = pd.read_json("hf://datasets/memray/kp20k/train.json", lines=True)

    print("Done downloading")

    all_pairs = []

    for i, line in enumerate(tqdm(df_kp20k.to_dict("records"))):
        title = line["title"]
        abstract = line["abstract"]

        keyphrases = line["keywords"]

        text = f"{title}. {abstract}"

        query = ", ".join(keyphrases)

        all_pairs.append([query, text])

    all_pairs = all_pairs[:max_documents]

    all_triplets = []
    for i in tqdm(range(len(all_pairs)), desc = "creating triplets"):
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

    for i, line in enumerate(ds_kpbiomed["train"]):
        title = line["title"]
        abstract = line["abstract"]

        keyphrases = line["keyphrases"]

        text = f"{title}. {abstract}"

        query = ", ".join(keyphrases)

        all_pairs.append([query, text])

    all_pairs = all_pairs[:max_documents]

    all_triplets = []
    for i in range(len(all_pairs)):
        j = random.choice([k for k in range(len(all_pairs)) if k != i])
        negative_doc = all_pairs[j][1]

        to_append = deepcopy(all_pairs[i])
        to_append += [negative_doc]

        all_triplets.append(to_append)

    return all_triplets




def main():
    parser = ArgumentParser()
    parser.add_argument("--max_collections", type = int, default = 3000000)
    parser.add_argument("--output_file", type = str, required=True)

    args = parser.parse_args()

    max_collections = args.max_collections
    output_file = args.output_file

    triplets_kp20k = process_kp20k(max_documents=max_collections)
    # triplets_kpbiomed = process_kpbiomed(max_documents = int(max_collections / 2))

    print("Number of datapoints", len(triplets_kp20k))

    create_tsv(data = triplets_kp20k, file_name=output_file)


if __name__ == "__main__":
    main()