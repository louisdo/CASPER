# python aol.py --input_file /scratch/lamdo/aol_concept_annotations/aol.2024.concept.annotation.data.json --max_collections 1000000000 --output_file /scratch/lamdo/phrase_splade_datasets/aol_concept_annotations/raw.tsv

import json, random
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from copy import deepcopy

def create_tsv(data, file_name):
    with open(file_name, "w") as f:
        for line in tqdm(data, desc = "Writing data"):
            to_write = "\t".join(line).strip()
            f.write(to_write + "\n")

def process_aol_concept_annotations(records, max_documents = 1000000):
    all_pairs = []

    for i, line in enumerate(tqdm(records, desc = "Reading aol concept annotations")):
        if i == max_documents:
            break
        title = line["title"]
        abstract = line["abstract"]
        abstract = abstract if abstract is not None else ""

        keyphrases = line["keywords"]

        text = f"{title}. {abstract}".lower().replace("\n", " ").strip()

        query = ", ".join([kp.lower() for kp in keyphrases])

        all_pairs.append([query, text])

    all_triplets = []
    for i in tqdm(range(len(all_pairs)), desc = "creating triplets for aol concept annotations"):
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
    parser.add_argument("--input_file", type = str, required = True)
    parser.add_argument("--output_file", type = str, required = True)
    parser.add_argument("--max_collections", type = int, default = 500000)

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    max_collections = args.max_collections

    with open(input_file) as f:
        records = json.load(f)


    triplets = process_aol_concept_annotations(
        records, max_collections
    )

    create_tsv(triplets, output_file)


if __name__ == "__main__":
    main()