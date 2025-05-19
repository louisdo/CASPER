# python prepare_training_dataset.py --output_file /scratch/lamdo/s2orc/processed/query_triplets/raw.tsv
import random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from argparse import ArgumentParser


def slightly_process_text(text):
    return text.replace("\n", "").replace("\t", "")


def write_tsv(triplets, output_file):
    with open(output_file, "w") as f:
        for line in tqdm(triplets, desc = "Writing dataset"):
            if len(line) != 3:
                print("Erroneous line!")
            to_write = "\t".join([str(item) for item in line])
            f.write(to_write + "\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type = str, required = True)

    args = parser.parse_args()

    output_file = args.output_file


    ds = load_dataset("allenai/scirepeval", "search")

    query_triplets = []
    for line in tqdm(ds["train"], desc = "Reading dataset and create triplets"):
        query = line.get("query")
        assert isinstance(query, str)
        query = slightly_process_text(query)

        positives = []
        negatives = []

        candidates = line.get("candidates")
        for cand in candidates:
            score = cand.get('score', 0)
            title = slightly_process_text(cand.get("title"))
            abstract = slightly_process_text(cand.get("abstract"))

            text = f"{title.lower()}. {abstract.lower()}"

            if score == 0:
                negatives.append(text)
            elif score > 0: 
                positives.append(text)

        if not positives or not negatives: continue

        for pos in positives:
            if not pos: continue
            neg = random.choice(negatives)
            query_triplets.append([query, pos, neg])


    write_tsv(query_triplets, output_file)


if __name__ == "__main__":
    main()