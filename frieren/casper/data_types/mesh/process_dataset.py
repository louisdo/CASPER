# python process_dataset.py --output_file "/scratch/lamdo/phrase_splade_datasets/mesh_descriptors/raw.tsv"
import random
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm
from langdetect import detect as detect_language

def check_document_high_quality(doc):
    # high quality here means lang is english and is long enough
    if detect_language(doc) != "en":
        return False
    
    if len(doc.split(" ")) < 30:
        return False

    return True


def load_dataset_from_huggingface(split = "train", max_collections = 500000):
    ds = load_dataset("allenai/scirepeval", "mesh_descriptors")
    dataset_to_load = ds[split].shuffle(seed = 42)
    
    data = []
    for i, line in enumerate(tqdm(dataset_to_load, desc = "Loading dataset to memory")):
        if i == max_collections: break
        data.append(line)

    return data

def create_random_neg_triplets_for_dataset(dataset, output_file):
    dataset_length = len(dataset)
    with open(output_file, "w") as f:

        for i, line in enumerate(tqdm(dataset, desc = "Creating triplets")):
            title = line["title"] if line["title"] else ""
            abstract = line["abstract"] if line["abstract"] else ""

            title = title.lower()
            abstract = abstract.lower()

            labels_text = line["descriptor"]

            neg_title, neg_abstract, neg_labels_text = None, None, None
            random_indices = random.sample(range(dataset_length), 3)
            for random_index in random_indices:
                if random_index == i: 
                    continue
                random_line = dataset[random_index]
                random_line_labels_text = random_line["descriptor"]

                if not labels_text == random_line_labels_text:
                    neg_title = random_line["title"].lower()
                    neg_abstract = random_line["abstract"].lower()
                    neg_labels_text = random_line_labels_text
                    break
            
            if neg_title is None:
                print('Cannot find random neg for', line["doc_id"])
                continue

            triplet_query = labels_text.replace("\n", " ")
            triplet_pos = (title + ". " + abstract).replace("\n", " ")
            triplet_neg = (neg_title + ". " + neg_abstract).replace("\n", " ")

            if not check_document_high_quality(triplet_pos) or not check_document_high_quality(triplet_neg):
                continue

            to_write = "\t".join([triplet_query, triplet_pos, triplet_neg]).strip()
            f.write(to_write + "\n")

def main():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type = str, required = True)
    parser.add_argument("--max_collections", type = int, default = 500000)

    args = parser.parse_args()

    output_file = args.output_file
    max_collections = args.max_collections

    ds = load_dataset_from_huggingface(max_collections=max_collections)

    create_random_neg_triplets_for_dataset(ds, output_file)


if __name__ == "__main__":
    main()