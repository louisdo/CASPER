# python process_dataset.py --output_file "/scratch/lamdo/phrase_splade_datasets/fos/raw.tsv" --temp_folder_for_processing "/scratch/lamdo/phrase_splade_datasets/fos/temp/"
import json, os, shutil, random
from argparse import ArgumentParser
from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from langdetect import detect as detect_language


def maybe_create_folder(folder_path):
    """
    Check if a folder exists and create it if it doesn't.

    Args:
    folder_path (str): The path of the folder to be created.

    Returns:
    bool: True if the folder was created, False if it already existed.
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
            return True
        except OSError as e:
            print(f"Error creating folder {folder_path}: {e}")
            return False
    else:
        print(f"Folder already exists: {folder_path}")
        return False


def remove_folder(folder_path):
    """
    Removes a folder and all its contents.
    
    Args:
    folder_path (str): The path to the folder to be removed.
    
    Returns:
    bool: True if the folder was successfully removed, False otherwise.
    """
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Remove the folder and all its contents
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' has been successfully removed.")
            return True
        else:
            print(f"The folder '{folder_path}' does not exist.")
            return False
    except Exception as e:
        print(f"An error occurred while trying to remove the folder: {e}")
        return False


def load_dataset_from_huggingface(split = "train"):
    ds = load_dataset("allenai/scirepeval", "fos")

    data = []
    for line in tqdm(ds[split], desc = "Loading dataset to memory"):
        data.append(line)

    return data


def write_to_collection_pyserini_format(dataset, out_folder, chunk_size = 2000):
    for chunk_idx, i in enumerate(range(0, len(dataset), chunk_size)):
        chunk = dataset[i:i+chunk_size]

        chunk_out_file = os.path.join(out_folder, f"{chunk_idx}.jsonl")
        with open(chunk_out_file, "w") as f:
            for line in chunk:
                title = line["title"]
                abstract = line["abstract"]
                doc_id = line["doc_id"]
                labels_text = line["labels_text"]

                to_write = {
                    "id": doc_id,
                    "contents": title.lower() + ". " + abstract.lower(),
                    "labels_text": labels_text
                }

                json.dump(to_write, f)
                f.write("\n")


def do_indexing(collection_folder, index_folder):
    command = f"""python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input "{collection_folder}" \
        --index "{index_folder}" \
        --generator DefaultLuceneDocumentGenerator \
        --threads 8 --storePositions --storeDocvectors --storeRaw"""
    
    os.system(command)

    remove_folder(collection_folder)

def check_overlap_label_text(labels1, labels2):
    return len(set(labels1).intersection(set(labels2))) > 0

def check_document_high_quality(doc):
    # high quality here means lang is english and is long enough
    if detect_language(doc) != "en":
        return False
    
    if len(doc.split(" ")) < 30:
        return False

    return True


def create_hard_neg_triplets_for_dataset(dataset, index_folder, output_file):
    searcher = LuceneSearcher(index_dir = index_folder)

    with open(output_file, "w") as f:

        for line in tqdm(dataset, desc = "Creating triplets"):
            title = line["title"].lower() # use as query
            abstract = line["abstract"].lower()
            labels_text = [item.lower() for item in line["labels_text"]]

            hits = searcher.search(title, k = 5)
            hits_raws = [json.loads(hit.lucene_document.get("raw")) for hit in hits]
            hits_raws = [hit for hit in hits_raws if hit["id"] != line["doc_id"] and not check_overlap_label_text(hit["labels_text"], labels_text)]

            if not hits_raws: 
                print('Cannot find hard neg for', line["doc_id"])
                continue
            raw = hits_raws[0]

            content = raw.get("contents")

            triplet_query = ", ".join(labels_text).replace("\n", " ")
            triplet_pos = (title + ". " + abstract).replace("\n", " ")
            triplet_neg = content.lower().replace("\n", " ")

            if not check_document_high_quality(triplet_pos) or not check_document_high_quality(triplet_neg):
                continue

            to_write = "\t".join([triplet_query, triplet_pos, triplet_neg]).strip()
            f.write(to_write + "\n")

    

def create_random_neg_triplets_for_dataset(dataset, output_file):
    dataset_length = len(dataset)
    with open(output_file, "w") as f:

        for i, line in enumerate(tqdm(dataset, desc = "Creating triplets")):
            title = line["title"].lower()
            abstract = line["abstract"].lower()
            labels_text = set([item.lower() for item in line["labels_text"]])

            neg_title, neg_abstract, neg_labels_text = None, None, None
            random_indices = random.sample(range(dataset_length), 3)
            for random_index in random_indices:
                if random_index == i: 
                    continue
                random_line = dataset[random_index]
                random_line_labels_text = set([item.lower() for item in random_line["labels_text"]])

                if not labels_text == random_line_labels_text:
                    neg_title = random_line["title"].lower()
                    neg_abstract = random_line["abstract"].lower()
                    neg_labels_text = random_line_labels_text
                    break
            
            if neg_title is None:
                print('Cannot find random neg for', line["doc_id"])
                continue

            triplet_query = ", ".join(list(labels_text - neg_labels_text)).replace("\n", " ")
            triplet_pos = (title + ". " + abstract).replace("\n", " ")
            triplet_neg = (neg_title + ". " + neg_abstract).replace("\n", " ")

            if not check_document_high_quality(triplet_pos) or not check_document_high_quality(triplet_neg):
                continue

            to_write = "\t".join([triplet_query, triplet_pos, triplet_neg]).strip()
            f.write(to_write + "\n")




def main():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type = str, required = True)
    parser.add_argument("--temp_folder_for_processing", type = str, default = "./temp")
    parser.add_argument("--neg_type", type = str, default="random")

    args = parser.parse_args()

    output_file = args.output_file
    temp_folder_for_processing = args.temp_folder_for_processing
    neg_type = args.neg_type

    assert neg_type in ["random", "hard"]


    ds = load_dataset_from_huggingface()

    if neg_type == "hard":
        maybe_create_folder(temp_folder_for_processing)

        collection_folder = os.path.join(temp_folder_for_processing, "collections/")
        index_folder = os.path.join(temp_folder_for_processing, "index/")

        remove_folder(collection_folder)
        remove_folder(index_folder)

        maybe_create_folder(collection_folder)
        maybe_create_folder(index_folder)

        write_to_collection_pyserini_format(dataset = ds, out_folder = collection_folder)
        do_indexing(collection_folder = collection_folder, index_folder = index_folder)

        create_hard_neg_triplets_for_dataset(dataset = ds, index_folder = index_folder, output_file=output_file)


        remove_folder(collection_folder)
        remove_folder(index_folder)
    elif neg_type == "random":
        create_random_neg_triplets_for_dataset(dataset = ds, output_file = output_file)


if __name__ == "__main__":
    main()