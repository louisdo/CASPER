# python training_dataset_creation.py --input_folder /scratch/lamdo/phrase_splade_datasets/combined_cc+cocit+kp1m+query+title
# python training_dataset_creation.py --input_folder /scratch/lamdo/phrase_splade_datasets/combined_cc_cs+cocit_cs+kp20k+query_cs+title_cs
import os, hashlib, json
from tqdm import tqdm
from argparse import ArgumentParser
from functools import lru_cache

def maybe_create_folder(folder_path):
    """
    Checks if the folder at 'folder_path' exists.
    If it does not exist, creates the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

@lru_cache(maxsize=10000)
def create_hash_id(text):
    """
    Creates a hash ID for the given text using SHA-256.
    Returns the hexadecimal representation of the hash.
    """
    hash_object = hashlib.sha256(text.encode())
    hash_id = hash_object.hexdigest()
    return hash_id[:16]

def process(triplets):

    documents = set([])
    queries = set([])
    for line in tqdm(triplets, desc = "Getting unique documents and queries"):
        query, pos, neg = line
        documents.add(pos)
        documents.add(neg)
        queries.add(query)

    
    documents = list(sorted(documents))
    queries = list(sorted(queries))

    doc2id = {doc:i for i,doc in enumerate(documents)}
    query2id = {query:i for i, query in enumerate(queries)}

    triplets_ids = []
    for line in tqdm(triplets, desc = "Creating triplets"):
        query, pos, neg = line

        query_id = query2id[query]
        pos_id = doc2id[pos]
        neg_id = doc2id[neg]
        triplets_ids.append([query_id, pos_id, neg_id])

    return triplets_ids, {i:doc for i,doc in enumerate(documents)}, {i:query for i, query in enumerate(queries)}
    


def main(): 
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, required = True)

    args = parser.parse_args()
    input_folder = args.input_folder

    input_file = os.path.join(input_folder, "raw.tsv") 
    assert os.path.exists(input_file)

    output_folder = os.path.join(input_folder, "colbert_training_format")
    maybe_create_folder(output_folder)

    with open(input_file) as f:
        error_count = 0
        triplets = []
        for i, line in enumerate(tqdm(f, desc = "Reading input file")):
            splitted_line = line.strip().split("\t")
            if len(splitted_line) == 3: triplets.append(splitted_line)
            else: error_count += 1
        
        print("Number of errors", error_count)

    triplets_ids, doc_metadata, query_metadata = process(triplets)


    triplets_file = os.path.join(output_folder, "triples.train.jsonl")
    with open(triplets_file, "w") as f:
        for line in tqdm(triplets_ids, desc = "Writing triplets"):
            json.dump(line, f)
            f.write("\n")


    corpus_file = os.path.join(output_folder, "corpus.train.tsv")
    with open(corpus_file, "w") as f:
        for doc_id, doc in tqdm(doc_metadata.items(), desc = "Writing document metadata"):
            to_write = "\t".join([str(doc_id), doc]).strip().replace("\n", " ")
            f.write(to_write + "\n")

    query_file = os.path.join(output_folder, "queries.train.tsv")
    with open(query_file, "w") as f:
        for query_id, query in tqdm(query_metadata.items(), desc = "Writing query metadata"):
            to_write = "\t".join([str(query_id), query]).strip().replace("\n", " ")
            f.write(to_write + "\n")

    

if __name__ == "__main__":
    main()