# python process_dataset.py --dataset_name cfscube --output_folder "/scratch/lamdo/phrase_splade_datasets/cfscube_taxoindex"
# python process_dataset.py --dataset_name doris_mae --output_folder "/scratch/lamdo/phrase_splade_datasets/doris_mae_taxoindex"

import json, pickle, os, random
from tqdm import tqdm
from argparse import ArgumentParser
from pyserini.search.lucene import LuceneSearcher


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

def process_id(input_id, dataset_name):
    if dataset_name == "cfscube":
        return input_id.split("_")[0]
    elif dataset_name == "doris_mae":
        return input_id[2:]
    

def negative_sample_mining(generated_queries, corpus, searcher, dataset_name, k = 50):
    negative_samples = {}
    total = len(generated_queries)
    for docid_ in tqdm(generated_queries, desc = "Mining negative samples", total = total):
        docid = process_id(docid_, dataset_name)
        hits = searcher.search(corpus[docid], k)

        negative_doc_ids = []
        for hit in hits:
            neg_docid = str(hit.docid)
            if docid == neg_docid: continue
            negative_doc_ids.append(neg_docid)

        negative_samples[docid] = negative_doc_ids

    return negative_samples



def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type = str, required = True)
    parser.add_argument("--work_dir", type = str, default = "../../")
    parser.add_argument("--index_folder", type = str, default = "/scratch/lamdo/beir_splade/indexes/")
    parser.add_argument("--output_folder", type = str, required = True)
    

    args = parser.parse_args()
    dataset_name = args.dataset_name
    work_dir = args.work_dir
    index_folder = args.index_folder
    output_folder = args.output_folder

    input_file = os.path.join(dataset_name, "Promptgator_training_queries")
    

    dataset_name_2_corpus_file = {
        "cfscube": "data/cfscube/cfscube_taxoindex/corpus.jsonl",
        "doris_mae": "data/doris_mae/doris_mae_taxoindex/corpus.jsonl"
    }
    corpus_file = os.path.join(work_dir, dataset_name_2_corpus_file[dataset_name])

    index_path_dataset_bm25 = os.path.join(index_folder, f"{dataset_name}_taxoindex__bm25")

    with open(input_file, "rb") as f:
        temp = pickle.load(f)
        generated_queries = {k:v for k,v in temp.items()}

    corpus = {}
    with open(corpus_file) as f:
        for line in f:
            jline = json.loads(line)
            cid = str(jline["_id"])

            title = jline["title"]
            abstract = jline["text"]
            content = title + ". " + abstract

            corpus[cid] = content


    searcher = LuceneSearcher(index_dir = index_path_dataset_bm25)
    # generate triplets

    negative_samples = negative_sample_mining(generated_queries, searcher = searcher, corpus = corpus, dataset_name=dataset_name)

    triplets = []
    for docid_ in generated_queries:
        docid = process_id(docid_, dataset_name)
        pos = corpus[docid]
        all_negs = [corpus[nsid] for nsid in negative_samples[docid]]

        query = generated_queries[docid_]

        for neg in all_negs:
            to_append = [query, pos, neg]

            triplets.append(to_append)

    maybe_create_folder(output_folder)
    output_file = os.path.join(output_folder, "raw.tsv")

    random.shuffle(triplets)
    with open(output_file, "w") as f:
        for query, pos, neg in triplets:
            f.write("\t".join([query, pos, neg]) + "\n")

if __name__ == "__main__":
    main()