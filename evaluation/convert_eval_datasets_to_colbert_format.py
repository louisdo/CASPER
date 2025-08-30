# python convert_eval_datasets_to_colbert_format.py --dataset_name scifact --output_folder ../data/
import json, os, shutil, csv
import pandas as pd
from argparse import ArgumentParser


def copy_folder(src, dst):
    if not os.path.exists(src):
        print(f"Source folder '{src}' does not exist.")
        return
    if os.path.exists(dst):
        print(f"Destination folder '{dst}' already exists.")
        return
    
    shutil.copytree(src, dst)
    print(f"Folder copied from '{src}' to '{dst}'.")


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


def write_to_jsonl(filename, data):
    with open(filename, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type = str, default = "scifact", choices=["scifact", "scidocs", "nfcorpus", 
                                                                                    "trec-covid", "doris_mae", "cfscube", "acm_cr",
                                                                                    "litsearch", "relish"])
    parser.add_argument("--source_folder", type = str, required=True)
    parser.add_argument("--output_folder", type = str, required=True)

    args = parser.parse_args()

    dataset_name = args.dataset_name
    output_folder = args.output_folder
    source_folder = args.source_folder

    # /home/lamdo/splade/data
    

    dataset_name_2_dataset_path = {
        "scifact": f"{source_folder}/beir/scifact", 
        "scidocs": f"{source_folder}/beir/scidocs",
        "nfcorpus": f"{source_folder}/beir/nfcorpus",
        "trec-covid": f"{source_folder}/beir/trec-covid",
        "doris_mae": f"{source_folder}/doris_mae/doris_mae",
        "cfscube": f"{source_folder}/cfscube/cfscube",
        "acm_cr": f"{source_folder}/acm_cr/acm_cr",
        "litsearch": f"{source_folder}/litsearch/litsearch",
        "relish": f"{source_folder}/relish/relish",
    }

    dataset_output_folder = os.path.join(output_folder, dataset_name)
    maybe_create_folder(dataset_output_folder)

    output_collection_file = os.path.join(output_folder, dataset_name, "collection.jsonl")
    output_query_file = os.path.join(output_folder, dataset_name, "queries.jsonl")
    output_id2originalid_file = os.path.join(output_folder, dataset_name, "id2originalid.json")
    output_qrel_folder = os.path.join(output_folder, dataset_name, "qrels")


    input_collection_file = os.path.join(dataset_name_2_dataset_path[dataset_name], "corpus.jsonl")
    input_query_file = os.path.join(dataset_name_2_dataset_path[dataset_name], "queries.jsonl")
    input_qrel_folder = os.path.join(dataset_name_2_dataset_path[dataset_name], "qrels")

    # reading collection
    collection = []
    collection_intid_2_id = []
    with open(input_collection_file) as f:
        for i, line in enumerate(f):
            jline = json.loads(line)

            id = i
            original_id = jline.get("_id") if jline.get("_id") is not None else jline.get("id")
            title = jline.get("title").replace("\n", " ")
            text = jline.get("text").replace("\n", " ")

            assert original_id is not None

            collection_intid_2_id.append(original_id)
            collection.append({"id": id, "title": title, "text": text})

    # reading query
    queries = []
    queries_intid_2_id = []
    with open(input_query_file) as f:
        for i, line in enumerate(f):
            jline = json.loads(line)
            original_id = jline.get("_id") if jline.get("_id") is not None else jline.get("id")
            id = i
            text = jline.get("text").replace("\n", " ")

            assert original_id is not None

            queries_intid_2_id.append(original_id)
            queries.append({"id": id, "title": "", "text": text})


    # writing collection
    write_to_jsonl(output_collection_file, collection)

    # writing queries
    write_to_jsonl(output_query_file, queries)

    # writing output_id2originalid_file
    with open(output_id2originalid_file, "w") as f:
        json.dump({"corpus": collection_intid_2_id, "queries": queries_intid_2_id}, f, indent = 4)

    # finally, copy qrels folder

    copy_folder(src = input_qrel_folder, dst = output_qrel_folder)

if __name__ == "__main__":
    main()