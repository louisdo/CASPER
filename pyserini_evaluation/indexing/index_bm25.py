import torch, sys, json, argparse, os, time, traceback
sys.path.append("../../")
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from pyserini_evaluation.indexing.model_name_2_info import model_name_2_path, model_name_2_model_class, model_name_2_is_maxsim
from pyserini_evaluation.indexing.bm25_model import BM25
from tqdm import tqdm
from memory_profiler import profile 
from copy import deepcopy


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, required=True)
    parser.add_argument("--work_dir", type = str, default="../../")
    parser.add_argument("--outfolder", type = str, required=True)
    # parser.add_argument("--chunking_size", type = int, default = 10000)
    parser.add_argument("--remove_collections_folder", type = str2bool, default = False)
    parser.add_argument("--store_documents_in_raw", type = str2bool, default = False)
    parser.add_argument("--num_chunks", type = int, default = 4)
    parser.add_argument("--chunk_idx", type = int, required = True)

    args = parser.parse_args()

    dataset_name = args.dataset
    work_dir = args.work_dir
    outfolder = args.outfolder
    # chunking_size = args.chunking_size
    remove_collections_folder = args.remove_collections_folder
    store_documents_in_raw = args.store_documents_in_raw
    num_chunks = args.num_chunks
    chunk_idx = args.chunk_idx

    dataset_name_2_relative_path = {
        "scifact": "data/beir/scifact",
        "scidocs": "data/beir/scidocs",
        "nfcorpus": "data/beir/nfcorpus",
        "arguana": "data/beir/arguana",
        "fiqa": "data/beir/fiqa",
        "trec-covid": "data/beir/trec-covid",
        "msmarco": "data/msmarco/msmarco",
        "doris_mae": "data/doris_mae/doris_mae",
        "cfscube": "data/cfscube/cfscube",
        "acm_cr": "data/acm_cr/acm_cr",
        "litsearch": "data/litsearch/litsearch",
        "relish": "data/relish/relish",

        "cfscube_taxoindex": "data/cfscube/cfscube_taxoindex",
        "doris_mae_taxoindex": "data/doris_mae/doris_mae_taxoindex",
        
        "irb": "data/irb",
    }


    # load corpus
    corpus_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset_name],
        "corpus.jsonl")
    assert os.path.exists(corpus_path)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            jline = json.loads(line)
            corpus.append(jline)


    outfolder_dataset = os.path.join(outfolder, "collections", f"{dataset_name}__bm25")

    outfile = os.path.join(outfolder_dataset, f"chunk{chunk_idx}.jsonl")

    with open(outfile, "w") as f:
        for line in corpus:
            docid = line.get("_id", line.get("id", None))
            assert docid is not None
            title = line["title"]
            abstract = line["text"]

            content = f"{title}. {abstract}"

            to_write = {
                "id": docid,
                "contents":  " ".join(content.split()[:1000])
            }

            json.dump(to_write, f)
            f.write("\n")




if __name__ == "__main__":
    main()