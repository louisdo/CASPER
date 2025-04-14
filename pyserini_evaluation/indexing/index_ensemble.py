import torch, sys, json, argparse, os, time, traceback
sys.path.append("../../")
import numpy as np
from copy import deepcopy
from transformers import AutoModelForMaskedLM, AutoTokenizer
from pyserini_evaluation.indexing.model_name_2_info import model_name_2_path, model_name_2_model_class, model_name_2_is_maxsim
from pyserini_evaluation.indexing.utils import maybe_create_folder, remove_folder, torch_csr_to_scipy_csr
# from scipy.sparse import csr_matrix, vstack, save_npz
from pyserini_evaluation.indexing.utils import merge_dicts

from tqdm import tqdm

PHRASE_SPLADE = {
    "model": None,
    "tokenizer": None,
    "reverse_voc": None,
    "model_name": None,
    "is_maxsim": None,
    "voc": None
}

SPLADE = {
    "model": None,
    "tokenizer": None,
    "reverse_voc": None,
    "model_name": None,
    "is_maxsim": None,
    "voc": None
}

MAX_LENGTH = 256

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def init_model_helper(model_name, CONTAINER):
    if CONTAINER["model_name"] is None or CONTAINER["model_name"] != model_name:
        print(f"Loading {model_name} for the first time. This will be done only once for {model_name}")
        model_type_or_dir = model_name_2_path.get(model_name)
        model_class = model_name_2_model_class.get(model_name)
        is_maxsim = model_name_2_is_maxsim.get(model_name)

        if model_type_or_dir is None and model_class is None:
            raise NotImplementedError



        model = model_class(model_type_or_dir, agg="max").to(DEVICE)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        voc = tokenizer.vocab
        reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
        
        CONTAINER["model"] = model
        CONTAINER["tokenizer"] = tokenizer
        CONTAINER["reverse_voc"] = reverse_voc
        CONTAINER["model_name"] = model_name
        CONTAINER["is_maxsim"] = is_maxsim
        CONTAINER["voc"] = voc
    else:
        print("model already loaded")


def init_model(phrase_splade_model_name, normal_model_name):
    init_model_helper(phrase_splade_model_name, PHRASE_SPLADE)
    init_model_helper(normal_model_name, SPLADE)


def get_representation_helper(batch, CONTAINER, store_documents_in_raw = False):
    text_batch = [f"{line['title']} | {line['text']}" for line in batch]
    with torch.no_grad():
        batch_doc_rep = CONTAINER["model"].encode(CONTAINER["tokenizer"](text_batch, return_tensors="pt", truncation = True, padding = True, max_length = MAX_LENGTH).to(DEVICE), is_q = False)

    res = []
    for i in range(batch_doc_rep.size(0)):
        doc_rep = batch_doc_rep[i]

        try:
            # get the number of non-zero dimensions in the rep:
            col = torch.nonzero(doc_rep).squeeze().cpu().tolist()

            # now let's inspect the bow representation:
            weights = doc_rep[col].cpu().tolist()
            d = {k: int(v * 100) for k, v in zip(col, weights)}

            d = {CONTAINER["reverse_voc"][k]: v for k, v in d.items()}
        except Exception as e:
            print("An error occurred", traceback.format_exc())
            d = {}

        to_append = deepcopy(batch[i])
        to_append["vector"] = d
        if "id" not in to_append and "_id" in to_append:
            to_append["id"] = to_append["_id"]
        
        if not store_documents_in_raw:
            del to_append["title"]
            del to_append["text"]

        res.append(to_append)

    return res

def merge_representation(phrase_splade_representation, normal_splade_representation):
    assert len(phrase_splade_representation) == len(normal_splade_representation)
    res = deepcopy(phrase_splade_representation)

    for i in range(len(res)):
        phrase_splade_rep_i = phrase_splade_representation[i]["vector"]
        normal_splade_rep_i = normal_splade_representation[i]["vector"]
        
        to_update_from_phrase_splade_rep_i = {k:v for k,v in phrase_splade_rep_i.items() if k not in SPLADE["voc"]}
        merged_rep_i = normal_splade_rep_i | to_update_from_phrase_splade_rep_i

        res[i]["vector"] = merged_rep_i

    return res

def get_representation(batch, store_documents_in_raw = False):
    phrase_splade_representation = get_representation_helper(batch, PHRASE_SPLADE, store_documents_in_raw=store_documents_in_raw)
    normal_splade_representation = get_representation_helper(batch, SPLADE, store_documents_in_raw=store_documents_in_raw)


    representation = merge_representation(
        phrase_splade_representation=phrase_splade_representation,
        normal_splade_representation=normal_splade_representation
    )

    return representation


def prepare_data(corpus, outfile, batch_size = 100, is_q = False, store_documents_in_raw = False, chunk_idx = None):
    with open(outfile, "w") as f:
        desc = f"Processing chunk #{chunk_idx}" if chunk_idx is not None else ""
        for i in tqdm(range(0, len(corpus), batch_size), desc = desc):
            batch = corpus[i:i+batch_size]

            representations = get_representation(batch, 
                                                 store_documents_in_raw = store_documents_in_raw)

            for rep in representations:
                json_string = json.dumps(rep)
                f.write(json_string + "\n")


def do_indexing(outfolder_dataset, index_folder, remove_collections_folder = False):
    command = f"""python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input {outfolder_dataset} \
  --index {index_folder} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --impact --pretokenized \
  --storePositions --storeDocvectors --storeRaw"""
    
    os.system(command)

    if remove_collections_folder:
        remove_folder(outfolder_dataset)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, required=True)
    parser.add_argument("--phrase_splade_model_name", type = str, default = "phrase_splade")
    parser.add_argument("--normal_splade_model_name", type = str, default = "eru_kg")
    parser.add_argument("--work_dir", type = str, default="../../")
    parser.add_argument("--outfolder", type = str, required=True)
    parser.add_argument("--batch_size", type = int, default = 100)
    parser.add_argument("--is_q", type = bool, default = False)
    parser.add_argument("--chunking_size", type = int, default = 10000)
    parser.add_argument("--remove_collections_folder", type = bool, default = False)
    parser.add_argument("--store_documents_in_raw", type = bool, default = False)

    args = parser.parse_args()

    dataset_name = args.dataset
    phrase_splade_model_name = args.phrase_splade_model_name
    normal_splade_model_name = args.normal_splade_model_name
    work_dir = args.work_dir
    outfolder = args.outfolder
    batch_size = args.batch_size
    is_q = args.is_q
    chunking_size = args.chunking_size
    remove_collections_folder = args.remove_collections_folder
    store_documents_in_raw = args.store_documents_in_raw

    dataset_name_2_relative_path = {
        "scifact": "data/beir/scifact",
        "scidocs": "data/beir/scidocs",
        "nfcorpus": "data/beir/nfcorpus",
        "arguana": "data/beir/arguana",
        "fiqa": "data/beir/fiqa",
        "msmarco": "data/msmarco/msmarco"
    }

    init_model(phrase_splade_model_name=phrase_splade_model_name, normal_model_name=normal_splade_model_name)

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

    outfolder_dataset = os.path.join(outfolder, "collections", f"{dataset_name}__{phrase_splade_model_name}+{normal_splade_model_name}")
    maybe_create_folder(outfolder_dataset)
    # chunking corpus
    for chunk_idx, i in enumerate(range(0, len(corpus), chunking_size)):
        chunk = corpus[i:i+chunking_size]
        outfile = os.path.join(outfolder_dataset, f"chunk{chunk_idx}.jsonl")

        prepare_data(
            corpus = chunk,
            outfile = outfile,
            batch_size=batch_size,
            is_q=is_q,
            store_documents_in_raw=store_documents_in_raw,
            chunk_idx=chunk_idx
        )


    # do indexing
    index_folder = os.path.join(outfolder, "indexes", f"{dataset_name}__{phrase_splade_model_name}+{normal_splade_model_name}")
    remove_folder(index_folder)
    do_indexing(
        outfolder_dataset = outfolder_dataset,
        index_folder = index_folder,
        remove_collections_folder = remove_collections_folder
    )


if __name__ == "__main__":
    main()