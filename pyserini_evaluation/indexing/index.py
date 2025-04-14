import torch, sys, json, argparse, os, time, traceback
sys.path.append("../../")
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from pyserini_evaluation.indexing.model_name_2_info import model_name_2_path, model_name_2_model_class, model_name_2_is_maxsim
from pyserini_evaluation.indexing.utils import maybe_create_folder, remove_folder, torch_csr_to_scipy_csr
# from scipy.sparse import csr_matrix, vstack, save_npz
from pyserini_evaluation.indexing.utils import merge_dicts

from tqdm import tqdm

SPLADE = {
    "model": None,
    "tokenizer": None,
    "reverse_voc": None,
    "model_name": None,
    "is_maxsim": None
}

MAX_LENGTH = 256

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def init_model(model_name):
    if SPLADE["model_name"] is None or SPLADE["model_name"] != model_name:
        print(f"Loading {model_name} for the first time. This will be done only once for {model_name}")
        model_type_or_dir = model_name_2_path.get(model_name)
        model_class = model_name_2_model_class.get(model_name)
        is_maxsim = model_name_2_is_maxsim.get(model_name)

        if model_type_or_dir is None and model_class is None:
            raise NotImplementedError



        model = model_class(model_type_or_dir, agg="max").to(DEVICE)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        reverse_voc = {v: k.replace(" ", "-") for k, v in tokenizer.vocab.items()}
        
        SPLADE["model"] = model
        SPLADE["tokenizer"] = tokenizer
        SPLADE["reverse_voc"] = reverse_voc
        SPLADE["model_name"] = model_name
        SPLADE["is_maxsim"] = is_maxsim
    else:
        print("model already loaded")




def get_representation(batch, is_q, store_documents_in_raw = False):
    text_batch = [f"{line['title']} | {line['text']}" for line in batch]
    batch_tokens = SPLADE["tokenizer"](text_batch, return_tensors="pt", truncation = True, padding = True, max_length = MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        if SPLADE["is_maxsim"]:
            batch_doc_rep, batch_doc_token_indices, batch_doc_pad_len = \
                SPLADE["model"].encode(
                    batch_tokens, 
                    is_q = is_q
                )
            
            bs = batch_doc_rep.shape[0]
            out_dim = batch_doc_rep.shape[1]
            # batch_doc_rep_full = torch.zeros((bs, MAX_LENGTH, out_dim), dtype=batch_doc_rep.dtype).to(batch_doc_rep.device)
            # batch_doc_rep_full.scatter_(1, batch_doc_token_indices.unsqueeze(1), batch_doc_rep.unsqueeze(1))


            # batch_doc_rep_full = [item.to_sparse_csr().cpu() for item in batch_doc_rep_full]
            # batch_doc_rep_full = [torch_csr_to_scipy_csr(item) for item in batch_doc_rep_full]

            batch_doc_rep_full = None

        else:
            batch_doc_rep = SPLADE["model"].encode(batch_tokens, is_q = is_q)
            batch_doc_token_indices = None
            batch_doc_pad_len = None
            batch_doc_rep_full = None
    
    assert len(batch) == batch_doc_rep.size(0)

    res = []
    for i in range(batch_doc_rep.size(0)):
        doc_rep = batch_doc_rep[i]
        doc_token_indices = batch_doc_token_indices[i] if batch_doc_token_indices is not None else None

        input_ids = batch_tokens["input_ids"][i]
        attention_mask = batch_tokens["attention_mask"][i]

        try:
            # get the number of non-zero dimensions in the rep:
            col = torch.nonzero(doc_rep).squeeze().cpu().tolist()

            weights = doc_rep[col].cpu().tolist()
            _indices = doc_token_indices[col].cpu().tolist() if doc_token_indices is not None else None
            d = {k: int(v * 100) for k, v in zip(col, weights)}
            d = {SPLADE["reverse_voc"][k]: v for k, v in d.items()}
            d_indices = {SPLADE["reverse_voc"][k]: v for k, v in zip(col, _indices)} if _indices is not None else None
        except Exception as e:
            print("An error occurred", traceback.format_exc())
            d = {}
            d_indices = {}

        to_append = batch[i]
        to_append["vector"] = d
        if "id" not in to_append and "_id" in to_append:
            to_append["id"] = to_append["_id"]
        
        if d_indices is not None:
            to_append["token_indices"] = d_indices
            to_append["pad_len"] = batch_doc_pad_len # may not be necessary
        
        if not store_documents_in_raw:
            del to_append["title"]
            del to_append["text"]

        res.append(to_append)

    return res, batch_doc_rep_full



def prepare_data(corpus, model_name, outfile, batch_size = 100, is_q = False, store_documents_in_raw = False, chunk_idx = None):
    init_model(model_name=model_name)

    with open(outfile, "w") as f:
        corpus_full_representations = []
        desc = f"Processing chunk #{chunk_idx}" if chunk_idx is not None else ""
        for i in tqdm(range(0, len(corpus), batch_size), desc = desc):
            batch = corpus[i:i+batch_size]

            representations, full_representations = get_representation(batch, is_q = is_q, 
                                                 store_documents_in_raw = store_documents_in_raw)

            if full_representations is not None:
                corpus_full_representations.extend(full_representations)

            for rep in representations:
                json_string = json.dumps(rep)
                f.write(json_string + "\n")

        # corpus_full_representations = vstack(corpus_full_representations)

    return corpus_full_representations

                
def do_indexing(outfolder_dataset, index_folder, docids = None, corpus_full_representations = None, remove_collections_folder = False):
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

    if docids is not None and corpus_full_representations is not None:
        maybe_create_folder(os.path.join(index_folder, "full_representations/"))
        for docid, full_rep in tqdm(zip(docids, corpus_full_representations)):
            save_npz(file = os.path.join(index_folder, "full_representations", f"{docid}.npz"), matrix = full_rep)

        with open(os.path.join(index_folder, "full_representations_docids.json"), "w") as f:
            json.dump(docids, f)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, required=True)
    parser.add_argument("--model_name", type = str, default = "splade_maxsim")
    parser.add_argument("--work_dir", type = str, default="../../")
    parser.add_argument("--outfolder", type = str, required=True)
    parser.add_argument("--batch_size", type = int, default = 100)
    parser.add_argument("--is_q", type = bool, default = False)
    parser.add_argument("--chunking_size", type = int, default = 10000)
    parser.add_argument("--remove_collections_folder", type = bool, default = False)
    parser.add_argument("--store_documents_in_raw", type = bool, default = False)

    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model_name
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
        "trec-covid": "data/beir/trec-covid",
        "msmarco": "data/msmarco/msmarco",
        "doris_mae": "data/doris_mae/doris_mae",
        "cfscube": "data/cfscube/cfscube",
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
    
    print("Corpus length", len(corpus))

    outfolder_dataset = os.path.join(outfolder, "collections", f"{dataset_name}__{model_name}")
    maybe_create_folder(outfolder_dataset)
    corpus_full_representation = []
    # chunking corpus
    for chunk_idx, i in enumerate(range(0, len(corpus), chunking_size)):
        chunk = corpus[i:i+chunking_size]
        outfile = os.path.join(outfolder_dataset, f"chunk{chunk_idx}.jsonl")

        chunk_full_representations = prepare_data(
            corpus = chunk,
            model_name = model_name,
            outfile = outfile,
            batch_size=batch_size,
            is_q=is_q,
            store_documents_in_raw=store_documents_in_raw,
            chunk_idx=chunk_idx
        )

        corpus_full_representation.extend(chunk_full_representations)

    corpus_full_representation = corpus_full_representation if corpus_full_representation else None


    # do indexing
    index_folder = os.path.join(outfolder, "indexes", f"{dataset_name}__{model_name}")
    remove_folder(index_folder)
    do_indexing(
        outfolder_dataset = outfolder_dataset,
        index_folder = index_folder,
        remove_collections_folder = remove_collections_folder,
        docids = [line["id"] if "id" in line else line["_id"] for line in corpus],
        corpus_full_representations=corpus_full_representation
    )


if __name__ == "__main__":
    main()