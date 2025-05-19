# CUDA_VISIBLE_DEVICES=1 python inference.py --input_file /home/lamdo/splade/data/cfscube/cfscube/corpus.jsonl --model_name phrase_splade_39 --output_file test_gitig_.json
# CUDA_VISIBLE_DEVICES=2 python inference.py --model_name phrase_splade_39 --output_file test_gitig_.json
import torch, sys, json, argparse, os, time, traceback, random
sys.path.append("../../")
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from pyserini_evaluation.indexing.model_name_2_info import model_name_2_path, model_name_2_model_class, model_name_2_is_maxsim

from tqdm import tqdm
from memory_profiler import profile 
from copy import deepcopy
from argparse import ArgumentParser

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
        for param in model.parameters():
            param.requires_grad = False
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
            batch_doc_rep, batch_doc_token_indices, _ = \
                SPLADE["model"].encode(
                    batch_tokens, 
                    is_q = is_q
                )

        else:
            batch_doc_rep = SPLADE["model"].encode(batch_tokens, is_q = is_q)
            batch_doc_token_indices = None
    
    assert len(batch) == batch_doc_rep.size(0)

    # res = []
    for i in range(batch_doc_rep.size(0)):
        doc_rep = batch_doc_rep[i].detach().cpu()
        doc_token_indices = batch_doc_token_indices[i].detach().cpu() if batch_doc_token_indices is not None else None

        try:
            # get the number of non-zero dimensions in the rep:
            col = torch.nonzero(doc_rep).squeeze().tolist()

            weights = doc_rep[col].tolist()
            _indices = doc_token_indices[col].tolist() if doc_token_indices is not None else None
            d = {k: int(v * 100) for k, v in zip(col, weights)}
            # d = {SPLADE["reverse_voc"][k]: v for k, v in d.items()}
            d_indices = {SPLADE["reverse_voc"][k]: v for k, v in zip(col, _indices)} if _indices is not None else None

            del col, weights, _indices
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
        
        if not store_documents_in_raw:
            del to_append["title"]
            del to_append["text"]

        yield to_append

    #     res.append(to_append)

    # return res


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type = str, default = None)
    parser.add_argument("--num_docs_to_test", type = int, default = 100000)
    parser.add_argument("--batch_size", type = int, default = 100)
    parser.add_argument("--model_name", type = str, required=True)
    parser.add_argument("--output_file", type = str, required=True)

    args = parser.parse_args()

    input_file = args.input_file
    num_docs_to_test = args.num_docs_to_test
    batch_size = args.batch_size
    output_file = args.output_file
    model_name = args.model_name

    init_model(model_name)

    if input_file:
        data = []
        with open(input_file) as f:
            for line in f:
                jline = json.loads(line)

                data.append(jline)
    else:
        from datasets import load_dataset
        ds = load_dataset("sentence-transformers/s2orc", "title-abstract-pair")
        data = []
        for i, line in enumerate(ds["train"]):
            if i == num_docs_to_test: break
            to_append = {"title": line["title"], "text": line["abstract"]}
            data.append(to_append)

    data = random.sample(data, k = num_docs_to_test)

    with open(output_file, "w") as f:
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            batch_rep = get_representation(batch, is_q=False, store_documents_in_raw=True)

            for rep in batch_rep:
                json.dump(rep, f)
                f.write("\n")

if __name__ == "__main__":
    main()