import torch, sys, json, argparse, os, time, traceback
sys.path.append("../../")
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from pyserini_evaluation.indexing.model_name_2_info import model_name_2_path, model_name_2_model_class, model_name_2_is_maxsim, model_name_2_original_bert_vocab_size
from pyserini_evaluation.indexing.bm25_model import BM25
from tqdm import tqdm
from memory_profiler import profile 
from copy import deepcopy

SPLADE = {
    "model": None,
    "tokenizer": None,
    "reverse_voc": None,
    "model_name": None,
    "is_maxsim": None
}

BM25_MODEL = {
    "model": None
}

MAX_LENGTH = 256

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def init_model(model_name):
    if SPLADE["model_name"] is None or SPLADE["model_name"] != model_name:
        print(f"Loading {model_name} for the first time. This will be done only once for {model_name}")
        model_type_or_dir = model_name_2_path.get(model_name)
        model_class = model_name_2_model_class.get(model_name)
        original_bert_vocab_size = model_name_2_original_bert_vocab_size.get(model_name, 30522)
        is_maxsim = model_name_2_is_maxsim.get(model_name)

        if model_type_or_dir is None and model_class is None:
            raise NotImplementedError


        try:
            model = model_class(model_type_or_dir, agg="max", original_bert_vocab_size = original_bert_vocab_size).to(DEVICE)
        except Exception:
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

def splade_pooling(out, tokens, class_name = "Splade"):
    if class_name not in ["Spladev3", "Spladev4", "Spladev5"]:
        # only Spladev3 has this special pooling function
        out_tokens = out[..., :30522] # shape (bs, pad_len, original_bert_vocab_size)
        out_phrases = out[..., 30522:] # shape (bs, pad_len, vocab_size - original_bert_vocab_size)
        values_tokens, _ = torch.max(torch.log(1 + torch.relu(out_tokens)) * tokens["attention_mask"].unsqueeze(-1), dim=1) # shape (bs, original_bert_vocab_size)
        values_phrases = torch.sum(torch.log(1 + torch.relu(out_phrases)) * tokens["attention_mask"].unsqueeze(-1), dim=1) # shape (bs, vocab_size - original_bert_vocab_size)

        values = torch.cat([values_tokens, values_phrases], dim = -1)
        return values
    
    else:
        # if not Spladev3, this apply max pooling like this
        values, _ = torch.max(torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1), dim=1)
        return values

def encode_batch_mask_special_tokens(tokens, model, puncid, is_q = False):
    out = model.encode_(tokens, is_q)["logits"]  # shape (bs, pad_len, voc_size)
    out = torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1)

    mask = ~torch.isin(tokens["input_ids"], puncid)
    out = out * mask.unsqueeze(-1)

    model_class_name = model.__class__.__name__
    return splade_pooling(out, tokens, class_name = model_class_name)


def get_representation(batch, is_q, store_documents_in_raw = False, add_bm25 = False, mask_special_tokens = False):
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
            if not mask_special_tokens:
                batch_doc_rep = SPLADE["model"].encode(batch_tokens, is_q = is_q)
            else:
                batch_doc_rep = encode_batch_mask_special_tokens(batch_tokens, SPLADE["model"], SPLADE["puncid"], is_q = is_q)
            batch_doc_token_indices = None
    
    assert len(batch) == batch_doc_rep.size(0)

    # res = []
    for i in range(batch_doc_rep.size(0)):
        doc_rep = batch_doc_rep[i].detach().cpu()
        doc_token_indices = batch_doc_token_indices[i].detach().cpu() if batch_doc_token_indices is not None else None

        if add_bm25:
            bm25_rep = BM25_MODEL["model"].get_term_scores(text_batch[i])
        else: bm25_rep = {}

        try:
            # get the number of non-zero dimensions in the rep:
            col = torch.nonzero(doc_rep).squeeze().tolist()

            weights = doc_rep[col].tolist()
            _indices = doc_token_indices[col].tolist() if doc_token_indices is not None else None
            d = {k: int(v * 100) for k, v in zip(col, weights)}
            d = {SPLADE["reverse_voc"][k]: v for k, v in d.items()}
            d = {k: v + 0.1 * bm25_rep.get(k, 0) for k,v in d.items()}
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


def init_bm25_model(splade_model_dir, corpus, save_path):
    model = BM25(corpus = corpus, splade_model_dir=splade_model_dir)

    BM25_MODEL["model"] = model

    model.save_model(save_path)



# @profile
def prepare_data(documents, batch_size = 100, is_q = False, store_documents_in_raw = False, chunk_idx = None, add_bm25 = False,
                 mask_special_tokens = False):

    # all_representations = []
    # with open(outfile, "w") as f:
    desc = f"Processing chunk #{chunk_idx}" if chunk_idx is not None else ""
    for i in tqdm(range(0, len(documents), batch_size), desc = desc):

        representations = get_representation(
            documents[i:i+batch_size], 
            is_q = is_q, 
            store_documents_in_raw = store_documents_in_raw,
            add_bm25 = add_bm25,
            mask_special_tokens = mask_special_tokens
        )

        for rep in representations:
            yield rep

    #     all_representations.extend(deepcopy(representations))

    #     del representations

    # return all_representations

            # for rep in representations:
            #     json_string = json.dumps(rep)
            #     f.write(json_string + "\n")
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
            


# @profile
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, required=True)
    parser.add_argument("--model_name", type = str, default = "splade_maxsim")
    parser.add_argument("--work_dir", type = str, default="../../")
    parser.add_argument("--outfolder", type = str, required=True)
    parser.add_argument("--batch_size", type = int, default = 100)
    parser.add_argument("--is_q", type = bool, default = False)
    # parser.add_argument("--chunking_size", type = int, default = 10000)
    parser.add_argument("--remove_collections_folder", type = str2bool, default = False)
    parser.add_argument("--store_documents_in_raw", type = str2bool, default = False)
    parser.add_argument("--num_chunks", type = int, default = 4)
    parser.add_argument("--chunk_idx", type = int, required = True)
    parser.add_argument("--add_bm25", type = str2bool, default = False)
    parser.add_argument("--mask_special_tokens", type = str2bool, default = False)

    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model_name
    work_dir = args.work_dir
    outfolder = args.outfolder
    batch_size = args.batch_size
    is_q = args.is_q
    # chunking_size = args.chunking_size
    remove_collections_folder = args.remove_collections_folder
    store_documents_in_raw = args.store_documents_in_raw
    num_chunks = args.num_chunks
    chunk_idx = args.chunk_idx
    add_bm25 = args.add_bm25
    mask_special_tokens = args.mask_special_tokens

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
    }

    init_model(model_name=model_name)

    if mask_special_tokens:
        import string
        SPLADE["puncid"] = torch.tensor([SPLADE["tokenizer"].vocab["[SEP]"], SPLADE["tokenizer"].vocab["[CLS]"]]).to(DEVICE)

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

    if add_bm25:
        bm25_model_path = os.path.join(outfolder, "bm25_models", f"{dataset_name}__{model_name}.pkl")
        init_bm25_model(splade_model_dir=model_name_2_path[model_name], 
                        corpus = [f"{line['title']} | {line['text']}" for line in corpus],
                        save_path=bm25_model_path)
    
    print("Corpus length", len(corpus))

    outfolder_dataset = os.path.join(outfolder, "collections", f"{dataset_name}__{model_name}")

    chunk_indices = np.array_split(np.arange(len(corpus)), num_chunks)[chunk_idx]
    chunk = [corpus[index] for index in chunk_indices]

    outfile = os.path.join(outfolder_dataset, f"chunk{chunk_idx}.jsonl")

    to_write = prepare_data(
        documents = chunk,
        # outfile = outfile,
        batch_size=batch_size,
        is_q=is_q,
        store_documents_in_raw=store_documents_in_raw,
        chunk_idx=chunk_idx,
        add_bm25 = add_bm25,
        mask_special_tokens=mask_special_tokens
    )

    with open(outfile, "w") as f:
        for rep in to_write:
            json.dump(rep, f)
            f.write("\n")


if __name__ == "__main__":
    main()