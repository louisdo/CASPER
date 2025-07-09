import json, sys, torch, argparse, os, pytrec_eval, logging
sys.path.append("../")
import pandas as pd
import numpy as np
from pyserini.search.lucene import LuceneImpactSearcher
from pyserini.pyclass import JInt, JHashMap, JArrayList
from pyserini_evaluation.indexing.index import init_model, SPLADE, init_bm25_model, BM25_MODEL, BM25
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import Counter
from scipy.sparse import csr_matrix, vstack, load_npz

logger = logging.getLogger(__name__)

SEARCHER = {
    "searcher": None,
    "index_path": None,
    "doc2fullrep": None,
    "splade_vocab": None
}

MAX_LENGTH = 256
BERT_ORIGINAL_VOCAB_SIZE = 30522

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_bm25_model(model_path):
    if not BM25_MODEL["model"]:
        model = BM25.load_model(model_path)
        BM25_MODEL["model"] = model


def create_jquery(encoded_query, searcher, fields = {}):
    # jfields = JHashMap()
    # for (field, boost) in fields.items():
    #     jfields.put(field, JFloat(boost))

    jquery = JHashMap()
    for (token, weight) in encoded_query.items():
        # if token in searcher.idf and searcher.idf[token] > searcher.min_idf:
        jquery.put(token, JInt(weight))

    return jquery


def init_searcher(index_path):
    if SEARCHER["searcher"] is None or SEARCHER["index_path"] != index_path:
        searcher = LuceneImpactSearcher(index_path, query_encoder=None)
        SEARCHER["searcher"] = searcher
        SEARCHER["index_path"] = index_path

        if os.path.exists(os.path.join(index_path, "full_representations")):
            with open(os.path.join(index_path, "full_representations_docids.json")) as f:
                docids = json.load(f)

            doc2fullrep = {}
            for i in tqdm(range(len(docids)), desc = "Processing documents' full representation"):
                docid = docids[i]

                full_rep = load_npz(os.path.join(index_path, "full_representations", f"{docid}.npz"))

                doc2fullrep[docid] = full_rep

            SEARCHER["doc2fullrep"] = doc2fullrep
        else:
            SEARCHER["doc2fullrep"] = None
    else:
        print("searcher already initialized")



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



def encode_queries(queries, ids = None, add_onehot = False, weight_tokens = 1.0, weight_phrases = 1.0, add_bm25 = False, mask_special_tokens = False):
    with torch.no_grad():
        batch_tokens = SPLADE["tokenizer"](queries, return_tensors="pt", truncation = True, padding = True, max_length=256).to(DEVICE)
        if SPLADE["is_maxsim"]:
            batch_doc_rep, batch_doc_token_indices, batch_doc_pad_len = \
                SPLADE["model"].encode(
                    batch_tokens, 
                    is_q = True
                )
            
            bs = batch_doc_rep.shape[0]
            out_dim = batch_doc_rep.shape[1]
            batch_doc_rep_full = torch.zeros((bs, MAX_LENGTH, out_dim), dtype=batch_doc_rep.dtype).to(batch_doc_rep.device)
            batch_doc_rep_full.scatter_(1, batch_doc_token_indices.unsqueeze(1), batch_doc_rep.unsqueeze(1))

            # batch_doc_rep_full = [item.to_sparse_csr().cpu() for item in batch_doc_rep_full]
            # batch_doc_rep_full = [torch_csr_to_scipy_csr(item) for item in batch_doc_rep_full]
        else:
            if not mask_special_tokens:
                batch_doc_rep = SPLADE["model"].encode(batch_tokens, is_q = True)
            else: batch_doc_rep = encode_batch_mask_special_tokens(batch_tokens, SPLADE["model"], SPLADE["puncid"], is_q = True)
            batch_doc_token_indices = None
            batch_doc_pad_len = None
            batch_doc_rep_full = None
    
    assert len(queries) == batch_doc_rep.size(0)

    res = []
    for i in range(batch_doc_rep.size(0)):
        doc_rep = batch_doc_rep[i]
        doc_token_indices = batch_doc_token_indices[i] if batch_doc_token_indices is not None else None

        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        _indices = doc_token_indices[col].cpu().tolist() if doc_token_indices is not None else None
        d = {k: int(v * 100) for k, v in zip(col, weights)}

        if add_onehot:
            input_ids = batch_tokens["input_ids"][i]
            attention_mask = batch_tokens["attention_mask"][i]
            doc_rep_onehot_full = torch.nn.functional.one_hot(input_ids, num_classes = len(SPLADE["reverse_voc"])) * attention_mask.unsqueeze(-1)
            doc_rep_onehot, doc_token_indices_onehot = torch.max(doc_rep_onehot_full, dim = 0)

            col_onehot = torch.nonzero(doc_rep_onehot).squeeze().cpu().tolist()
            weights_onehot = doc_rep_onehot[col_onehot].cpu().tolist()
            _indices_onehot = doc_token_indices_onehot[col_onehot].cpu().tolist() if doc_token_indices_onehot is not None else None
            d_onehot = {k: int(v * 100) for k, v in zip(col_onehot, weights_onehot)}
        else: 
            col_onehot = None
            weights_onehot = None
            _indices_onehot = None
            d_onehot = {}

        if add_bm25:
            bm25_rep = BM25_MODEL["model"].get_term_scores(queries[i])
            bm25_rep = {SPLADE["voc"][k]: v for k,v in bm25_rep.items()}
            print(bm25_rep)
            d = {k: v + 0.1 * bm25_rep.get(k, 0) for k, v in d.items()}
        d_phrase_only = {SPLADE["reverse_voc"][k]: v for k, v in d.items() if k >= BERT_ORIGINAL_VOCAB_SIZE}
        d = {SPLADE["reverse_voc"][k]: (v * (weight_tokens if k < BERT_ORIGINAL_VOCAB_SIZE else weight_phrases)) for k, v in d.items()}
        # d = {SPLADE["reverse_voc"][k]: (v * (1 if k < 30522 else 0.5)) for k, v in d.items() if v >= 5}
        # d = merge_dicts(d, d_added)
        d_onehot = {SPLADE["reverse_voc"][k]: v for k, v in d_onehot.items()}
        d_indices = {SPLADE["reverse_voc"][k]: v for k, v in zip(col, _indices)} if _indices is not None else None
        d_indices_onehot = {SPLADE["reverse_voc"][k]: v for k, v in zip(col_onehot, _indices_onehot)} if _indices_onehot is not None else None

        to_append = {"query_id": ids[i] if ids else None}
        to_append["vector"] = d
        to_append["vector_phrase_only"] = d_phrase_only
        to_append["vector_onehot"] = d_onehot

        if d_indices is not None:
            to_append["token_indices"] = d_indices
            to_append["token_indices_onehot"] = d_indices_onehot
            to_append["pad_len"] = batch_doc_pad_len # may not be necessary

        res.append(to_append)

    return res


def convert_to_pytrec_eval_format(queries, all_search_results, type = "relevance"):
    """
    queries: [q1, q2, ...]
    all_search_results: [[{'docid': '22711954', 'score': 4.043900012969971}, ...]]
    """

    assert len(queries) == len(all_search_results)

    score_converter = {
        "relevance": lambda x: int(x),
        "prediction": lambda x:float(x)
    }

    res = {}
    for query, search_results in zip(queries, all_search_results):
        if query not in res:
            res[query] = {}
        

        for sr in search_results:
            docid = str(sr["docid"])
            score = score_converter[type](sr["score"])

            res[query][docid] = score

    return res


def mrr(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> tuple[dict[str, float]]:
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]

    for query_id in top_hits:
        if query_id in qrels:
            query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        else: query_relevant_docs = set([])
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"] / len(qrels), 5)
        logging.info("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    return MRR


def evaluate(qrels: Dict[str, Dict[str, int]], 
                results: Dict[str, Dict[str, float]], 
                k_values: List[int],
                ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    
    if ignore_identical_ids:
        logger.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
        popped = []
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)
                    popped.append(pid)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
    
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
    
    for eval in [ndcg, _map, recall, precision]:
        logger.info("\n")
        for k in eval.keys():
            logger.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision


def read_qrels(qrel_path):
    _qrels = pd.read_csv(qrel_path, sep='\t').to_dict("records")
    
    metadata = {}
    for line in _qrels:
        query_id = str(line["query-id"])
        doc_id = str(line["corpus-id"])
        score = line["score"]

        if query_id not in metadata:
            metadata[query_id] = []

        metadata[query_id].append({
            "docid": doc_id,
            "score": score
        })

    queries_ids = list(metadata.keys())
    queries_all_labels = [metadata[k] for k in queries_ids]

    qrels = convert_to_pytrec_eval_format(queries = queries_ids, all_search_results=queries_all_labels)

    return qrels


def do_reranking(full_encoded_query_info, hits):
    # do reranking with maxsim
    query_vector = full_encoded_query_info["vector"] if "vector_original" not in full_encoded_query_info else full_encoded_query_info["vector_original"]
    query_token_indices = full_encoded_query_info["token_indices"]

    unique_query_token_indices = set(query_token_indices.values())
    query_vector_by_token_indices = {}
    for token_index in unique_query_token_indices:
        temp = Counter({k:v for k,v in query_vector.items() if query_token_indices[k] == token_index})
        query_vector_by_token_indices[token_index] = temp

    formatted_results = []
    for hit in hits:
        docid = hit.docid
        original_score = hit.score
        raw = json.loads(hit.lucene_document.get("raw"))
        doc_vector = raw["vector"]
        doc_token_indices = raw["token_indices"]

        unique_doc_token_indices = set(doc_token_indices.values())
        doc_vector_by_token_indices = {}
        for token_index in unique_doc_token_indices:
            temp = Counter({k:v for k,v in doc_vector.items() if doc_token_indices[k] == token_index})
            doc_vector_by_token_indices[token_index] = temp

        # recompute the score
        recomputed_score = 0
        for query_token_index in unique_query_token_indices:
            query_token_index_vector = query_vector_by_token_indices[query_token_index]
            query_token_index_scores = []
            for doc_token_index in unique_doc_token_indices:
                doc_token_index_vector = doc_vector_by_token_indices[doc_token_index]
                
                score = sum([query_token_index_vector[tok] * doc_token_index_vector[tok] for tok in query_token_index_vector])
                query_token_index_scores.append(score)

            recomputed_score += max(query_token_index_scores)


        formatted_results.append({
            "docid": docid,
            "score": recomputed_score
        })

    formatted_results = list(sorted(formatted_results, key = lambda x: -x["score"]))

    return formatted_results


def create_sparse_matrix(vector, token_indices):
    row_col_val = [[v, SEARCHER["splade_vocab"][k], vector[k]] for k,v in token_indices.items()]
    row = [item[0] for item in row_col_val]
    col = [item[1] for item in row_col_val]
    val = [item[2] for item in row_col_val]

    sparse_matrix = csr_matrix((val, (row, col)), shape=(MAX_LENGTH, SEARCHER["vocab_length"]))
    return sparse_matrix


def do_reranking_v2(full_encoded_query_info, hits, add_onehot):
    query_vector = full_encoded_query_info["vector"] if "vector_original" not in full_encoded_query_info else full_encoded_query_info["vector_original"]
    query_token_indices = full_encoded_query_info["token_indices"]
    query_full_rep = create_sparse_matrix(query_vector, query_token_indices)

    if add_onehot:
        query_vector_onehot = full_encoded_query_info["vector_onehot"]
        query_token_indices_onehot = full_encoded_query_info["token_indices_onehot"]
        query_full_rep_onehot = create_sparse_matrix(query_vector_onehot, query_token_indices_onehot)

        assert query_full_rep.shape == query_full_rep_onehot.shape

        query_full_rep = query_full_rep + query_full_rep_onehot

    candidate_full_rep = []
    candidate_docids = []
    for hit in hits:
        candidate_docids.append(hit.docid)
        raw = json.loads(hit.lucene_document.get("raw"))
        doc_vector = raw["vector"]
        doc_token_indices = raw["token_indices"]
        doc_full_rep = create_sparse_matrix(doc_vector, doc_token_indices)

        if add_onehot:
            doc_vector_onehot = raw["vector_onehot"]
            doc_token_indices_onehot = raw["token_indices_onehot"]
            
            doc_full_rep_onehot = create_sparse_matrix(doc_vector_onehot, doc_token_indices_onehot)

            assert doc_full_rep.shape == doc_full_rep_onehot.shape
            doc_full_rep = doc_full_rep + doc_full_rep_onehot

        candidate_full_rep.append(doc_full_rep)
    candidate_full_rep = vstack(candidate_full_rep).transpose() # [dim, num docs * MAX_LENGTH]

    max_scores = (query_full_rep @ candidate_full_rep).todense()

    scores = []
    for i in range(len(hits)):
        scores.append(max_scores[:, i * MAX_LENGTH: (i+1) * MAX_LENGTH].max(1).sum())

    
    scores, candidate_docids = list(zip(*sorted(list(zip(scores, candidate_docids)), key=lambda x: -x[0])))

    formatted_results = [{"docid": docid, "score": score} for score, docid in zip(scores, candidate_docids)]

    return formatted_results
 


def fast_reranking(full_encoded_query_info, hits):
    query_full_rep = full_encoded_query_info["full_rep"] # [query length, dim]
    
    candidate_docids = [hit.docid for hit in hits]
    candidate_full_rep = vstack([SEARCHER["doc2fullrep"][docid] for docid in candidate_docids]).transpose() # [dim, num docs * MAX_LENGTH]


    max_scores = (query_full_rep @ candidate_full_rep).todense()

    scores = []
    for i in range(len(candidate_docids)):
        scores.append(max_scores[:, i * MAX_LENGTH: (i+1) * MAX_LENGTH].max(1).sum())

    
    scores, candidate_docids = list(zip(*sorted(list(zip(scores, candidate_docids)), key=lambda x: -x[0])))

    formatted_results = [{"docid": docid, "score": score} for score, docid in zip(scores, candidate_docids)]

    return formatted_results

        


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_jarray_from_list(input_list):
    res = JArrayList()
    for item in input_list:
        res.add(item)

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splade_model_name", type = str, default = "splade_maxsim")
    parser.add_argument("--index_folder", type = str, required=True)
    parser.add_argument("--work_dir", type = str, default = "../")
    parser.add_argument("--dataset", type = str, required=True)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--save_metadata_for_debugging", type = str2bool, default = False)
    parser.add_argument("--add_onehot", type = str2bool, default = False)
    parser.add_argument("--weight_tokens", type = float, default = 1.0)
    parser.add_argument("--weight_phrases", type = float, default = 1.0)
    parser.add_argument("--threads", type = int, default = 8)
    parser.add_argument("--mode", type = str, choices = ["eval", "predict"])
    parser.add_argument("--num_chunks", type=int, default = 4)
    parser.add_argument("--chunk_idx", type = int, default = 0)
    parser.add_argument("--add_bm25", type = str2bool, default = False)
    parser.add_argument("--bm25_models_folder", type = str, default = None)
    parser.add_argument("--mask_special_tokens", type = str2bool, default = False)

    args = parser.parse_args()

    splade_model_name = args.splade_model_name
    index_folder = args.index_folder
    work_dir = args.work_dir
    dataset_name = args.dataset
    batch_size = args.batch_size
    save_metadata_for_debugging = args.save_metadata_for_debugging
    add_onehot = args.add_onehot
    weight_tokens = args.weight_tokens
    weight_phrases = args.weight_phrases
    threads = args.threads
    mode = args.mode
    num_chunks = args.num_chunks
    chunk_idx = args.chunk_idx
    add_bm25 = args.add_bm25
    bm25_models_folder = args.bm25_models_folder
    mask_special_tokens = args.mask_special_tokens

    if dataset_name in ["trec_dl_2019", "trec_dl_2020"]:
        index_path = os.path.join(index_folder, f"msmarco__{splade_model_name}")
    else:
        index_path = os.path.join(index_folder, f"{dataset_name}__{splade_model_name}")


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
        "trec_dl_2019": "data/msmarco/trec_dl_2019",
        "trec_dl_2020": "data/msmarco/trec_dl_2020",
        "acm_cr": "data/acm_cr/acm_cr",
        "litsearch": "data/litsearch/litsearch",
        "relish": "data/relish/relish",

        "cfscube_taxoindex":"data/cfscube/cfscube_taxoindex",
        "doris_mae_taxoindex": "data/doris_mae/doris_mae_taxoindex",
    }

    queries_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset_name],
        "queries.jsonl")
    
    qrel_path = os.path.join(
        work_dir,
        dataset_name_2_relative_path[dataset_name],
        "qrels/test.tsv" if dataset_name != "msmarco" else "qrels/dev.tsv"
    )
    qrels = read_qrels(qrel_path=qrel_path)


    # load queries
    with open(queries_path) as f:
        queries = [json.loads(line) for line in f]
        queries = [line for line in queries if line["_id"] in qrels]

    if mode == "predict":
        chunk_indices = np.array_split(np.arange(len(queries)), num_chunks)[chunk_idx]
        queries = [queries[i] for i in chunk_indices]
    

        # init splade model
        init_model(model_name = splade_model_name)
        SPLADE["voc"] = {k.replace(" ", "-"): v for k, v in SPLADE["tokenizer"].vocab.items()}

        if mask_special_tokens:
            import string
            SPLADE["puncid"] = torch.tensor([SPLADE["tokenizer"].vocab["[SEP]"], SPLADE["tokenizer"].vocab["[CLS]"]]).to(DEVICE)

        if add_bm25:
            bm25_model_filename = f"{dataset_name}__{splade_model_name}.pkl" if dataset_name not in ["trec_dl_2019", "trec_dl_2020"] else f"msmarco__{splade_model_name}.pkl"
            bm25_model_path = os.path.join(bm25_models_folder, bm25_model_filename)
            load_bm25_model(bm25_model_path)

        # init searcher
        init_searcher(index_path = index_path)

        splade_vocab = SPLADE["tokenizer"].vocab
        SEARCHER["splade_vocab"] = splade_vocab
        SEARCHER["vocab_length"] = len(splade_vocab)

        queries_texts = [line["text"] for line in queries]
        queries_ids = [line["_id"] for line in queries]

        # encode the queries
        encoded_queries = []
        for i in tqdm(range(0, len(queries_texts), batch_size)):
            batch_queries = queries_texts[i:i+batch_size]
            batch_queries_ids = queries_ids[i:i+batch_size]
            temp = encode_queries(queries = batch_queries, ids = batch_queries_ids, add_onehot=add_onehot,
                                weight_tokens=weight_tokens, weight_phrases = weight_phrases, add_bm25=add_bm25)
            encoded_queries.extend(temp)

        assert len(queries_ids) == len(encoded_queries)

        all_jqueries = []
        all_jqueries_phrase_only = []
        all_jqids = []
        for query_id, line in zip(queries_ids, encoded_queries):
            encoded_query = line["vector"]
            encoded_query = {item[0]: item[1] for item in Counter(encoded_query).most_common(1024)}

            encoded_query_phrase_only = line["vector_phrase_only"]
            encoded_query_phrase_only = {item[0]: item[1] for item in Counter(encoded_query_phrase_only).most_common(1024)}


            jquery = create_jquery(encoded_query=encoded_query, searcher = SEARCHER["searcher"])
            jquery_phrase_only = create_jquery(encoded_query = encoded_query_phrase_only, searcher = SEARCHER["searcher"])

            all_jqueries.append(jquery)
            all_jqueries_phrase_only.append(jquery_phrase_only)
            all_jqids.append(query_id)


        top_k = 5000 if not SPLADE["is_maxsim"] else 100
        top_k_phrase_only = 1000

        all_hits = {}
        all_hits_phrase_only = {}
        for i in tqdm(range(0, len(all_jqueries), batch_size), desc = "Running batch search"):
            batch_jqueries = create_jarray_from_list(all_jqueries[i:i+batch_size])
            batch_jqueries_phrase_only = create_jarray_from_list(all_jqueries_phrase_only[i:i+batch_size])
            batch_jqids = create_jarray_from_list(all_jqids[i:i+batch_size])

            to_update = SEARCHER["searcher"].object.batch_search(batch_jqueries, batch_jqids, top_k, threads)
            to_update = {r.getKey(): r.getValue() for r in to_update.entrySet().toArray()}
            all_hits.update(to_update)

            to_update_phrase_only = SEARCHER["searcher"].object.batch_search(batch_jqueries_phrase_only, batch_jqids, top_k_phrase_only, threads)
            to_update_phrase_only = {r.getKey(): r.getValue() for r in to_update_phrase_only.entrySet().toArray()}
            all_hits_phrase_only.update(to_update_phrase_only)

        # do the search
        all_search_results = []
        for query_id in queries_ids:
            hits = all_hits[query_id]
            hits_phrase_only = all_hits_phrase_only[query_id]
            hits_phrase_only_docid = set([hit.docid for hit in hits_phrase_only])
            if not SPLADE["is_maxsim"]:
                formatted_results = []
                for hit in hits:
                    if hit.docid not in hits_phrase_only_docid: continue
                    to_append = {
                        "docid": hit.docid,
                        "score": hit.score
                    }
                    formatted_results.append(to_append)
            else:
                formatted_results = do_reranking_v2(
                    full_encoded_query_info=line,
                    hits = hits,
                    add_onehot=add_onehot
                )

            all_search_results.append(formatted_results)

        # then evaluate
        predictions = convert_to_pytrec_eval_format(queries = queries_ids, all_search_results=all_search_results, type = "prediction")
        
        metadata_path = os.path.join("./metadata/",  f"{dataset_name}__{splade_model_name}")
        with open(metadata_path, "w" if chunk_idx == 0 else "a") as f:
            metadata_to_save = {
                "chunk_idx": chunk_idx,
                "predictions": predictions
            }

            if save_metadata_for_debugging:
                metadata_to_save["encoded_queries"] = encoded_queries

            json.dump(metadata_to_save, f)
            f.write("\n")


    elif mode == "eval":
        metadata_path = os.path.join("./metadata/",  f"{dataset_name}__{splade_model_name}")

        predictions = {}
        with open(metadata_path) as f:
            for line in f:
                jline = json.loads(line)
                to_update = jline.get("predictions", {})
                predictions.update(to_update)

        evaluation_result = evaluate(qrels = qrels, results = predictions, k_values = [5, 10, 50, 100, 1000])
        mrr_result = mrr(qrels = qrels, results = predictions, k_values = [5, 10, 100, 1000])

        print(evaluation_result, mrr_result)



if __name__ == "__main__":
    main()