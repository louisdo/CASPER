import json, sys, torch, argparse, os, pytrec_eval, logging
sys.path.append("../")
import pandas as pd
from pyserini.search.lucene import LuceneImpactSearcher
from pyserini.pyclass import JFloat, JInt, JHashMap
from pyserini_evaluation.indexing.index_ensemble import init_model, SPLADE, PHRASE_SPLADE, merge_representation
from pyserini_evaluation.indexing.utils import torch_csr_to_scipy_csr, merge_dicts
from pyserini_evaluation.indexing.model_name_2_info import model_name_2_path, model_name_2_model_class, model_name_2_is_maxsim
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import Counter
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from transformers import AutoTokenizer

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


def create_jquery(encoded_query, searcher, fields = {}):
    jfields = JHashMap()
    for (field, boost) in fields.items():
        jfields.put(field, JFloat(boost))

    jquery = JHashMap()
    for (token, weight) in encoded_query.items():
        if token in searcher.idf and searcher.idf[token] > searcher.min_idf:
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


def encode_queries_helper(queries, CONTAINER):
    with torch.no_grad():
        if CONTAINER["is_maxsim"]:
            batch_doc_rep, batch_doc_token_indices, batch_doc_pad_len = \
                CONTAINER["model"].encode(
                    CONTAINER["tokenizer"](queries, return_tensors="pt", truncation = True, padding = True, max_length=256).to(DEVICE), 
                    is_q = True
                )
            
            bs = batch_doc_rep.shape[0]
            out_dim = batch_doc_rep.shape[1]
            batch_doc_rep_full = torch.zeros((bs, MAX_LENGTH, out_dim), dtype=batch_doc_rep.dtype).to(batch_doc_rep.device)
            batch_doc_rep_full.scatter_(1, batch_doc_token_indices.unsqueeze(1), batch_doc_rep.unsqueeze(1))

            batch_doc_rep_full = [item.to_sparse_csr().cpu() for item in batch_doc_rep_full]
            batch_doc_rep_full = [torch_csr_to_scipy_csr(item) for item in batch_doc_rep_full]
        else:
            batch_doc_rep = CONTAINER["model"].encode(CONTAINER["tokenizer"](queries, return_tensors="pt", truncation = True, padding = True, max_length=256).to(DEVICE), is_q = True)
            batch_doc_token_indices = None
            batch_doc_pad_len = None
            batch_doc_rep_full = None
    
    assert len(queries) == batch_doc_rep.size(0)

    res = []
    for i in range(batch_doc_rep.size(0)):
        doc_rep = batch_doc_rep[i]
        doc_token_indices = batch_doc_token_indices[i] if batch_doc_token_indices is not None else None

        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        _indices = doc_token_indices[col].cpu().tolist() if doc_token_indices is not None else None
        d = {k: int(v * 100) for k, v in zip(col, weights)}

        d = {CONTAINER["reverse_voc"][k]: v for k, v in d.items()}
        # d = merge_dicts(d, d_added)
        d_indices = {CONTAINER["reverse_voc"][k]: v for k, v in zip(col, _indices)} if _indices is not None else None

        to_append = {}
        to_append["vector"] = d

        if d_indices is not None:
            to_append["token_indices"] = d_indices
            to_append["pad_len"] = batch_doc_pad_len # may not be necessary


        if batch_doc_rep_full is not None:
            to_append["full_rep"] = batch_doc_rep_full[i]

        res.append(to_append)

    return res




def encode_queries(queries):
    phrase_splade_encoded_queries = encode_queries_helper(queries, PHRASE_SPLADE)
    normal_splade_encoded_queries = encode_queries_helper(queries, SPLADE)
    return merge_representation(phrase_splade_representation=phrase_splade_encoded_queries, normal_splade_representation=normal_splade_encoded_queries)


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
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
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
        doc_id = line["corpus-id"]
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




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrase_splade_model_name", type = str, default = "phrase_splade")
    parser.add_argument("--normal_splade_model_name", type = str, default = "eru_kg")
    parser.add_argument("--index_folder", type = str, required=True)
    parser.add_argument("--work_dir", type = str, default = "../")
    parser.add_argument("--dataset", type = str, required=True)
    parser.add_argument("--batch_size", type = int, default = 100)

    args = parser.parse_args()

    phrase_splade_model_name = args.phrase_splade_model_name
    normal_splade_model_name = args.normal_splade_model_name
    index_folder = args.index_folder
    work_dir = args.work_dir
    dataset_name = args.dataset
    batch_size = args.batch_size

    index_path = os.path.join(index_folder, f"{dataset_name}__{phrase_splade_model_name}+{normal_splade_model_name}")


    dataset_name_2_relative_path = {
        "scifact": "data/beir/scifact",
        "scidocs": "data/beir/scidocs",
        "nfcorpus": "data/beir/nfcorpus",
        "arguana": "data/beir/arguana",
        "fiqa": "data/beir/fiqa",
        "msmarco": "data/msmarco/msmarco",
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

    


    # init splade model
    init_model(phrase_splade_model_name = phrase_splade_model_name, normal_model_name=normal_splade_model_name)

    # init searcher
    init_searcher(index_path = index_path)

    queries_texts = [line["text"] for line in queries]
    queries_ids = [line["_id"] for line in queries]

    # encode the queries
    encoded_queries = []
    for i in tqdm(range(0, len(queries_texts), batch_size)):
        batch_queries = queries_texts[i:i+batch_size]
        temp = encode_queries(queries = batch_queries)
        encoded_queries.extend(temp)
    
    with open("test_gitig___", "w") as f:
        json.dump(encoded_queries, f, indent = 4)

    # do the search
    all_search_results = []
    for line in tqdm(encoded_queries):
        encoded_query = line["vector"]
        jquery = create_jquery(encoded_query=encoded_query, searcher = SEARCHER["searcher"])
        
        top_k = 100
        hits = SEARCHER["searcher"].object.search(jquery, top_k)
        formatted_results = []
        for hit in hits:
            to_append = {
                "docid": hit.docid,
                "score": hit.score
            }
            formatted_results.append(to_append)

        all_search_results.append(formatted_results)

    # then evaluate
    predictions = convert_to_pytrec_eval_format(queries = queries_ids, all_search_results=all_search_results, type = "prediction")
    evaluation_result = evaluate(qrels = qrels, results = predictions, k_values = [10, 100])
    mrr_result = mrr(qrels = qrels, results = predictions, k_values = [10, 100])

    print(evaluation_result, mrr_result)



if __name__ == "__main__":
    main()