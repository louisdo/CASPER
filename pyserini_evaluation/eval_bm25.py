import json, sys, torch, argparse, os, pytrec_eval, logging
sys.path.append("../")
import pandas as pd
import numpy as np
from pyserini.search.lucene import LuceneSearcher
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
}

def init_searcher(index_path):
    if SEARCHER["searcher"] is None or SEARCHER["index_path"] != index_path:
        searcher = LuceneSearcher(index_dir = index_path)
        SEARCHER["searcher"] = searcher
        SEARCHER["index_path"] = index_path

        SEARCHER["searcher"].set_bm25(k1=0.9, b=0.4)

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_folder", type = str, required=True)
    parser.add_argument("--work_dir", type = str, default = "../")
    parser.add_argument("--dataset", type = str, required=True)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--threads", type = int, default = 8)

    args = parser.parse_args()

    index_folder = args.index_folder
    work_dir = args.work_dir
    dataset_name = args.dataset
    batch_size = args.batch_size
    threads = args.threads

    if dataset_name in ["trec_dl_2019", "trec_dl_2020"]:
        index_path = os.path.join(index_folder, f"msmarco__bm25")
    else:
        index_path = os.path.join(index_folder, f"{dataset_name}__bm25")


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

    init_searcher(index_path = index_path)


    # load queries
    with open(queries_path) as f:
        queries = [json.loads(line) for line in f]
        queries = [line for line in queries if line["_id"] in qrels]

    top_k = 1000
    all_hits = {}
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        batch_queries = [line["text"] for line in batch]
        batch_qids = [line["_id"] for line in batch]

        batch_search_results = SEARCHER["searcher"].batch_search(batch_queries, batch_qids, top_k, threads)
        all_hits.update(batch_search_results)


    queries_ids = [line["_id"] for line in queries]
    all_search_results = []
    for query_id in queries_ids:
        hits = all_hits[query_id]
        formatted_results = []
        for hit in hits:
            to_append = {
                "docid": hit.docid,
                "score": hit.score
            }
            formatted_results.append(to_append)

        all_search_results.append(formatted_results)


    predictions = convert_to_pytrec_eval_format(queries = queries_ids, all_search_results=all_search_results, type = "prediction")
    evaluation_result = evaluate(qrels = qrels, results = predictions, k_values = [5, 10, 50, 100, 1000])
    mrr_result = mrr(qrels = qrels, results = predictions, k_values = [5, 10, 50, 100, 1000])

    print(evaluation_result, mrr_result)


if __name__ == "__main__":
    main()