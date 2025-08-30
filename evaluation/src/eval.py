import sys, json, logging, pytrec_eval
sys.path.append("../..")
import pandas as pd
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from argparse import ArgumentParser
from typing import List, Dict, Tuple
from model_name_2_model_info import model_name_2_model_dir

logger = logging.getLogger(__name__)

DATASETS_WITH_LONG_QUERY = ["doris_mae"]


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
    qrels,
    results,
    k_values,
):
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
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type = str, required = True)
    parser.add_argument("--model_name", type = str, required=True)
    parser.add_argument("--experiment_path", type = str, default = "./experiments")
    parser.add_argument("--eval_data_folder", type = str, default="../../data")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    experiment_path = args.experiment_path
    model_name = args.model_name
    eval_data_folder = args.eval_data_folder

    queries_path = f"{eval_data_folder}/{dataset_name}/queries.jsonl"
    id2originalid_path = f"{eval_data_folder}/{dataset_name}/id2originalid.json"
    qrel_path = f"{eval_data_folder}/{dataset_name}/qrels/test.tsv"

    with open(id2originalid_path) as f:
        id2originalid = json.load(f)

    _queries = {}
    with open(queries_path) as f:
        for line in f:
            jline = json.loads(line)
            id = jline["id"]
            content = jline["text"]
            _queries[id] = content

    with Run().context(RunConfig(nranks=1, experiment=dataset_name, root = experiment_path)):

        config = ColBERTConfig(
            query_maxlen= 180 if dataset_name in DATASETS_WITH_LONG_QUERY else 64,
            doc_maxlen=256,
        )
        searcher = Searcher(index=f"{dataset_name}.nbits=2", config=config)

        queries = Queries(data = _queries)
        ranking = searcher.search_all(queries, k=1000)
        ranking_dict = ranking.todict()

    all_search_results = []
    queries_ids = []
    qrels = read_qrels(qrel_path=qrel_path)
    for queryid, results in ranking_dict.items():
        queryid = int(queryid)
        original_query_id = str(id2originalid["queries"][queryid])
        if original_query_id not in qrels: continue

        queries_ids.append(original_query_id)
        to_append = [{"docid": str(id2originalid["corpus"][int(r[0])]), "score": r[2]} for r in results]
        all_search_results.append(to_append)

    predictions = convert_to_pytrec_eval_format(queries = queries_ids, all_search_results=all_search_results, type = "prediction")

    evaluation_result = evaluate(qrels = qrels, results = predictions, k_values = [5, 10, 100, 1000])
    mrr_result = mrr(qrels = qrels, results = predictions, k_values = [5, 10, 100, 1000])

    print(evaluation_result, mrr_result)
        




if __name__=='__main__':
    main()