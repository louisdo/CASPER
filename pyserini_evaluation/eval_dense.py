import json, sys, torch, argparse, os, pytrec_eval, logging, faiss
sys.path.append("../")
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pyserini_evaluation.indexing.model_name_2_model_info_dense import model_name_2_model_class, model_name_2_model_path, model_name_2_tokenizer_class, model_name_2_prefix

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@dataclass
class DenseSearchResult:
    docid: str
    score: float


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


def batch_search(embeddings, q_ids, index, k, id_map=None):
    faiss.normalize_L2(embeddings)
    D, I = index.search(embeddings, k)
    results = {
        key: [
            DenseSearchResult(
                docid=id_map[int(idx)] if id_map is not None else int(idx),
                score=float(score)
            )
            for score, idx in zip(distances, indexes) if idx != -1
        ]
        for key, distances, indexes in zip(q_ids, D, I)
    }
    return results


def text_embedding_batch(batch, model, tokenizer, model_name, prefix = None):
    if prefix is not None:
        batch = [prefix + " " + text for text in batch]
    inputs = tokenizer(batch, padding=True, truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=256).to(DEVICE)
    output = model(**inputs)

    if model_name in ["specter2"]:
        return output.last_hidden_state[:, 0, :].cpu()
    
    elif model_name in ["e5_base"]:
        attention_mask = inputs["attention_mask"]
        last_hidden = output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    else:
        raise NotImplementedError


def load_index(index_folder):
    index = faiss.read_index(os.path.join(index_folder, "faiss_index_flatip.index"))
    id_map_path = os.path.join(index_folder, "id_map.json")
    if os.path.exists(id_map_path):
        import json
        with open(id_map_path, "r") as f:
            id_map = json.load(f)
        # JSON keys are strings, convert them back to int
        id_map = {int(k): v for k, v in id_map.items()}
    else:
        id_map = None
    return index, id_map



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str, required = True)
    parser.add_argument("--index_folder", type = str, required=True)
    parser.add_argument("--work_dir", type = str, default = "../")
    parser.add_argument("--dataset", type = str, required=True)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--threads", type = int, default = 8)

    args = parser.parse_args()

    model_name = args.model_name
    index_folder = args.index_folder
    work_dir = args.work_dir
    dataset_name = args.dataset
    batch_size = args.batch_size
    threads = args.threads

    if dataset_name in ["trec_dl_2019", "trec_dl_2020"]:
        index_path = os.path.join(index_folder, f"msmarco__{model_name}")
    else:
        index_path = os.path.join(index_folder, f"{dataset_name}__{model_name}")


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

    index, id_map = load_index(index_path)

    model = model_name_2_model_class[model_name].from_pretrained(model_name_2_model_path[model_name])
    tokenizer = model_name_2_tokenizer_class[model_name].from_pretrained(model_name_2_model_path[model_name])

    model = model.to(DEVICE)

    model.eval()

    prefix_ = model_name_2_prefix.get(model_name)
    if not prefix_:
        prefix = None
    else: prefix = prefix_["query"]

    # load queries
    with open(queries_path) as f:
        queries = [json.loads(line) for line in f]
        queries = [line for line in queries if line["_id"] in qrels]


    queries_texts = [line["text"] for line in queries]
    queries_ids = [line["_id"] for line in queries]

    k = 1000
    all_hits = {}
    for i in tqdm(range(0, len(queries), batch_size), desc = f"Searching ({dataset_name})"):
        batch_queries = queries_texts[i:i+batch_size]
        batch_queries_ids = queries_ids[i:i+batch_size]


        batch_queries_embeddings = text_embedding_batch(batch = batch_queries, model = model, 
                                                        tokenizer = tokenizer, model_name = model_name, prefix = prefix).cpu().detach().numpy()
        

        batch_search_results = batch_search(embeddings = batch_queries_embeddings, 
                                            q_ids = batch_queries_ids, index = index, k = k, id_map = id_map)
        
        all_hits.update(batch_search_results)

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