import json, sys, torch, argparse, os, pytrec_eval, logging
sys.path.append("../")
import pandas as pd
from pyserini.search.lucene import LuceneImpactSearcher
from pyserini.pyclass import JFloat, JInt, JHashMap
from pyserini_evaluation.indexing.index import init_model, SPLADE
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




def encode_queries(queries, ids = None, add_onehot = False):
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
            batch_doc_rep = SPLADE["model"].encode(batch_tokens, is_q = True)
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

        d = {SPLADE["reverse_voc"][k]: v for k, v in d.items()}
        # d = merge_dicts(d, d_added)
        d_onehot = {SPLADE["reverse_voc"][k]: v for k, v in d_onehot.items()}
        d_indices = {SPLADE["reverse_voc"][k]: v for k, v in zip(col, _indices)} if _indices is not None else None
        d_indices_onehot = {SPLADE["reverse_voc"][k]: v for k, v in zip(col_onehot, _indices_onehot)} if _indices_onehot is not None else None

        to_append = {"query_id": ids[i] if ids else None}
        to_append["vector"] = d
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splade_model_name", type = str, default = "splade_maxsim")
    parser.add_argument("--index_folder", type = str, required=True)
    parser.add_argument("--work_dir", type = str, default = "../")
    parser.add_argument("--dataset", type = str, required=True)
    parser.add_argument("--batch_size", type = int, default = 100)
    parser.add_argument("--save_metadata_for_debugging", type = str2bool, default = False)
    parser.add_argument("--add_onehot", type = str2bool, default = False)

    args = parser.parse_args()

    splade_model_name = args.splade_model_name
    index_folder = args.index_folder
    work_dir = args.work_dir
    dataset_name = args.dataset
    batch_size = args.batch_size
    save_metadata_for_debugging = args.save_metadata_for_debugging
    add_onehot = args.add_onehot

    print("yoyoyoyoyo", add_onehot)

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
        "acm_cr": "data/acm_cr/acm_cr"
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
    init_model(model_name = splade_model_name)

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
        temp = encode_queries(queries = batch_queries, ids = batch_queries_ids, add_onehot=add_onehot)
        encoded_queries.extend(temp)


    # do the search
    all_search_results = []
    for line in tqdm(encoded_queries):
        encoded_query = line["vector"]
        jquery = create_jquery(encoded_query=encoded_query, searcher = SEARCHER["searcher"])
        
        top_k = 1000 if not SPLADE["is_maxsim"] else 100
        hits = SEARCHER["searcher"].object.search(jquery, top_k)

        if not SPLADE["is_maxsim"]:
            formatted_results = []
            for hit in hits:
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
    evaluation_result = evaluate(qrels = qrels, results = predictions, k_values = [5, 10, 100, 1000])
    mrr_result = mrr(qrels = qrels, results = predictions, k_values = [5, 10, 100, 1000])

    print(evaluation_result, mrr_result)

    if save_metadata_for_debugging:
        with open(os.path.join("./metadata/",  f"{dataset_name}__{splade_model_name}"), "w") as f:
            metadata_to_save = {
                "encoded_queries": encoded_queries,
                "predictions": predictions
            }
            json.dump(metadata_to_save, f)



if __name__ == "__main__":
    main()