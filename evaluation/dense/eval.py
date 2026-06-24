import json, sys, torch, argparse, os, faiss, time
from tqdm import tqdm
from dataclasses import dataclass
from evaluation.dense.dense_utils import init_model, text_embedding_batch, DenseSearchResult
from evaluation.shared.datasets_utils import DATASET2RELPATH
from evaluation.shared.eval_utils import convert_to_pytrec_eval_format, evaluate, mrr, read_qrels

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")




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


    queries_path = os.path.join(
        work_dir, 
        "benchmarks",
        DATASET2RELPATH[dataset_name],
        "queries.jsonl")
    
    qrel_path = os.path.join(
        work_dir,
        "benchmarks",
        DATASET2RELPATH[dataset_name],
        "qrels/test.tsv" if dataset_name != "msmarco" else "qrels/dev.tsv"
    )
    qrels = read_qrels(qrel_path=qrel_path)

    index, id_map = load_index(index_path)

    model, tokenizer, prefix_ = init_model(model_name)

    model = model.to(DEVICE)

    model.eval()

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
    total_search_time = 0.0
    for i in tqdm(range(0, len(queries), batch_size), desc = f"Searching ({dataset_name})"):
        batch_queries = queries_texts[i:i+batch_size]
        batch_queries_ids = queries_ids[i:i+batch_size]


        batch_queries_embeddings = text_embedding_batch(batch = batch_queries, model = model, 
                                                        tokenizer = tokenizer, model_name = model_name, 
                                                        prefix = prefix).cpu().detach().numpy()
        

        start = time.time()
        batch_search_results = batch_search(embeddings = batch_queries_embeddings, 
                                            q_ids = batch_queries_ids, index = index, k = k, id_map = id_map)
        total_search_time += time.time() - start
        
        all_hits.update(batch_search_results)

    print(total_search_time)

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