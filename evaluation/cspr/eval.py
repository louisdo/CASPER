import json, os
from argparse import ArgumentParser

from tqdm import tqdm
from pyserini.search.lucene import LuceneImpactSearcher

from evaluation.cspr.cspr_utils import init_model, CSpRQueryEncoder
from evaluation.shared.datasets_utils import DATASET2RELPATH
from evaluation.shared.eval_utils import evaluate, mrr, read_qrels


def format_predictions_for_eval(all_hits: dict) -> dict:
    """Convert Pyserini ``batch_search`` results into the {qid: {docid: score}} dict
    expected by pytrec_eval."""
    predictions = {}
    for qid, hits in all_hits.items():
        predictions[qid] = {str(hit.docid): float(hit.score) for hit in hits}
    return predictions


def main():
    parser = ArgumentParser("Evaluation for CSpR")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--config_path", type = str, default = None)
    parser.add_argument("--index_folder", type=str, required=True)
    parser.add_argument("--model_tag", type=str, default="cspr",
                        help="short tag used to name the index folder (matches index.py)")
    parser.add_argument("--work_dir", type=str, default="../")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--quantization_factor", type=int, default=100)
    parser.add_argument("--top_k_token", type=int, default=300)
    parser.add_argument("--top_k_phrase", type=int, default=100)

    args = parser.parse_args()

    top_k = {"token": args.top_k_token, "phrase": args.top_k_phrase}

    # index folder name mirrors index.py: "{dataset}__{model_tag}"
    dataset_for_index = "msmarco" if args.dataset in ["trec_dl_2019", "trec_dl_2020"] else args.dataset
    index_path = os.path.join(args.index_folder, f"{dataset_for_index}__{args.model_tag}")

    lucene_path = os.path.join(index_path, "lucene")
    assert os.path.exists(lucene_path), lucene_path

    queries_path = os.path.join(
        args.work_dir, "benchmarks", DATASET2RELPATH[args.dataset], "queries.jsonl")
    qrel_path = os.path.join(
        args.work_dir, "benchmarks", DATASET2RELPATH[args.dataset],
        "qrels/test.tsv" if args.dataset != "msmarco" else "qrels/dev.tsv")
    qrels = read_qrels(qrel_path=qrel_path)

    model, tokenizer, vocab_partitions = init_model(args.model_name, args.config_path)

    query_encoder = CSpRQueryEncoder(
        model, tokenizer, vocab_partitions,
        max_length=args.max_length,
        quantization_factor=args.quantization_factor,
        top_k=top_k,
    )
    searcher = LuceneImpactSearcher(lucene_path, query_encoder)

    queries = []
    with open(queries_path) as f:
        for line in f:
            jline = json.loads(line)
            qid = str(jline["_id"])
            if qid in qrels:
                queries.append((qid, jline["text"]))
    print(f"Searching {len(queries)} queries (k=1000)")

    all_hits = {}
    for i in tqdm(range(0, len(queries), args.batch_size), desc="Searching"):
        batch = queries[i:i + args.batch_size]
        batch_qids = [qid for qid, _ in batch]
        batch_queries = [text for _, text in batch]
        batch_hits = searcher.batch_search(
            batch_queries, batch_qids, k=1000, threads=args.threads)
        all_hits.update(batch_hits)


    predictions = format_predictions_for_eval(all_hits)

    k_values = [5, 10, 50, 100, 1000]
    evaluation_result = evaluate(qrels=qrels, results=predictions, k_values=k_values)
    mrr_result = mrr(qrels=qrels, results=predictions, k_values=k_values)

    print(evaluation_result, mrr_result)

if __name__ == "__main__":
    main()
