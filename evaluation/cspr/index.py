import os, json
from argparse import ArgumentParser

from tqdm import tqdm

from evaluation.cspr.cspr_utils import init_model, encode_sparse_batch
from evaluation.shared.datasets_utils import DATASET2RELPATH

def main():
    parser = ArgumentParser("Indexing for CSpR")

    parser.add_argument("--model_name", type = str, required = True)
    parser.add_argument("--config_path", type = str, default = None)
    parser.add_argument("--dataset", type = str, required = True)
    parser.add_argument("--work_dir", type = str, default="../../")
    parser.add_argument("--index_path", type = str, required=True)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--max_length", type = int, default = 256)
    parser.add_argument("--quantization_factor", type = int, default = 100)
    parser.add_argument("--top_k_token", type=int, default=300)
    parser.add_argument("--top_k_phrase", type=int, default=100)

    args = parser.parse_args()

    top_k = {"token": args.top_k_token, "phrase": args.top_k_phrase}

    model, tokenizer, vocab_partitions = init_model(args.model_name, args.config_path)

    # load corpus
    corpus_path = os.path.join(
        args.work_dir, 
        "benchmarks",
        DATASET2RELPATH[args.dataset],
        "corpus.jsonl")
    assert os.path.exists(corpus_path), corpus_path

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            jline = json.loads(line)
            corpus.append(jline)
    
    print("Corpus length", len(corpus))


    collection_path = os.path.join(args.index_path, "collection", "chunk.jsonl")
    os.makedirs(os.path.dirname(collection_path) or ".", exist_ok=True)

    with open(collection_path, "w") as out_f:
        for i in tqdm(range(0, len(corpus), args.batch_size), desc="Encoding corpus"):
            batch = corpus[i:i + args.batch_size]
            contents_batch = [f"{item.get('title', '')} | {item['text']}" for item in batch]

            vectors = encode_sparse_batch(
                contents_batch, model, tokenizer, vocab_partitions,
                max_length=args.max_length,
                quantization_factor=args.quantization_factor,
                top_k = top_k,
            )

            for item, contents, vector in zip(batch, contents_batch, vectors):
                to_write = {
                    "id": item.get("_id", item.get("id")),
                    "contents": contents,
                    "vector": vector,
                }
                out_f.write(json.dumps(to_write) + "\n")

    print("Wrote collection to", collection_path)


if __name__ == "__main__":
    main()