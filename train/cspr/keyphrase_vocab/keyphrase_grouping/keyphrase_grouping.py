import json
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent, a, b):
    parent[_find(parent, a)] = _find(parent, b)


def group_keyphrases(
    keyphrase_index: dict,
    model_name: str,
    threshold: float,
    batch_size: int,
    device: str,
    top_k: int = 32,
    frequency_filter: int | None = 3,
) -> dict[str, list[str]]:
    keyphrases = list(keyphrase_index.keys())
    keyphrases = [kp for kp in keyphrases \
                  if len(keyphrase_index[kp]) >= frequency_filter]
    n = len(keyphrases)

    print(f"Embedding {n} keyphrases...")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        keyphrases,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Searching top-{top_k} neighbors per keyphrase...")
    similarities, indices = index.search(embeddings, top_k)

    print("Clustering...")
    parent = list(range(n))

    for i in range(n):
        for j, sim in zip(indices[i], similarities[i]):
            if j != i and sim >= threshold:
                _union(parent, i, int(j))

    groups: dict[int, list[str]] = defaultdict(list)
    for i, kp in enumerate(keyphrases):
        groups[_find(parent, i)].append(kp)

    return {keyphrases[root]: members for root, members in groups.items()}


def main():
    parser = ArgumentParser(description="Group near-duplicate keyphrases via embedding similarity.")
    parser.add_argument("--input", required=True, help="JSON file: {keyphrase: [doc_id, ...]}")
    parser.add_argument("--output", required=True, help="Output JSON file: {canonical: [keyphrase, ...]}")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--threshold", type=float, default=0.92, help="Cosine similarity threshold (0-1)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=32, help="Neighbors to check per keyphrase in FAISS")
    parser.add_argument("--frequency-filter", type = int, default = 10, help = "Frequency filter, remove all keyphrases with lower frequency")
    args = parser.parse_args()

    with open(args.input) as f:
        keyphrase_index = json.load(f)

    groups = group_keyphrases(
        keyphrase_index,
        model_name=args.model,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
        top_k=args.top_k,
        frequency_filter = args.frequency_filter
    )

    with open(args.output, "w") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)

    merged_groups = [g for g in groups.values() if len(g) > 1]
    print(f"Done. {len(groups)} groups total, {len(merged_groups)} groups with 2+ keyphrases.")


if __name__ == "__main__":
    main()
