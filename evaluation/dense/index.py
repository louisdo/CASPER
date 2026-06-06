import os, json, torch, faiss
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from evaluation.dense.dense_utils import init_model, text_embedding_batch
from evaluation.shared.datasets_utils import DATASET2RELPATH

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    

def do_indexing(embeddings, string_ids=None, index_folder=None):
    """
    Build a Faiss Flat IP index for cosine similarity with string IDs.
    Args:
        embeddings (np.ndarray): shape (num_vectors, dim)
        string_ids (list of str or None): List of string IDs.
        index_folder (str or None): Folder to save the index and mapping.
    Returns:
        index: Faiss index (IndexIDMap)
        id_map: dict mapping int_id -> string_id
    """
    embeddings = embeddings.astype('float32')
    num_vectors, dim = embeddings.shape

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build Flat IP index
    base_index = faiss.IndexFlatIP(dim)

    # Handle string IDs
    if string_ids is not None:
        assert len(string_ids) == num_vectors
        int_ids = np.arange(num_vectors, dtype=int)
        id_map = dict(zip(int_ids, string_ids))
        index = faiss.IndexIDMap(base_index)
        index.add_with_ids(embeddings, int_ids)
    else:
        id_map = None
        index = base_index
        index.add(embeddings)

    # Optionally save
    if index_folder is not None:
        if not os.path.exists(index_folder):
            os.makedirs(index_folder)
        faiss.write_index(index, os.path.join(index_folder, "faiss_index_flatip.index"))
        if id_map is not None:
            import json
            with open(os.path.join(index_folder, "id_map.json"), "w") as f:
                json.dump({int(k): str(v) for k,v in id_map.items()}, f)

    return index, id_map


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type = str, required = True)
    parser.add_argument("--dataset", type = str, required = True)
    parser.add_argument("--work_dir", type = str, default="../../")
    parser.add_argument("--index_path", type = str, required=True)
    parser.add_argument("--batch_size", type = int, default = 32)

    args = parser.parse_args()


    model_name = args.model_name
    dataset_name = args.dataset
    work_dir = args.work_dir
    index_path = args.index_path
    batch_size = args.batch_size


    model, tokenizer, prefix_ = init_model(model_name)
    if not prefix_:
        prefix = None
    else: prefix = prefix_["doc"]

    model = model.to(DEVICE)

    # load corpus
    corpus_path = os.path.join(
        work_dir, 
        "benchmarks",
        DATASET2RELPATH[dataset_name],
        "corpus.jsonl")
    assert os.path.exists(corpus_path), corpus_path

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            jline = json.loads(line)
            corpus.append(jline)
    
    print("Corpus length", len(corpus))

    ids = []
    embeddings = []
    for i in tqdm(range(0, len(corpus), batch_size), desc = "Generating embeddings"):
        batch = corpus[i:i+batch_size]
        text_batch = [f"{line['title']} | {line['text']}" for line in batch]

        batch_embeddings = text_embedding_batch(batch = text_batch, model = model, 
                                                tokenizer = tokenizer, model_name = model_name, 
                                                prefix = prefix)

        for line, embedding in zip(batch, batch_embeddings):
            line_id = line.get("_id", line.get("id", None))
            list_embedding = embedding.cpu().tolist()
            embeddings.append(list_embedding)
            ids.append(line_id)

    do_indexing(embeddings = np.array(embeddings), string_ids = ids, index_folder = index_path)


if __name__ == "__main__":
    main()