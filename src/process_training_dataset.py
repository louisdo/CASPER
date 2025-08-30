import json, hashlib
from argparse import ArgumentParser
from datasets import load_dataset
from tqdm import tqdm

def generate_doc_id(text):
    encoded_text = text.encode('utf-8')
    
    hash_object = hashlib.sha256(encoded_text)
    
    return hash_object.hexdigest()


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type = str, required=True)
    parser.add_argument("--max_documents", type = int, default = 1000000000)


    args = parser.parse_args()

    output_file = args.output_file
    max_documents = args.max_documents

    assert output_file.endswith(".txt"), "Must be a txt file"

    ds = load_dataset("allenai/scirepeval", "search")
    visited_doc_id = set()

    count = 0
    with open(output_file, "w") as f:
        pbar = tqdm(ds["train"])
        for line in pbar:
            candidates = line.get("candidates")
            if not candidates: continue

            for doc in candidates:
                title = doc.get("title")
                abstract = doc.get("abstract")

                doc_id = generate_doc_id(title.lower())

                if doc_id in visited_doc_id:
                    continue

                text = f"{title}. {abstract}".replace("..", ".")

                f.write(text + "\n")
                visited_doc_id.add(doc_id)
                count += 1

                
            if count >= max_documents: break

            pbar.set_postfix({"count": count})


if __name__ == "__main__":
    main()