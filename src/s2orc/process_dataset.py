# python process_dataset.py --output_file /scratch/lamdo/s2orc/sampled_10m/collections.jsonl

import hashlib, random, json
from argparse import ArgumentParser
from datasets import load_dataset
from tqdm import tqdm

def hash_document(document: str) -> str:
    """
    Hashes a document using the SHA-256 algorithm.

    Args:
        document (str): The input document as a string.

    Returns:
        str: The hexadecimal representation of the hash.
    """
    # Encode the document to bytes
    document_bytes = document.encode('utf-8')
    
    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()
    
    # Update the hash object with the document bytes
    sha256_hash.update(document_bytes)
    
    # Return the hexadecimal digest of the hash
    return sha256_hash.hexdigest()


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type = str, required = True)

    args = parser.parse_args()

    output_file = args.output_file

    ds = load_dataset("sentence-transformers/s2orc", "title-abstract-pair")

    indices = random.sample(range(len(ds["train"])), 10000000)

    with open(output_file, "w") as f:
        for i in tqdm(indices, desc = "Writing output"):
            to_write = ds["train"][i]
            json.dump(to_write, f)
            f.write('\n')
    
    

if __name__ == "__main__":
    main()