import torch, random, hashlib
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset
from functools import lru_cache

@lru_cache(maxsize = 10000)
def hash_document(document: str) -> str:
    document_bytes = document.encode('utf-8')
    sha256_hash = hashlib.sha256()
    sha256_hash.update(document_bytes)
    return sha256_hash.hexdigest()

class ERUKGDataset(Dataset):
    def __init__(self, path, tokenizer, max_collections = 1000000000):
        self.tokenizer = tokenizer

        self.texts = []
        visited = set([])
        with open(path) as f:
            for line in tqdm(f, desc = "Reading file"):
                splitted_line = line.split("\t")[1:]

                for doc in splitted_line:
                    doc_id = hash_document(doc)
                    if doc_id not in visited:
                        visited.add(doc_id)
                        self.texts.append(doc)

        self.texts = self.texts[:max_collections]
        print(f"INFO: There are {len(self.texts)} texts")



    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]

        res = self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors='pt',

        )

        res["input_ids"] = res["input_ids"][0]
        res["attention_mask"] = res["attention_mask"][0]

        return res