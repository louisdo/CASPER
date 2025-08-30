import torch, random, hashlib
from torch.utils.data import Dataset
from tqdm import tqdm

class MSMARCODataset(Dataset):
    def __init__(self, path, tokenizer, max_collections = 1000000000):
        self.tokenizer = tokenizer

        self.texts = []
        with open(path) as f:
            error_count = 0
            for line in tqdm(f, desc = "Reading file"):
                splitted_line = line.split("\t")[:]

                if len(splitted_line) != 2:
                    error_count += 1
                    continue
                docid, text = splitted_line
                self.texts.append(text)

            print("Number of errors", error_count)

        self.texts = self.texts[:max_collections]
        print("Number of text in dataset", len(self.texts))


    
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