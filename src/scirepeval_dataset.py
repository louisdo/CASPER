import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class SciRepEvalDataset(Dataset):
    def __init__(self, path, tokenizer, max_collections = 1000000000):
        self.tokenizer = tokenizer
        
        self.texts = []
        count = 0
        with open(path) as f:
            for line in tqdm(f, desc = "Loading collection"):
                if count >= max_collections: break
                self.texts.append(line)
                count += 1
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        res = self.tokenizer(
            self.texts[idx], 
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors='pt',

        )

        res["input_ids"] = res["input_ids"][0]
        res["attention_mask"] = res["attention_mask"][0]

        return res