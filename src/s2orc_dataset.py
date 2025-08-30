import torch, random
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset

class S2ORCDataset(Dataset):
    def __init__(self, path, tokenizer, max_collections = 1000000000, sampling_seed = 0, sample_size = 10000000):
        self.tokenizer = tokenizer

        print("WARNING: 'path' is not in effect for S2ORCDataset")

        ds = load_dataset("sentence-transformers/s2orc", "title-abstract-pair")
        
        random.seed(sampling_seed)
        # self.indices = random.sample(range(len(ds["train"])), sample_size)
        self.indices = set(random.sample(range(len(ds["train"])), sample_size))

        self.dataset = []
        # for index in tqdm(self.indices[:max_collections], desc = "Loading dataset into memory"):
        #     self.dataset.append(ds["train"][index])

        for index in tqdm(range(len(ds["train"])), desc = "Loading dataset into memory"):
            if index in self.indices: self.dataset.append(ds["train"][index])



        

    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):


        line = self.dataset[idx]
        title = line.get("title", "")
        abstract = line.get("abstract", "")

        text = f"{title}. {abstract}"

        res = self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=512,
            # return_tensors='pt',

        )

        # res["input_ids"] = res["input_ids"][0]
        # res["attention_mask"] = res["attention_mask"][0]
        return {key: torch.tensor(val) for key, val in res.items()}
        # return res