import torch, random, json
from torch.utils.data import Dataset
from tqdm import tqdm

class S2ORCCSDataset(Dataset):
    def __init__(self, path, tokenizer, max_collections = 1000000000, sampling_seed = 0, sample_size = 10000000):
        self.tokenizer = tokenizer

        print("WARNING: 'sampling_seed', 'sample_size' is not in effect for S2ORCDataset")

        self.dataset = []
        with open(path) as f:
            for line in tqdm(f):
                jline = json.loads(line)
                self.dataset.append(jline)

        if max_collections < len(self.dataset):
            random.Random(4).shuffle(self.dataset)
            self.dataset = self.dataset[:max_collections]


    
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