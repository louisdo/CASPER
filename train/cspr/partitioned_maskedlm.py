import torch
from typing import Dict, List
from transformers import AutoModelForMaskedLM


class PartitionedMaskedLM(torch.nn.Module):
    def __init__(self, model: AutoModelForMaskedLM, vocab_partitions: Dict[str, List[int]]):
        super().__init__()
        self.model = model
        self.partition_names = list(vocab_partitions.keys())
        for name, indices in vocab_partitions.items():
            self.register_buffer(f"partition_{name}", torch.tensor(indices, dtype=torch.long))

    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        logits = self.model(**inputs).logits  # [B, S, vocab_size]
        return {name: logits[:, :, getattr(self, f"partition_{name}")] for name in self.partition_names}
