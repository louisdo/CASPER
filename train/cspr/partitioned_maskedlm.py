import torch
from typing import Dict, List
from transformers import AutoModelForMaskedLM


class PartitionedMaskedLM(torch.nn.Module):
    def __init__(self, model: AutoModelForMaskedLM, vocab_partitions: Dict[str, List[int]]):
        super().__init__()
        self.model = model
        self.partition_names = list(vocab_partitions.keys())

        self._partition_slice: Dict[str, tuple[int, int] | None] = {}
        for name, indices in vocab_partitions.items():
            self.register_buffer(f"partition_{name}", torch.tensor(indices, dtype=torch.long))
            self._partition_slice[name] = self._as_contiguous_range(indices)

    @staticmethod
    def _as_contiguous_range(indices: List[int]) -> tuple[int, int] | None:
        if len(indices) > 0 and indices == list(range(indices[0], indices[0] + len(indices))):
            return indices[0], indices[0] + len(indices)
        return None

    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        logits = self.model(**inputs).logits  # [B, S, vocab_size]
        out = {}
        for name in self.partition_names:
            rng = self._partition_slice[name]
            if rng is not None:
                out[name] = logits[:, :, rng[0]:rng[1]]  # view, no copy
            else:
                out[name] = logits.index_select(2, getattr(self, f"partition_{name}"))
        return out
