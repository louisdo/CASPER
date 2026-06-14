from transformers import ModernBertConfig
from typing import Dict, List, Optional



class PartitionedModernBertConfig(ModernBertConfig):
    # PMB: Partitioned Modern BERT
    model_type = "PMB"

    def __init__(self, vocab_partitions: Optional[Dict[str, List[int]]] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vocab_partitions = vocab_partitions
        if vocab_partitions is not None and not self._check_vocab_partitions():
            raise ValueError(
                "Invalid vocab_partitions: must contain a 'token' partition starting at index 0 "
                f"with size equal to vocab_size ({self.vocab_size}), and all partitions must "
                "form a contiguous, gap-free index space. "
                f"Got: { {k: (v[0], v[-1]) for k, v in self.vocab_partitions.items()} }"
            )


    def _check_vocab_partitions(self) -> bool:
        """
        Validates that vocab_partitions covers a contiguous, gap-free index space
        starting at 0, and that the "token" partition starts from 0 and have size
        equals to 

        A valid partition example: {"token": range(30522), "phrase": range(30522, 50000)}
        """
        MODERNBERT_VOCAB_SIZE = self.vocab_size

        if "token" not in self.vocab_partitions:
            return False
        if len(self.vocab_partitions["token"]) != MODERNBERT_VOCAB_SIZE \
            and self.vocab_partitions["token"][0] != 0:
            return False

        sorted_parts = sorted(self.vocab_partitions.values(), key=lambda p: p[0])
        expected = 0
        for part in sorted_parts:
            start, end = part[0], part[-1]
            if start != expected or len(part) != end - start + 1:
                return False
            expected = end + 1

        return True