import json
import os
from typing import Optional, Dict, List

from transformers import AutoTokenizer, PreTrainedTokenizer


class CSpRTokenizer:
    def __init__(self, base_tokenizer: AutoTokenizer, vocab_partitions: Dict[str, List[int]]):
        self._tokenizer = base_tokenizer
        self.vocab_partitions = vocab_partitions

    def preprocess(self, text: str) -> str:
        # TODO: your preprocessing here
        return text

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            text = self.preprocess(text)
        elif isinstance(text, list):
            text = [self.preprocess(t) for t in text]
        return self._tokenizer(text, **kwargs)

    def __getattr__(self, name):
        # Delegate everything else (vocab, special tokens, decode, ...) to base
        return getattr(self._tokenizer, name)
