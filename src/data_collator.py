from transformers import DataCollatorForLanguageModeling
import torch
import numpy as np
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

class FocusedMaskDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, *args, special_token_id_start=None, special_token_mask_probability=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.special_token_id_start = special_token_id_start 
        self.special_token_mask_probability = special_token_mask_probability  # Probability multiplier for special tokens

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        if self.special_token_id_start is not None:
            special_tokens_mask = (inputs >= self.special_token_id_start).bool()
            probability_matrix[special_tokens_mask] = self.special_token_mask_probability

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels