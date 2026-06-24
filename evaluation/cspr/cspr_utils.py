import json
import os

import torch
import yaml
from transformers import AutoTokenizer
from collections import Counter

from train.cspr.model import CSpR
from train.cspr.tokenizer import CSpRTokenizer
from train.train import build_vocab_partitions

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def init_model(model_name: str, config_path: str | None):
    model_dir = model_name

    vocab_partitions_path = os.path.join(model_dir, "vocab_partitions.json")
    train_config_path = os.path.join(model_dir, "train_config.json")

    if os.path.exists(vocab_partitions_path) and os.path.exists(train_config_path):
        with open(vocab_partitions_path) as f:
            vocab_partitions = json.load(f)
        with open(train_config_path) as f:
            cfg = json.load(f)
    elif config_path is not None and os.path.exists(config_path):
        vocab_partitions = None
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        raise FileNotFoundError(
            f"No model config found: expected '{train_config_path}' (with "
            f"'{vocab_partitions_path}') or a training YAML at '{config_path}'."
        )

    try:
        base_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        base_tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"])

    tok_cfg = cfg["tokenizer"]
    tokenizer = CSpRTokenizer(
        base_tokenizer,
        keyphrase_vocab_path=tok_cfg["keyphrase_vocab_path"],
        original_bert_vocab_size=cfg["original_vocab_size"],
        num_keyphrase_slots=tok_cfg["num_keyphrase_slots"],
        mask_only=tok_cfg["mask_only"],
        mode=tok_cfg.get("mode", "append"),
        keep_surface=tok_cfg.get("keep_surface", False),
        use_default=tok_cfg.get("use_default", False),
    )


    if vocab_partitions is None:
        vocab_partitions = build_vocab_partitions(tokenizer, cfg["original_vocab_size"])

    model = CSpR(
        model_type_or_dir=model_dir,
        vocab_partitions=vocab_partitions,
        bf16=cfg.get("bf16", False),
        pooling_method=cfg["pooling_method"],
        pooling_segments=cfg.get("pooling_segments"),
        sep_token_id=base_tokenizer.sep_token_id,
    )
    model.to(DEVICE)
    model.eval()

    return model, tokenizer, vocab_partitions


DEFAULT_TOP_K = {"token": 300, "phrase": 100}


@torch.no_grad()
def encode_sparse_batch(texts, model, tokenizer, vocab_partitions,
                        max_length: int = 256, quantization_factor: int = 100,
                        top_k: dict[str, int] | None = None):
    if top_k is None:
        top_k = DEFAULT_TOP_K

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    out = model.encode(enc)  # {partition: (bs, partition_size)}

    batch_size = next(iter(out.values())).shape[0]
    vectors = []
    for row in range(batch_size):
        vector = Counter()
        for partition_name, scores in out.items():
            row_scores = scores[row] * 0.25 if partition_name != "token" else scores[row]
            indices = vocab_partitions[partition_name]
            nonzero = (row_scores > 0).nonzero(as_tuple=True)[0]

            k = top_k.get(partition_name, 0)
            if k > 0 and nonzero.numel() > k:
                top = torch.topk(row_scores[nonzero], k).indices
                nonzero = nonzero[top]

            for i in nonzero.tolist():
                weight = int(row_scores[i].item() * quantization_factor)
                if weight <= 0:
                    continue
                token = tokenizer.convert_ids_to_tokens(indices[i])
                # token = token.replace("<<", "").replace(">>", "").replace("_", "-")
                token = token.replace(" ", "-")
                vector[token] += weight
        
        vectors.append(vector)
    return vectors


class CSpRQueryEncoder:

    def __init__(self, model, tokenizer, vocab_partitions,
                 max_length: int = 64, quantization_factor: int = 100,
                 top_k: dict[str, int] | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_partitions = vocab_partitions
        self.max_length = max_length
        self.quantization_factor = quantization_factor
        self.top_k = top_k

    def encode(self, query: str) -> dict:
        vectors = encode_sparse_batch(
            [query], self.model, self.tokenizer, self.vocab_partitions,
            max_length=self.max_length,
            quantization_factor=self.quantization_factor,
            top_k=self.top_k,
        )
        res = vectors[0]
        res = {k:v * 0.25 if "<<" in k and ">>" in k else v \
              for k,v in res.items()}

        return res