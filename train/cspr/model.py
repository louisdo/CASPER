import torch
from abc import ABC

from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          BatchEncoding)
from cspr.partitioned_maskedlm import PartitionedMaskedLM


def _apply_pooling_cspr(inp: torch.Tensor,
                   attention_mask: torch.Tensor,
                   pm: str):
    if pm == "sum":
        return torch.sum(torch.log(1 + torch.relu(inp)) * attention_mask.unsqueeze(-1), dim=1)
    elif pm == "max":
        values, _ = torch.max(torch.log(1 + torch.relu(inp)) * attention_mask.unsqueeze(-1), dim=1)
        return values
    else:
        raise ValueError(f"Unknown pooling method '{pm}'. Expected 'sum' or 'max'.")



class BaseSparse(torch.nn.Module, ABC):
    def __init__(self, 
                 model_type_or_dir: str, 
                 model_class: type[AutoModelForMaskedLM] | None = AutoModelForMaskedLM,
                 tokenizer_class: type[AutoTokenizer] | None = AutoTokenizer,
                 bf16: bool = False):
        super().__init__()

        self.transformer = model_class.from_pretrained(
            model_type_or_dir,
            torch_dtype=torch.bfloat16 if bf16 else torch.float32
        )
        self.tokenizer = tokenizer_class.from_pretrained(model_type_or_dir)

        self.bf16 = bf16

    def forward(self, **inputs):
        out = self.transformer(**inputs)
        return out





class BaseSiamese(torch.nn.Module, ABC):
    def __init__(self, 
                 model_type_or_dir: str, 
                 model_class: type[AutoModelForMaskedLM] | None = AutoModelForMaskedLM,
                 tokenizer_class: type[AutoTokenizer] | None = AutoTokenizer,
                 bf16: bool = False):
        super().__init__()

        self.sparse_model = BaseSparse(
            model_type_or_dir = model_type_or_dir,
            model_class = model_class,
            tokenizer_class = tokenizer_class,
            bf16 = bf16
        )

    def encode(self, kwargs):
        raise NotImplementedError
    

    def encode_(self, inputs: BatchEncoding):
        return self.sparse_model(**inputs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.sparse_model.train(mode)

    def forward(self, **kwargs):
        """forward takes as inputs 1 or 2 dict
        "d_kwargs" => contains all inputs for document encoding
        "q_kwargs" => contains all inputs for query encoding ([OPTIONAL], e.g. for indexing)
        """

        out = {}
        do_d, do_q = "d_kwargs" in kwargs, "q_kwargs" in kwargs
        if do_d:
            d_rep = self.encode(kwargs["d_kwargs"])
            out.update({"d_rep": d_rep})
        if do_q:
            q_rep = self.encode(kwargs["q_kwargs"])
            out.update({"q_rep": q_rep})

        if do_d and do_q:
            if not isinstance(self.sparse_model.transformer, PartitionedMaskedLM):
                # regular sparse model
                if "score_batch" in kwargs:
                    score = torch.matmul(q_rep, d_rep.t())  # shape (bs_q, bs_d)
                else:
                    score = torch.sum(q_rep * d_rep, dim=1, keepdim=True)  # shape (bs, )
                out.update({"score": score})
            else:
                # partitioned model, including CSpR
                for partition_name in q_rep:
                    # q_rep and d_rep should be dicts
                    if "score_batch" in kwargs:
                        score = torch.matmul(q_rep[partition_name], d_rep[partition_name].t())  # shape (bs_q, bs_d)
                    else:
                        score = torch.sum(q_rep[partition_name] * d_rep[partition_name], dim=1, keepdim=True)  # shape (bs, )
                    out.update({f"score_{partition_name}": score})

        return out
    



class CSpRTrain(BaseSiamese):
    """CSpR class, this is supposed to be used for training"""

    def __init__(self,
               model_type_or_dir: str,
               vocab_partitions: dict[str, list[int]],
               model_class: type[AutoModelForMaskedLM] | None = AutoModelForMaskedLM,
               tokenizer_class: type[AutoTokenizer] | None = AutoTokenizer,
               bf16: bool = False,
               pooling_method: dict[str, str] | None = None):
        super().__init__(
            model_type_or_dir = model_type_or_dir,
            model_class = model_class,
            tokenizer_class = tokenizer_class,
            bf16 = bf16
        )

        self.sparse_model.transformer = PartitionedMaskedLM(
            self.sparse_model.transformer, vocab_partitions
        )
        self.pooling_method = pooling_method if pooling_method is not None else {"token": "max", "phrase": "sum"}

    def encode(self, inputs: BatchEncoding):
        _out = self.encode_(inputs)

        out = {}
        for partition_type in _out:
            if partition_type not in self.pooling_method:
                raise KeyError(f"Partition '{partition_type}' has no entry in pooling_method. "
                               f"Available keys: {list(self.pooling_method.keys())}")
            out[partition_type] = _apply_pooling_cspr(
                inp = _out[partition_type],
                attention_mask = inputs["attention_mask"],
                pm = self.pooling_method[partition_type]
            )
        return out
    



class CSpR(BaseSparse):
    """Also a CSpR class, but this is for inference"""
    def __init__(self,
                 model_type_or_dir: str,
                 vocab_partitions: dict[str, list[int]],
                 model_class: type[AutoModelForMaskedLM] | None = AutoModelForMaskedLM,
                 tokenizer_class: type[AutoTokenizer] | None = AutoTokenizer,
                 bf16: bool = False,
                 pooling_method: dict[str, str] | None = None):
        super().__init__(
            model_type_or_dir=model_type_or_dir,
            model_class=model_class,
            tokenizer_class=tokenizer_class,
            bf16=bf16
        )

        self.transformer = PartitionedMaskedLM(self.transformer, vocab_partitions)
        self.pooling_method = pooling_method if pooling_method is not None else {"token": "max", "phrase": "sum"}
        self.eval()

    def encode(self, inputs: BatchEncoding) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            _out = self.forward(**inputs)

        out = {}
        for partition_type in _out:
            if partition_type not in self.pooling_method:
                raise KeyError(f"Partition '{partition_type}' has no entry in pooling_method. "
                               f"Available keys: {list(self.pooling_method.keys())}")
            out[partition_type] = _apply_pooling_cspr(
                inp=_out[partition_type],
                attention_mask=inputs["attention_mask"],
                pm=self.pooling_method[partition_type]
            )
        return out
    

# utility functions down here

def to_sparse_dict(encode_out, vocab_partitions, tokenizer):
    result = {}
    for partition_name, scores in encode_out.items():
        scores = scores.squeeze(0)  # [partition_size]
        indices = vocab_partitions[partition_name]
        nonzero = (scores > 0).nonzero(as_tuple=True)[0].tolist()
        result[partition_name] = {
            tokenizer.convert_ids_to_tokens(indices[i]): scores[i].item()
            for i in nonzero
        }
    return result