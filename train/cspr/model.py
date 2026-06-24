import torch
from abc import ABC

from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          BatchEncoding)
from train.cspr.partitioned_maskedlm import PartitionedMaskedLM


def _apply_pooling_cspr(inp: torch.Tensor,
                   attention_mask: torch.Tensor,
                   pm: str):
    if pm == "sum":
        return torch.sum(torch.log(1 + torch.relu(inp)) * attention_mask.unsqueeze(-1), dim=1)
    elif pm == "max":
        values, _ = torch.max(torch.log(1 + torch.relu(inp)) * attention_mask.unsqueeze(-1), dim=1)
        return values
    elif pm == "mean":
        weighted = torch.log(1 + torch.relu(inp)) * attention_mask.unsqueeze(-1)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(weighted.dtype)
        return torch.sum(weighted, dim=1) / denom
    else:
        raise ValueError(f"Unknown pooling method '{pm}'. Expected 'sum' or 'max'.")


def _segment_position_mask(input_ids: torch.Tensor,
                           attention_mask: torch.Tensor,
                           sep_token_id: int | None,
                           region: str) -> torch.Tensor:
    """Boolean [B, S] mask selecting attended positions in the requested region,
    split at the first [SEP] (text | keyphrase/[MASK] segment).

    region:
        "full"   -> every attended position (current behaviour)
        "before" -> positions before the first [SEP] (text side, [SEP] excluded)
        "after"  -> positions after the first [SEP] (segment side, that [SEP] excluded)
    """
    am = attention_mask.bool()
    if region == "full":
        return am
    if sep_token_id is None:
        raise ValueError(f"sep_token_id is required for region '{region}'.")

    is_sep = input_ids == sep_token_id
    sep_rank = torch.cumsum(is_sep.long(), dim=1)  # 0 before 1st [SEP], >=1 from it onward
    if region == "before":
        region_mask = sep_rank == 0
    elif region == "after":
        first_sep = is_sep & (sep_rank == 1)
        region_mask = (sep_rank >= 1) & (~first_sep)
    else:
        raise ValueError(f"Unknown region '{region}'. Expected 'full', 'before', or 'after'.")
    return region_mask & am


def _pool_cspr_partitions(_out: dict[str, torch.Tensor],
                          inputs: BatchEncoding,
                          pooling_method: dict[str, str],
                          pooling_segments: dict[str, str] | None,
                          sep_token_id: int | None) -> dict[str, torch.Tensor]:
    out = {}
    for partition_type in _out:
        if partition_type not in pooling_method:
            raise KeyError(f"Partition '{partition_type}' has no entry in pooling_method. "
                           f"Available keys: {list(pooling_method.keys())}")
        region = (pooling_segments or {}).get(partition_type, "full")
        mask = _segment_position_mask(
            inputs["input_ids"], inputs["attention_mask"], sep_token_id, region
        )
        out[partition_type] = _apply_pooling_cspr(
            inp=_out[partition_type],
            attention_mask=mask.to(inputs["attention_mask"].dtype),
            pm=pooling_method[partition_type],
        )
    return out



class BaseSparse(torch.nn.Module, ABC):
    def __init__(self, 
                 model_type_or_dir: str, 
                 model_class: type[AutoModelForMaskedLM] | None = AutoModelForMaskedLM,
                 bf16: bool = False):
        super().__init__()

        self.transformer = model_class.from_pretrained(
            model_type_or_dir,
            # torch_dtype=torch.bfloat16 if bf16 else torch.float32
            torch_dtype = torch.float32
        )
        self.bf16 = bf16

    def forward(self, **inputs: torch.Tensor):
        out = self.transformer(**inputs)
        return out





class BaseSiamese(torch.nn.Module, ABC):
    def __init__(self, 
                 model_type_or_dir: str, 
                 model_class: type[AutoModelForMaskedLM] | None = AutoModelForMaskedLM,
                 bf16: bool = False):
        super().__init__()

        self.sparse_model = BaseSparse(
            model_type_or_dir = model_type_or_dir,
            model_class = model_class,
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
        "q_rep"    => a precomputed query representation to reuse instead of
                      re-encoding "q_kwargs" (e.g. scoring the same query against
                      a second batch of documents).
        """

        out = {}
        do_d, do_q = "d_kwargs" in kwargs, "q_kwargs" in kwargs
        if do_d:
            d_rep = self.encode(kwargs["d_kwargs"])
            out.update({"d_rep": d_rep})
        if do_q:
            q_rep = self.encode(kwargs["q_kwargs"])
            out.update({"q_rep": q_rep})
        elif "q_rep" in kwargs:
            q_rep = kwargs["q_rep"]
            out.update({"q_rep": q_rep})

        have_q = do_q or ("q_rep" in kwargs)
        if do_d and have_q:
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
                out["score"] = sum(out[f"score_{partition_name}"] for partition_name in q_rep)

        return out
    



class CSpRTrain(BaseSiamese):
    """CSpR class, this is supposed to be used for training"""

    def __init__(self,
               model_type_or_dir: str,
               vocab_partitions: dict[str, list[int]],
               model_class: type[AutoModelForMaskedLM] | None = AutoModelForMaskedLM,
               bf16: bool = False,
               pooling_method: dict[str, str] | None = None,
               pooling_segments: dict[str, str] | None = None,
               sep_token_id: int | None = None):
        super().__init__(
            model_type_or_dir = model_type_or_dir,
            model_class = model_class,
            bf16 = bf16
        )

        self.sparse_model.transformer = PartitionedMaskedLM(
            self.sparse_model.transformer, vocab_partitions
        )
        self.pooling_method = pooling_method if pooling_method is not None else {"token": "max", "phrase": "sum"}
        # None -> pool over the full sequence (original behaviour); otherwise map
        # each partition to "before"/"after"/"full" relative to the first [SEP].
        self.pooling_segments = pooling_segments
        self.sep_token_id = sep_token_id

    def encode(self, inputs: BatchEncoding):
        _out = self.encode_(inputs)
        return _pool_cspr_partitions(
            _out, inputs, self.pooling_method, self.pooling_segments, self.sep_token_id
        )
    



class CSpR(BaseSparse):
    """Also a CSpR class, but this is for inference"""
    def __init__(self,
                 model_type_or_dir: str,
                 vocab_partitions: dict[str, list[int]],
                 model_class: type[AutoModelForMaskedLM] | None = AutoModelForMaskedLM,
                 bf16: bool = False,
                 pooling_method: dict[str, str] | None = None,
                 pooling_segments: dict[str, str] | None = None,
                 sep_token_id: int | None = None):
        super().__init__(
            model_type_or_dir=model_type_or_dir,
            model_class=model_class,
            bf16=bf16
        )

        self.transformer = PartitionedMaskedLM(self.transformer, vocab_partitions)
        self.pooling_method = pooling_method if pooling_method is not None else {"token": "max", "phrase": "sum"}
        self.pooling_segments = pooling_segments
        self.sep_token_id = sep_token_id
        self.eval()

    def encode(self, inputs: BatchEncoding) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            _out = self.forward(**inputs)
        return _pool_cspr_partitions(
            _out, inputs, self.pooling_method, self.pooling_segments, self.sep_token_id
        )
    

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