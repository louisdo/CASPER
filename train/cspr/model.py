import math
import torch
import torch.nn.functional as F
from abc import ABC

from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          BatchEncoding)
from train.cspr.partitioned_maskedlm import PartitionedMaskedLM


# Partition that carries the fine-grained ranking signal; its score enters the
# combined score with weight 1 and every other partition is weighted by a
# learnable lambda. "token" is the default base partition for CSpR.
_BASE_PARTITION = "token"


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
                 bf16: bool = False,
                 trust_remote_code: bool = False):
        super().__init__()

        self.transformer = model_class.from_pretrained(
            model_type_or_dir,
            # torch_dtype=torch.bfloat16 if bf16 else torch.float32
            torch_dtype = torch.float32,
            trust_remote_code = trust_remote_code
        )
        self.bf16 = bf16

    def forward(self, **inputs: torch.Tensor):
        out = self.transformer(**inputs)
        return out





class BaseSiamese(torch.nn.Module, ABC):
    def __init__(self,
                 model_type_or_dir: str,
                 model_class: type[AutoModelForMaskedLM] | None = AutoModelForMaskedLM,
                 bf16: bool = False,
                 trust_remote_code: bool = False):
        super().__init__()

        self.sparse_model = BaseSparse(
            model_type_or_dir = model_type_or_dir,
            model_class = model_class,
            bf16 = bf16,
            trust_remote_code = trust_remote_code
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
    



class _CombinedLambda(torch.nn.Module):
    """Owns the learnable per-partition weight lambda used to form the combined
    score ``score_base + sum_p lambda_p * score_p`` (base partition has weight 1).

    lambda is stored as an unconstrained ``logit_scale``-style parameter and
    passed through softplus so the effective weight is always >= 0. Initialised
    so that the starting effective lambda equals ``init_lambda`` (default 0.25,
    matching the long-standing eval-time phrase weight).
    """

    def __init__(self, partition_names: list[str], init_lambda: float = 0.25,
                 base_partition: str = _BASE_PARTITION):
        super().__init__()
        self.base_partition = base_partition
        self.weighted_partitions = [p for p in partition_names if p != base_partition]
        # softplus(raw) = init_lambda  =>  raw = log(exp(init_lambda) - 1)
        raw_init = math.log(math.expm1(init_lambda)) if init_lambda > 0 else -10.0
        self.lambda_raw = torch.nn.ParameterDict(
            {p: torch.nn.Parameter(torch.tensor(float(raw_init))) for p in self.weighted_partitions}
        )

    def weight(self, partition_name: str) -> torch.Tensor | float:
        if partition_name == self.base_partition:
            return 1.0
        return F.softplus(self.lambda_raw[partition_name])

    def current_lambdas(self) -> dict[str, float]:
        return {p: F.softplus(raw).item() for p, raw in self.lambda_raw.items()}

    def combine(self, partition_scores: dict[str, torch.Tensor]) -> torch.Tensor:
        # weighted sum of per-partition scores into a single combined score
        total = None
        for name, score in partition_scores.items():
            term = self.weight(name) * score
            total = term if total is None else total + term
        return total

    def scale_reps(self, rep: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Fold lambda into the (non-base) partition representations so that a
        # plain dot product of partitions downstream reproduces the combined
        # score. Both the query and the document rep pass through here, so the
        # retrieval dot product (q_p . d_p) squares the per-rep factor -- scale
        # each rep by sqrt(lambda) so that sqrt(lambda)^2 == lambda on the score.
        return {
            name: (vec if name == self.base_partition else (self.weight(name) ** 0.5) * vec)
            for name, vec in rep.items()
        }


class CSpRTrain(BaseSiamese):
    """CSpR class, this is supposed to be used for training"""

    def __init__(self,
               model_type_or_dir: str,
               vocab_partitions: dict[str, list[int]],
               model_class: type[AutoModelForMaskedLM] | None = AutoModelForMaskedLM,
               bf16: bool = False,
               pooling_method: dict[str, str] | None = None,
               pooling_segments: dict[str, str] | None = None,
               sep_token_id: int | None = None,
               trust_remote_code: bool = False):
        super().__init__(
            model_type_or_dir = model_type_or_dir,
            model_class = model_class,
            bf16 = bf16,
            trust_remote_code = trust_remote_code
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



class CSpRCombinedTrain(CSpRTrain):
    """CSpR trained on a single combined score ``score_token + lambda * score_phrase``
    with a learnable lambda (per non-base partition).

    Identical to :class:`CSpRTrain` in how documents/queries are encoded and pooled;
    the only difference is that the ``"score"`` field returned by ``forward`` is the
    lambda-weighted combination instead of the plain unweighted partition sum. The
    per-partition ``score_<partition>`` fields are still emitted, so matryoshka-style
    per-field losses keep working unchanged.
    """

    def __init__(self, *args, init_lambda: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.combiner = _CombinedLambda(
            partition_names=self.sparse_model.transformer.partition_names,
            init_lambda=init_lambda,
        )

    def forward(self, **kwargs):
        out = super().forward(**kwargs)
        # super() already produced per-partition score_<p> (when both sides given)
        # and an unweighted "score". Replace "score" with the lambda-weighted combine.
        if "score" in out:
            partition_scores = {
                name: out[f"score_{name}"]
                for name in self.sparse_model.transformer.partition_names
                if f"score_{name}" in out
            }
            out["score"] = self.combiner.combine(partition_scores)
        return out

    def current_lambdas(self) -> dict[str, float]:
        return self.combiner.current_lambdas()


class CSpR(BaseSparse):
    """Also a CSpR class, but this is for inference"""
    def __init__(self,
                 model_type_or_dir: str,
                 vocab_partitions: dict[str, list[int]],
                 model_class: type[AutoModelForMaskedLM] | None = AutoModelForMaskedLM,
                 bf16: bool = False,
                 pooling_method: dict[str, str] | None = None,
                 pooling_segments: dict[str, str] | None = None,
                 sep_token_id: int | None = None,
                 trust_remote_code: bool = False):
        super().__init__(
            model_type_or_dir=model_type_or_dir,
            model_class=model_class,
            bf16=bf16,
            trust_remote_code=trust_remote_code
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


class CSpRCombined(CSpR):
    """Inference counterpart of :class:`CSpRCombinedTrain`.

    Folds the learned lambda into the (non-base) partition representations at
    encode time, so any downstream code that simply sums partition scores
    (e.g. the dot product in retrieval, or :func:`encode_sparse_batch`) reproduces
    ``score_token + lambda * score_phrase`` without a manual per-partition weight.

    Load ``lambda_raw`` from the trained checkpoint via ``set_lambdas`` (or by
    loading the saved combiner state) so eval matches training exactly. If left
    unset, lambdas default to ``init_lambda``.
    """

    # signals to downstream sparse-encoding code that lambda is already baked into
    # the partition reps, so no further per-partition weighting should be applied.
    lambda_in_rep = True

    def __init__(self, *args, init_lambda: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.combiner = _CombinedLambda(
            partition_names=self.transformer.partition_names,
            init_lambda=init_lambda,
        )
        self.combiner.eval()

    def set_lambdas(self, lambdas: dict[str, float]):
        """Set effective (post-softplus) lambda values, inverting softplus to raw."""
        for name, value in lambdas.items():
            if name not in self.combiner.lambda_raw:
                continue
            raw = math.log(math.expm1(value)) if value > 0 else -10.0
            with torch.no_grad():
                self.combiner.lambda_raw[name].fill_(float(raw))

    def encode(self, inputs: BatchEncoding) -> dict[str, torch.Tensor]:
        rep = super().encode(inputs)
        return self.combiner.scale_reps(rep)


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