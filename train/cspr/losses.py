import math
import torch

class InBatchPairwiseNLL:

    def __init__(self):
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def __call__(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        nb_columns = pos_score.shape[1]               # chunk = bs / nb_gpus
        nb_gpus = int(pos_score.shape[0] / nb_columns)

        scores = torch.cat([pos_score, neg_score], dim=1)  # (bs, chunk + 1)
        scores = self.logsoftmax(scores.float())

        rows = torch.arange(pos_score.shape[0], device=pos_score.device)
        labels = torch.arange(nb_columns, device=pos_score.device).repeat(nb_gpus)
        return torch.mean(-scores[rows, labels])


# temperature is clamped to this range so the learnable scale can sharpen as the
# reps sparsify (down toward ~1) but can neither over-sharpen below 1 (which would
# bring back the large-magnitude / bf16 precision problem) nor flatten past 256.
_MIN_TEMPERATURE = 1.0
_MAX_TEMPERATURE = 256.0


class MatryoshkaContrastiveLoss(torch.nn.Module):
    def __init__(self, field_names: tuple[str, ...] = ("score", "score_phrase"),
                 temperature: float = 1.0, learnable_temperature: bool = False,
                 neg_usage: tuple[bool, ...] | None = None):
        super().__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.field_names = field_names
        if neg_usage is not None and len(neg_usage) != len(field_names):
            raise ValueError(
                f"neg_usage has length {len(neg_usage)} but expected {len(field_names)} "
                f"to match field_names {field_names}"
            )
        
        self.neg_usage = neg_usage
        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            self.logit_scale = torch.nn.Parameter(torch.tensor(-math.log(temperature)))
            self.temperature = None
        else:
            self.register_parameter("logit_scale", None)
            self.temperature = temperature

    def _inv_temperature(self) -> torch.Tensor | float:
        if self.learnable_temperature:
            return self.logit_scale.clamp(
                min=-math.log(_MAX_TEMPERATURE), max=-math.log(_MIN_TEMPERATURE)
            ).exp()
        return 1.0 / self.temperature

    def current_temperature(self) -> float:
        inv = self._inv_temperature()
        inv = inv.item() if torch.is_tensor(inv) else inv
        return 1.0 / inv

    def helper(self, out_d: dict[str, torch.Tensor], field_name: str, use_hardneg: bool = True):
        in_batch_scores, neg_scores = out_d[f"pos_{field_name}"], out_d[f"neg_{field_name}"]
        nb_columns = in_batch_scores.shape[1]
        nb_gpus = int(in_batch_scores.shape[0] / nb_columns)

        if use_hardneg:
            temp = torch.cat([in_batch_scores, neg_scores], dim=1)  # concat neg score from BM25 sampling
            # shape (batch_size, batch_size/nb_gpus + 1)
        else:
            zero_neg_scores = torch.zeros_like(neg_scores)
            temp = torch.cat([in_batch_scores, zero_neg_scores], dim=1) # shape (batch_size, batch_size/nb_gpus + 1)

        scores = self.logsoftmax(temp.float() * self._inv_temperature())

        rows = torch.arange(in_batch_scores.shape[0], device=in_batch_scores.device)
        labels = torch.arange(nb_columns, device=in_batch_scores.device).repeat(nb_gpus)
        return torch.mean(-scores[rows, labels])

    def use_hardneg_for(self, field_name: str) -> bool:
        """Whether the hard negative should be included in `field_name`'s loss."""
        if self.neg_usage is None:
            return True
        return self.neg_usage[self.field_names.index(field_name)]

    def forward(self, out_d, field_names: tuple[str, ...] | None = None,
                neg_usage: tuple[bool, ...] | None = None):
        field_names = field_names if field_names is not None else self.field_names
        if neg_usage is None:
            neg_usage = self.neg_usage
        if neg_usage is not None and len(neg_usage) != len(field_names):
            raise ValueError(
                f"neg_usage has length {len(neg_usage)} but expected {len(field_names)} "
                f"to match field_names {field_names}"
            )
        loss = None
        for i, field_name in enumerate(field_names):
            use_hardneg = neg_usage[i] if neg_usage is not None else True
            val = self.helper(out_d, field_name, use_hardneg=use_hardneg)
            loss = val if loss is None else loss + val
        return loss
    


class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)