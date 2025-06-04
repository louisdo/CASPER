import torch

ORIGINAL_BERT_VOCAB_SIZE = 30522

class L1:

    def __call__(self, batch_rep):
        return torch.sum(torch.abs(batch_rep), dim=-1).mean()
    

class L2:

    def __call__(self, batch_rep):
        return torch.sum(batch_rep ** 2, dim=-1).mean()


class L0:
    """non-differentiable
    """

    def __call__(self, batch_rep):
        return torch.count_nonzero(batch_rep, dim=-1).float().mean()


class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)
    
class FLOPSPhrase:
    def __init__(self, phrase_reg_magnifier = 5):
        self.phrase_reg_magnifier = phrase_reg_magnifier

    def __call__(self, batch_rep):
        batch_rep_tokens = batch_rep[...,:ORIGINAL_BERT_VOCAB_SIZE]
        batch_rep_phrases = batch_rep[..., ORIGINAL_BERT_VOCAB_SIZE:]

        tokens_reg = torch.sum(torch.mean(torch.abs(batch_rep_tokens), dim=0) ** 2)
        phrases_reg = torch.sum(torch.mean(torch.abs(batch_rep_phrases), dim=0) ** 2)

        return tokens_reg + self.phrase_reg_magnifier * phrases_reg

class FLOPSPhrasev2:
    def __call__(self, batch_rep):
        batch_rep_tokens = batch_rep[...,:ORIGINAL_BERT_VOCAB_SIZE]
        batch_rep_phrases = batch_rep[..., ORIGINAL_BERT_VOCAB_SIZE:]

        tokens_reg = torch.sum(torch.mean(torch.abs(batch_rep_tokens), dim=0) ** 2)
        phrases_reg = torch.sum(torch.mean(torch.abs(batch_rep_phrases), dim=0) ** 2)

        scale = batch_rep_tokens.size(-1) / batch_rep_phrases.size(-1)

        return tokens_reg + scale * phrases_reg

class RegWeightScheduler:
    """same scheduling as in: Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __init__(self, lambda_, T):
        self.lambda_ = lambda_
        self.T = T
        self.t = 0
        self.lambda_t = 0

    def step(self):
        """quadratic increase until time T
        """
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.lambda_t = self.lambda_ * (self.t / self.T) ** 2
        return self.lambda_t

    def get_lambda(self):
        return self.lambda_t


class SparsityRatio:
    """non-differentiable
    """

    def __init__(self, output_dim):
        self.output_dim = output_dim

    def __call__(self, batch_rep):
        return 1 - torch.sum(batch_rep != 0, dim=-1).float().mean() / self.output_dim


def init_regularizer(reg, **kwargs):
    if reg == "L0":
        return L0()
    elif reg == "sparsity_ratio":
        return SparsityRatio(output_dim=kwargs["output_dim"])
    elif reg == "L1":
        return L1()
    elif reg == "FLOPS":
        return FLOPS()
    elif reg =="FLOPSPhrasev2":
        return FLOPSPhrasev2()
    elif reg.startswith("FLOPSPhrase--"):
        phrase_reg_magnifier = int(reg.replace("FLOPSPhrase--", ""))
        return FLOPSPhrase(phrase_reg_magnifier=phrase_reg_magnifier)
    elif reg == "L2":
        return L2()
    else:
        raise NotImplementedError("provide valid regularizer")
