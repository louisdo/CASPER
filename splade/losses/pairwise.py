import torch

"""general API for losses: the __call__ method receives out_d, a dict containing at least scores for positives 
and negatives  
"""


class PairwiseNLL:
    def __init__(self):
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def __call__(self, out_d):
        pos_scores, neg_scores = out_d["pos_score"], out_d["neg_score"]
        scores = self.logsoftmax(torch.cat([pos_scores, neg_scores], dim=1))
        return torch.mean(-scores[:, 0])


class InBatchPairwiseNLL:
    """in batch negatives version
    """

    def __init__(self):
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def __call__(self, out_d):
        in_batch_scores, neg_scores = out_d["pos_score"], out_d["neg_score"]
        # here in_batch_scores is a matrix of size bs * (bs / nb_gpus)
        nb_columns = in_batch_scores.shape[1]
        nb_gpus = int(in_batch_scores.shape[0] / nb_columns)
        temp = torch.cat([in_batch_scores, neg_scores], dim=1)  # concat neg score from BM25 sampling
        # shape (batch_size, batch_size/nb_gpus + 1)
        scores = self.logsoftmax(temp)
        return torch.mean(-scores[torch.arange(in_batch_scores.shape[0]),
                                  torch.arange(nb_columns).repeat(nb_gpus)])
    

class InBatchPairwiseNLLv2:
    """in batch negatives version
    """

    def __init__(self):
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def helper(self, out_d, field_name):
        in_batch_scores, neg_scores = out_d[f"pos_{field_name}"], out_d[f"neg_{field_name}"]
        # here in_batch_scores is a matrix of size bs * (bs / nb_gpus)
        nb_columns = in_batch_scores.shape[1]
        nb_gpus = int(in_batch_scores.shape[0] / nb_columns)
        temp = torch.cat([in_batch_scores, neg_scores], dim=1)  # concat neg score from BM25 sampling
        # shape (batch_size, batch_size/nb_gpus + 1)
        scores = self.logsoftmax(temp)
        return torch.mean(-scores[torch.arange(in_batch_scores.shape[0]),
                                  torch.arange(nb_columns).repeat(nb_gpus)])
    
    def __call__(self, out_d):
        loss_maxsim = self.helper(out_d=out_d, field_name="score")
        loss_normal = self.helper(out_d = out_d, field_name="score_normal")

        return (loss_maxsim + loss_normal) / 2
    

class InBatchPairwiseNLLPhraseSpladev2:
    """in batch negatives version
    """

    def __init__(self):
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def helper(self, out_d, field_name):
        in_batch_scores, neg_scores = out_d[f"pos_{field_name}"], out_d[f"neg_{field_name}"]
        # here in_batch_scores is a matrix of size bs * (bs / nb_gpus)
        nb_columns = in_batch_scores.shape[1]
        nb_gpus = int(in_batch_scores.shape[0] / nb_columns)
        temp = torch.cat([in_batch_scores, neg_scores], dim=1)  # concat neg score from BM25 sampling
        # shape (batch_size, batch_size/nb_gpus + 1)
        scores = self.logsoftmax(temp)
        return torch.mean(-scores[torch.arange(in_batch_scores.shape[0]),
                                  torch.arange(nb_columns).repeat(nb_gpus)])
    
    def __call__(self, out_d):
        loss = self.helper(out_d=out_d, field_name="score")
        loss_phrase = out_d["pos_l2_loss_phrase"]

        # TODO: The param 0.25 is arbitrary
        return loss + 0.25 * loss_phrase
    


class PairwiseBPR:
    """BPR loss from: http://webia.lip6.fr/~gallinar/gallinari/uploads/Teaching/WSDM2014-rendle.pdf
    """

    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def __call__(self, out_d):
        pos_scores, neg_scores = out_d["pos_score"], out_d["neg_score"]
        return self.loss((pos_scores - neg_scores).squeeze(), torch.ones(pos_scores.shape[0]).to(self.device))


class DistilMarginMSE:
    """MSE margin distillation loss from: Improving Efficient Neural Ranking Models with Cross-Architecture
    Knowledge Distillation
    link: https://arxiv.org/abs/2010.02666
    """

    def __init__(self):
        self.loss = torch.nn.MSELoss()

    def __call__(self, out_d):
        """out_d also contains scores from teacher
        """
        pos_scores, neg_scores = out_d["pos_score"], out_d["neg_score"]
        teacher_pos_scores, teacher_neg_scores = out_d["teacher_pos_score"], out_d["teacher_neg_score"]
        margin = pos_scores - neg_scores
        teacher_margin = teacher_pos_scores - teacher_neg_scores
        return self.loss(margin.squeeze(), teacher_margin.squeeze())  # forces the margins to be similar


class DistilMarginMSEv2:
    """MSE margin distillation loss from: Improving Efficient Neural Ranking Models with Cross-Architecture
    Knowledge Distillation
    link: https://arxiv.org/abs/2010.02666
    """

    def __init__(self):
        self.loss = torch.nn.MSELoss()

    def helper(self, out_d, field_name):
        """out_d also contains scores from teacher
        """
        pos_scores, neg_scores = out_d[f"pos_{field_name}"], out_d[f"neg_{field_name}"]
        teacher_pos_scores, teacher_neg_scores = out_d["teacher_pos_score"], out_d["teacher_neg_score"]
        margin = pos_scores - neg_scores
        teacher_margin = teacher_pos_scores - teacher_neg_scores
        return self.loss(margin.squeeze(), teacher_margin.squeeze())  # forces the margins to be similar
    
    def __call__(self, out_d):
        loss_maxsim = self.helper(out_d=out_d, field_name="score")
        loss_normal = self.helper(out_d=out_d, field_name="score_normal")

        return loss_maxsim + loss_normal



class DistilKLLoss:
    """Distillation loss from: Distilling Dense Representations for Ranking using Tightly-Coupled Teachers
    link: https://arxiv.org/abs/2010.11386
    """

    def __init__(self):
        self.loss = torch.nn.KLDivLoss(reduction="none")

    def __call__(self, out_d):
        pos_scores, neg_scores = out_d["pos_score"], out_d["neg_score"]
        teacher_pos_scores, teacher_neg_scores = out_d["teacher_pos_score"], out_d["teacher_neg_score"]
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        local_scores = torch.log_softmax(scores, dim=1)
        teacher_scores = torch.cat([teacher_pos_scores.unsqueeze(-1), teacher_neg_scores.unsqueeze(-1)], dim=1)
        teacher_scores = torch.softmax(teacher_scores, dim=1)
        return self.loss(local_scores, teacher_scores).sum(dim=1).mean(dim=0)
