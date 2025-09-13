# based on https://github.com/BorealisAI/keyphrase-generation/blob/main/keyphrase_generation/evaluation/diversity_eval.py
import gin
from typing import List
from scipy import sparse
import numpy as np
from tqdm import tqdm

from metrics import KeyphraseMetric

import nltk
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction

from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer


@gin.configurable
class DiversityMetric(KeyphraseMetric):
    def __init__(self, ngram=3, sbert_model=None, batch_size=32):
        """
        Metrics for calculating diversity (based on https://arxiv.org/abs/2010.07665)
        - Unique phrase ratio
        - Unique token ratio
        - Self-BLEU (https://arxiv.org/abs/1802.01886)
        - Edit distance
        - Pairwise embedding similarity
        """
        super(DiversityMetric, self).__init__()
        self.ngram = ngram
        self.batch_size = batch_size
        
        self.sbert_model = None
        if sbert_model:
            self.sbert_model = SentenceTransformer(sbert_model)

    def calc_bleu(self, references, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def calc_selfbleu(self, kps: List[str]):
        if len(kps) <= 1:
            return 0.0
        
        weights = tuple((1. / self.ngram for _ in range(self.ngram)))
        
        bleu_list = []
        for i in range(len(kps)):
            hypothesis = word_tokenize(kps[i])        
            references = kps[:i] + kps[i+1:]
            references = [word_tokenize(r) for r in references]
            bleu_list.append(self.calc_bleu(references, hypothesis, weights))
            
        return np.mean(bleu_list)

    def calc_edit_distance(self, kps: List[str]):
        if len(kps) <= 1:
            return 0.0
        
        edit_dists = []
        for i in range(len(kps)):
            source = kps[i]
            targets = kps[:i] + kps[i+1:]
            target_dists = []
            for t in targets:
                target_dists.append(fuzz.ratio(source, t))
            edit_dists.append(np.mean(target_dists))
        
        return np.mean(edit_dists)
    
    def calc_embed_similarity(self, kps: List[str], model):
        if len(kps) == 0:
            return 0.0
        elif len(kps) == 1:
            return 1.0
        
        n_kps = len(kps)
        if type(model) == SentenceTransformer:
            embs = model.encode(kps, batch_size=self.batch_size)
        else:
            raise NotImplementedError
        
        embs_sparse = sparse.csr_matrix(embs)
        similarities = cosine_similarity(embs_sparse)
        avg_emb_sim = np.mean(similarities[np.triu_indices(n_kps, k=1)])

        return avg_emb_sim
    
    def calc_sem_diverse_index(self, kps: List[str], model):
        if len(kps) == 0:
            return 0.0
        
        if type(model) == SentenceTransformer:
            embs = model.encode(kps, batch_size=self.batch_size)
        else:
            raise NotImplementedError
        
        embs_orig = np.array(embs)
        embs_orig = embs_orig[np.nonzero(embs_orig.sum(axis=1))]
        embs = embs_orig / np.linalg.norm(embs_orig, axis=1, keepdims=True)   # normalize
        sims = embs.dot(embs.T)
        diverse_index = np.linalg.norm(sims * (np.ones_like(sims) - np.identity(sims.shape[0])))

        return diverse_index
    
    def score_corpus(self, all_preds, all_refs, all_inputs):
        source, target, preds = all_inputs, all_refs, all_preds
        
        score_dict = {}        
        for m in ['unique_phrase_ratio', 'unique_token_ratio', 'dup_token_ratio', 'self_bleu', 'self_edit_distance']:
            score_dict[f'{m}'] = []
        if self.sbert_model:
            score_dict["self_embed_similarity_sbert"] = []
            score_dict["sem_diverse_index_sbert"] = []
            
        for data_idx, (src_l, trg_l, pred_l) in \
                enumerate(tqdm(zip(source, target, preds),
                            total=len(source), desc='Evaluating...')):
                        
            src_token_list, stemmed_src_token_list = src_l 
            pred_str_list, pred_token_2dlist, stemmed_pred_token_2dlist = pred_l
            trg_str_list, trg_token_2dlist, stemmed_trg_token_2dlist = trg_l
            num_predictions = len(pred_str_list)
            
            stemmed_pred_phrase_list = [' '.join(x) for x in stemmed_pred_token_2dlist]
            stemmed_pred_token_list = [y for x in stemmed_pred_token_2dlist for y in x]
            
            if len(stemmed_pred_token_2dlist) == 0 or len(stemmed_pred_phrase_list) == 0 or len(stemmed_pred_token_list) == 0:
                score_dict['unique_phrase_ratio'].append(0)
                score_dict['unique_token_ratio'].append(0)
                score_dict['dup_token_ratio'].append(1)
                score_dict['self_bleu'].append(0)
                score_dict['self_edit_distance'].append(0)
                if self.sbert_model:
                    score_dict['self_embed_similarity_sbert'].append(0)
            else:
                # Unique phrase and token ratio
                score_dict['unique_phrase_ratio'].append(len(set(stemmed_pred_phrase_list)) / len(stemmed_pred_phrase_list))
                score_dict['unique_token_ratio'].append(len(set(stemmed_pred_token_list)) / len(stemmed_pred_token_list))
                score_dict['dup_token_ratio'].append(1 - (len(set(stemmed_pred_token_list)) / len(stemmed_pred_token_list)))

                # Self-BLEU
                score_dict['self_bleu'].append(self.calc_selfbleu(stemmed_pred_phrase_list))

                # Edit Distance 
                score_dict['self_edit_distance'].append(self.calc_edit_distance(stemmed_pred_phrase_list))
                
                # Semantic Similarity
                if self.sbert_model:
                    score_dict['self_embed_similarity_sbert'].append(self.calc_embed_similarity(pred_str_list, self.sbert_model))
                    score_dict['sem_diverse_index_sbert'].append(self.calc_sem_diverse_index(pred_str_list, self.sbert_model))

        return score_dict
