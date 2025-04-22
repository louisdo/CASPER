import gin
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics import KeyphraseMetric
from metrics.metric_utils import separate_present_absent_by_source, filter_prediction, find_unique_target
from sentence_transformers import util, SentenceTransformer

from nltk.stem.porter import *


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    attention_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]


def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]


@gin.configurable
class SemanticMatchingMetric(KeyphraseMetric):
    def __init__(self, model_name_or_path, similarity_threshold, pooling_across_phrases):
        """
        Semantic matching metric (based on sentence transformers)
        """
        super(SemanticMatchingMetric, self).__init__()
        
        self.model = SentenceTransformer(model_name_or_path)
        self.similarity_threshold = similarity_threshold
        self.pooling_across_phrases = pooling_across_phrases
        assert self.pooling_across_phrases in ['mean', 'max']
        
    def evaluate_single_example(self, preds, preds_stemmed, labels, labels_stemmed):
        n_labels, n_preds = len(labels), len(preds)
        input_tokens = labels + preds
        
        phrase_embed_mean_pool = self.model.encode(input_tokens)
        label_embeds = phrase_embed_mean_pool[:n_labels]
        pred_embeds = phrase_embed_mean_pool[n_labels:]

        ###########################################################
        # SemP for each prediction phrase
        ###########################################################
        if n_labels == 0 or n_preds == 0:
            cur_p = 0
        else:
            all_cos_sim = util.cos_sim(pred_embeds, label_embeds)
            top_sim_values, top_sim_indices = torch.topk(all_cos_sim, min(3, n_labels))
            match_results = {}
            for pred_i in range(n_preds):
                cur_pred, cur_pred_stemmed = preds[pred_i], preds_stemmed[pred_i]
                
                cur_support_label = None
                # if exact match, then give a score of 1
                if cur_pred_stemmed in labels_stemmed:
                    cur_pred_score = 1
                elif top_sim_values[pred_i][0] > self.similarity_threshold:
                    cur_pred_score = top_sim_values[pred_i][0].item()
                    cur_support_label = labels[top_sim_indices[pred_i][0]]
                else:
                    cur_pred_score = 0
                match_results[cur_pred] = [cur_pred_score, cur_support_label]
            cur_p = np.mean([x[0] for x in match_results.values()]).item()

        ###########################################################
        # SemR for each prediction phrase
        ###########################################################
        if n_labels == 0 or n_preds == 0:
            cur_r = 0
        else:
            all_cos_sim = util.cos_sim(label_embeds, pred_embeds)
            top_sim_values, top_sim_indices = torch.topk(all_cos_sim, min(3, n_preds))
            match_results = {}
            for label_i in range(n_labels):
                cur_label, cur_label_stemmed = labels[label_i], labels_stemmed[label_i]
                
                cur_support_pred = None
                # if exact match, then give a score of 1
                if labels_stemmed in preds_stemmed:
                    cur_label_score = 1
                elif top_sim_values[label_i][0] > self.similarity_threshold:
                    cur_label_score = top_sim_values[label_i][0].item()
                    cur_support_pred = preds[top_sim_indices[label_i][0]]
                else:
                    cur_label_score = 0
                match_results[cur_label] = [cur_label_score, cur_support_pred]
            cur_r = np.mean([x[0] for x in match_results.values()]).item()

        cur_f1 = 0 if cur_p * cur_r == 0 else 2 * cur_p * cur_r / (cur_p + cur_r)
        # cur_method_sem_f1.append('{:.4f}'.format(cur_f1))
        
        return {'p': cur_p, 'r': cur_r, 'f1': cur_f1}

    def score_corpus(self, all_preds, all_refs, all_inputs):
        source, target, preds = all_inputs, all_refs, all_preds
        
        predicted_keyphrases = []
        score_dict = {}        
        for m in ['p', 'r', 'f1']:
            score_dict[f'present_semantic_{m}'] = []
            score_dict[f'absent_semantic_{m}'] = []
        
        for data_idx, (src_l, trg_l, pred_l) in \
                enumerate(tqdm(zip(source, target, preds),
                            total=len(source), desc='Evaluating...')):
                        
            src_token_list, stemmed_src_token_list = src_l 
            pred_str_list, pred_token_2dlist, stemmed_pred_token_2dlist = pred_l
            trg_str_list, trg_token_2dlist, stemmed_trg_token_2dlist = trg_l
            num_predictions = len(pred_str_list)

            # Filter out duplicate or invalid predictions
            filtered_stemmed_pred_token_2dlist, num_duplicated_predictions, is_unique_mask = filter_prediction(
                self.invalidate_unk, stemmed_pred_token_2dlist, self.unk_word
            )
            num_filtered_predictions = len(filtered_stemmed_pred_token_2dlist)

            # Remove duplicated targets
            unique_stemmed_trg_token_2dlist, num_duplicated_trg = find_unique_target(stemmed_trg_token_2dlist)
            num_unique_targets = len(unique_stemmed_trg_token_2dlist)

            # separate present and absent keyphrases
            present_filtered_stemmed_pred_token_2dlist, absent_filtered_stemmed_pred_token_2dlist, is_present_mask = \
                separate_present_absent_by_source(stemmed_src_token_list, filtered_stemmed_pred_token_2dlist, False)
            present_unique_stemmed_trg_token_2dlist, absent_unique_stemmed_trg_token_2dlist, is_present_mask_gt = \
                separate_present_absent_by_source(stemmed_src_token_list, unique_stemmed_trg_token_2dlist, False)
                
            # save the predicted keyphrases
            filtered_pred_token_2dlist = [kp_tokens for kp_tokens, is_unique
                                          in zip(pred_token_2dlist, is_unique_mask) if is_unique]
            result = {'id': data_idx, 'present': [], 'absent': [], 'present_ref': [], 'absent_ref': []}
            for kp_tokens, is_present in zip(filtered_pred_token_2dlist, is_present_mask):
                if is_present:
                    result['present'] += [' '.join(kp_tokens)]
                else:
                    result['absent'] += [' '.join(kp_tokens)]
            
            filtered_pred_token_2dlist = [kp_tokens for kp_tokens in trg_token_2dlist]
            for kp_tokens, is_present in zip(trg_token_2dlist, is_present_mask_gt):
                if is_present:
                    result['present_ref'] += [' '.join(kp_tokens)]
                else:
                    result['absent_ref'] += [' '.join(kp_tokens)]
            predicted_keyphrases.append(result)
            
            present_preds_stemmed = [' '.join(x) for x in present_filtered_stemmed_pred_token_2dlist]
            assert len(result['present']) == len(present_preds_stemmed)
            absent_preds_stemmed = [' '.join(x) for x in absent_filtered_stemmed_pred_token_2dlist]
            assert len(result['absent']) == len(absent_preds_stemmed)
            present_labels_stemmed = [' '.join(x) for x in present_unique_stemmed_trg_token_2dlist]
            assert len(result['present_ref']) == len(present_labels_stemmed)
            absent_labels_stemmed = [' '.join(x) for x in absent_unique_stemmed_trg_token_2dlist]
            assert len(result['absent_ref']) == len(absent_labels_stemmed)
            
            # calculate semantic matching scores
            all_pkp_scores = self.evaluate_single_example(result['present'],
                                                        present_preds_stemmed,
                                                        result['present_ref'],
                                                        present_labels_stemmed)
            all_akp_scores = self.evaluate_single_example(result['absent'],
                                                        absent_preds_stemmed,
                                                        result['absent_ref'],
                                                        absent_labels_stemmed)
            for m in ['p', 'r', 'f1']:
                score_dict[f'present_semantic_{m}'].append(all_pkp_scores[m])
                score_dict[f'absent_semantic_{m}'].append(all_akp_scores[m])
            
        return score_dict
