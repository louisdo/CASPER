import gin
import numpy as np
from tqdm import tqdm

from metrics import KeyphraseMetric
from metrics.metric_utils import separate_present_absent_by_source, filter_prediction, find_unique_target

from nltk.stem.porter import *


@gin.configurable
class ApproximateMatchingMetric(KeyphraseMetric):
    def __init__(self):
        """
        Approximate matching metric
        - R-precision with "includes" and "part-of" relationship (https://aclanthology.org/R09-1086.pdf)
        - Precision, Recall, and F1 with "includes" and "part-of" relationship (i.e., substring matching).
        """
        super(ApproximateMatchingMetric, self).__init__()
        
    def evaluate_single_example(self, preds, preds_stemmed, labels, labels_stemmed):
        # P, R, and F1 with substring matching
        if len(preds_stemmed) == 0:
            substring_precision = 0
        else:
            substring_precision = len([x for x in preds_stemmed if any([y in x for y in labels_stemmed]) \
                or any([x in y for y in labels_stemmed])]) / len(preds_stemmed)
            
        if len(labels_stemmed) == 0:
            substring_recall = 0
        else:
            substring_recall = len([x for x in labels_stemmed if any([y in x for y in preds_stemmed]) \
                or any([x in y for y in preds_stemmed])]) / len(labels_stemmed)
            
        substring_f1 = 2 * substring_precision * substring_recall / (substring_precision + substring_recall) if substring_precision * substring_recall != 0 else 0.0
        
        # R-precision
        preds_stemmed = preds_stemmed[:len(labels_stemmed)]
        if len(preds_stemmed) == 0:
            r_precision = 0
        else:
            # Note: here we always use len(labels) as the denominator. 
            # In other words, when len(preds_stemmed) < len(labels_stemmed), we calculate the score as if 
            # random false keyphrases are appended to preds_stemmed.
            r_precision = len([x for x in preds_stemmed if any([y in x for y in labels_stemmed]) \
                or any([x in y for y in labels_stemmed])]) / len(labels_stemmed)
        return {'r-precision': r_precision, 
                'substring_precision': substring_precision, 
                'substring_recall': substring_recall, 
                'substring_f1': substring_f1}

    def score_corpus(self, all_preds, all_refs, all_inputs):
        source, target, preds = all_inputs, all_refs, all_preds
        
        predicted_keyphrases = []
        score_dict = {}        
        for subset in ['present', 'absent', 'all']:
            for m in ['r-precision', 'substring_precision', 'substring_recall', 'substring_f1']:
                score_dict[f'{subset}_{m}'] = []
        
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
            
            # calculate approximate matching scores
            pkp_scores = self.evaluate_single_example(result['present'], present_preds_stemmed, 
                                                          result['present_ref'], present_labels_stemmed)
            akp_scores = self.evaluate_single_example(result['absent'], absent_preds_stemmed, 
                                                          result['absent_ref'], absent_labels_stemmed)
            allkp_scores = self.evaluate_single_example(result['present'] + result['absent'],
                                                            present_preds_stemmed + absent_preds_stemmed,
                                                            result['present_ref'] + result['absent_ref'],
                                                            present_labels_stemmed + absent_labels_stemmed)
            
            for m in ['r-precision', 'substring_precision', 'substring_recall', 'substring_f1']:
                score_dict[f'present_{m}'].append(pkp_scores[m])
                score_dict[f'absent_{m}'].append(akp_scores[m])
                score_dict[f'all_{m}'].append(allkp_scores[m])
            
        return score_dict
