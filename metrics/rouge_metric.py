import gin
from rouge import Rouge
import numpy as np
from tqdm import tqdm

from metrics import KeyphraseMetric
from metrics.metric_utils import separate_present_absent_by_source, filter_prediction, find_unique_target

from nltk.stem.porter import *


@gin.configurable
class RougeMetric(KeyphraseMetric):
    def __init__(self, n_list):
        super(RougeMetric, self).__init__()
        self.n_list = n_list
        self.rouge_score = Rouge()

    def score_corpus(self, all_preds, all_refs, all_inputs):
        source, target, preds = all_inputs, all_refs, all_preds
        
        predicted_keyphrases = []
        score_dict = {}
        for subset in ['present', 'absent', 'all']:
            for rouge_n in self.n_list:
                for score_type in ['p', 'r', 'f']:
                    score_dict[f'{subset}_rouge-{str(rouge_n).lower()}_{score_type}'] = []
        
        def __push_all_zero(subset):
            for rouge_n in self.n_list:
                for score_type in ['p', 'r', 'f']:
                    score_dict[f'{subset}_rouge-{str(rouge_n).lower()}_{score_type}'].append(0.0)
        
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
            
            # calculate rouge scores
            if ' '.join(result['present']).strip() == "" or ' '.join(result['present_ref']).strip() == "":
                __push_all_zero('present')
            else:
                # pkp_scores = self.rouge_score.get_scores(' {} '.format(self.kp_sep).join(result['present']), ' {} '.format(self.kp_sep).join(result['present_ref']))
                pkp_scores = self.rouge_score.get_scores(' '.join(result['present']), ' '.join(result['present_ref']))
                print(' '.join(result['present']), ' '.join(result['present_ref']), pkp_scores)
                assert len(pkp_scores) == 1
                for rouge_x, val_dict in pkp_scores[0].items():
                    for measure, val in val_dict.items():
                        score_dict[f'present_{rouge_x}_{measure}'].append(val)      
                
            if ' '.join(result['absent']).strip() == "" or ' '.join(result['absent_ref']).strip() == "":
                __push_all_zero('absent')
            else:
                # akp_scores = self.rouge_score.get_scores(' {} '.format(self.kp_sep).join(result['absent']), ' {} '.format(self.kp_sep).join(result['absent_ref']))
                akp_scores = self.rouge_score.get_scores(', '.join(result['absent']), ', '.join(result['absent_ref']))
                assert len(akp_scores) == 1
                for rouge_x, val_dict in akp_scores[0].items():
                    for measure, val in val_dict.items():
                        score_dict[f'absent_{rouge_x}_{measure}'].append(val)           
                        
            if ' '.join(result['present'] + result['absent']).strip() == "" or ' '.join(result['present_ref'] + result['absent_ref']).strip() == "":
                __push_all_zero('all')
            else:
                # allkp_scores = self.rouge_score.get_scores(' {} '.format(self.kp_sep).join(result['present'] + result['absent']), ' {} '.format(self.kp_sep).join(result['present_ref'] + result['absent_ref']))
                allkp_scores = self.rouge_score.get_scores(', '.join(result['present'] + result['absent']), ', '.join(result['present_ref'] + result['absent_ref']))
                assert len(allkp_scores) == 1
                for rouge_x, val_dict in allkp_scores[0].items():
                    for measure, val in val_dict.items():
                        score_dict[f'all_{rouge_x}_{measure}'].append(val)
            
        return score_dict
