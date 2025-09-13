import os
import gin
import numpy as np
from tqdm import tqdm
import evaluate

from metrics import KeyphraseMetric
from metrics.metric_utils import separate_present_absent_by_source, filter_prediction, find_unique_target

from nltk.stem.porter import *


@gin.configurable
class MeteorMetric(KeyphraseMetric):
    def __init__(self, alpha, beta, gamma):
        """
        METEOR metric (based on huggingface evaluation)
        """
        super(MeteorMetric, self).__init__()
        self.meteor = evaluate.load('meteor')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def score_corpus(self, all_preds, all_refs, all_inputs):
        source, target, preds = all_inputs, all_refs, all_preds
        
        score_dict = {'present_kp_meteor': [], 'absent_kp_meteor': [], 'all_kp_meteor': []}
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

            # Use the stemmed phrases to calculate METEOR
            # filtered_pred_token_2dlist = [kp_tokens for kp_tokens, is_unique
            #                               in zip(pred_token_2dlist, is_unique_mask) if is_unique]
            # result = {'id': data_idx, 'present': [], 'absent': [], 'present_ref': [], 'absent_ref': []}
            # for kp_tokens, is_present in zip(filtered_pred_token_2dlist, is_present_mask):
            #     if is_present:
            #         result['present'] += [' '.join(kp_tokens)]
            #     else:
            #         result['absent'] += [' '.join(kp_tokens)]
            
            # for kp_tokens, is_present in zip(trg_token_2dlist, is_present_mask_gt):
            #     if is_present:
            #         result['present_ref'] += [' '.join(kp_tokens)]
            #     else:
            #         result['absent_ref'] += [' '.join(kp_tokens)]
            
            # calculate meteor scores
            # pkp_scores = self.meteor.compute(predictions=[' {} '.format(self.kp_sep).join(result['present'])], references=[' {} '.format(self.kp_sep).join(result['present_ref'])], alpha=self.alpha, beta=self.beta, gamma=self.gamma)
            # score_dict['present_kp_meteor'].append(pkp_scores['meteor'])
            
            # akp_scores = self.meteor.compute(predictions=[' {} '.format(self.kp_sep).join(result['absent'])], references=[' {} '.format(self.kp_sep).join(result['absent_ref'])], alpha=self.alpha, beta=self.beta, gamma=self.gamma)
            # score_dict['absent_kp_meteor'].append(akp_scores['meteor'])
            
            # allkp_scores = self.meteor.compute(predictions=[' {} '.format(self.kp_sep).join(result['present'] + result['absent'])], references=[' {} '.format(self.kp_sep).join(result['present_ref'] + result['absent_ref'])], alpha=self.alpha, beta=self.beta, gamma=self.gamma)
            # score_dict['all_kp_meteor'].append(allkp_scores['meteor'])
            
            # Use the stemmed phrases to calculate METEOR
            pkp_scores = self.meteor.compute(predictions=[', '.join([' '.join(x) for x in present_filtered_stemmed_pred_token_2dlist])], 
                                             references=[', '.join([' '.join(x) for x in present_unique_stemmed_trg_token_2dlist])], 
                                             alpha=self.alpha, beta=self.beta, gamma=self.gamma)
            score_dict['present_kp_meteor'].append(pkp_scores['meteor'])
            
            akp_scores = self.meteor.compute(predictions=[', '.join([' '.join(x) for x in absent_filtered_stemmed_pred_token_2dlist])], 
                                             references=[', '.join([' '.join(x) for x in absent_unique_stemmed_trg_token_2dlist])], 
                                             alpha=self.alpha, beta=self.beta, gamma=self.gamma)
            score_dict['absent_kp_meteor'].append(akp_scores['meteor'])
            
            allkp_scores = self.meteor.compute(predictions=[', '.join([' '.join(x) for x in present_filtered_stemmed_pred_token_2dlist] + [' '.join(x) for x in absent_filtered_stemmed_pred_token_2dlist])], 
                                               references=[', '.join([' '.join(x) for x in present_filtered_stemmed_pred_token_2dlist] + [' '.join(x) for x in absent_unique_stemmed_trg_token_2dlist])], 
                                               alpha=self.alpha, beta=self.beta, gamma=self.gamma)
            score_dict['all_kp_meteor'].append(allkp_scores['meteor'])
            
        return score_dict
