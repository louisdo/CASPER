import gin
import numpy as np
from tqdm import tqdm

from metrics import KeyphraseMetric
from metrics.metric_utils import separate_present_absent_by_source, filter_prediction, find_unique_target

from nltk.stem.porter import *
from sklearn.feature_extraction import _stop_words


@gin.configurable
class MoverScoreMetric(KeyphraseMetric):
    def __init__(self, version=2, n_gram=1, remove_subwords=True, batch_size=48, remove_stop_words=False):
        """
        Mover Score metric
        Interfaces https://github.com/AIPHES/emnlp19-moverscore
        NOTE: mover score assumes GPU usage
        Args:
                :param version: Which version of moverscore to use; v2 makes use of DistilBert and will
                        run quicker.
                :param remove_stop_words: remove stop words when calculating mover_score
                :param n_gram: n_gram size to use in mover score calculation; see Section 3.1 of paper for details
                :param remove_subwords: whether to remove subword tokens before calculating n-grams and proceeding
                        with mover score calculation
                :param batch_size:
                        batch size for mover score calculation; change according to hardware for improved speed
        """
        super(MoverScoreMetric, self).__init__()
        self.version = version
        if self.version == 1:
            from moverscore import get_idf_dict, word_mover_score
        else:
            from moverscore_v2 import get_idf_dict, word_mover_score
        self.get_idf_dict = get_idf_dict
        self.word_mover_score = word_mover_score
        self.stop_words = []
        if remove_stop_words:
            self.stop_words = list(_stop_words.ENGLISH_STOP_WORDS)
        self.n_gram = n_gram
        self.remove_subwords = remove_subwords
        self.batch_size = batch_size

    def score_corpus(self, all_preds, all_refs, all_inputs):
        source, target, preds = all_inputs, all_refs, all_preds
        
        predicted_keyphrases = []
        pkp_preds, pkp_refs, akp_preds, akp_refs, allkp_preds, allkp_refs = [], [], [], [], [], []
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
            
            # entries for bert_score calculation
            pkp_preds.append(', '.join(result['present']))
            pkp_refs.append(', '.join(result['present_ref']))
            akp_preds.append(', '.join(result['absent']))
            akp_refs.append(', '.join(result['absent_ref']))
            allkp_preds.append(', '.join(result['present'] + result['absent']))
            allkp_refs.append(', '.join(result['present_ref'] + result['absent_ref']))
            
        # calculate mover_score
        idf_dict_refs = self.get_idf_dict(allkp_refs)
        idf_dict_preds = self.get_idf_dict(allkp_preds)
        pkp_scores = self.word_mover_score(pkp_preds, pkp_refs, idf_dict_refs, idf_dict_preds, 
                                           stop_words=self.stop_words, n_gram=self.n_gram, 
                                           remove_subwords=self.remove_subwords, batch_size=self.batch_size)
        akp_scores = self.word_mover_score(akp_preds, akp_refs, idf_dict_refs, idf_dict_preds, 
                                           stop_words=self.stop_words, n_gram=self.n_gram, 
                                           remove_subwords=self.remove_subwords, batch_size=self.batch_size)
        allkp_scores = self.word_mover_score(allkp_preds, allkp_refs, idf_dict_refs, idf_dict_preds, 
                                             stop_words=self.stop_words, n_gram=self.n_gram, 
                                             remove_subwords=self.remove_subwords, batch_size=self.batch_size)
        
        score_dict = {'mover_score_present': pkp_scores,
                      'mover_score_absent': akp_scores,
                      'mover_score_all': allkp_scores}
        
        return score_dict
