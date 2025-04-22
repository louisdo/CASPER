import gin
import bert_score
import numpy as np
from tqdm import tqdm

from metrics import KeyphraseMetric
from metrics.metric_utils import separate_present_absent_by_source, filter_prediction, find_unique_target

from nltk.stem.porter import *

@gin.configurable
class BertScoreMetric(KeyphraseMetric):
    def __init__(self, lang='en', model_type='bert-base-uncased', num_layers=8, verbose=False, idf=False,
                 nthreads=4, batch_size=64, rescale_with_baseline=False):
        """
        BERT-Score metric
        Args (copied from https://github.com/Tiiiger/bert_score/blob/master/bert_score/score.py):
            :param model_type (str): bert specification, default using the suggested
                      model for the target langauge; has to specify at least one of
                      model_type or lang
            :param num_layers (int): the layer of representation to use.
                      default using the number of layer tuned on WMT16 correlation data
            :param verbose (bool): turn on intermediate status update
            :param idf (bool or dict): use idf weighting, can also be a precomputed idf_dict
            :param device (str): on which the contextual embedding model will be allocated on.
                      If this argument is None, the model lives on cuda:0 if cuda is available.
            param nthreads (int): number of threads
            param batch_size (int): bert score processing batch size
            param lang (str): language of the sentences; has to specify
                      at least one of model_type or lang. lang needs to be
                      specified when rescale_with_baseline is True.
            param rescale_with_baseline (bool): rescale bertscore with pre-computed baseline
        """
        super(BertScoreMetric, self).__init__()
        
        self.lang = lang
        self.model_type = model_type
        self.num_layers = num_layers
        self.verbose = verbose
        self.idf = idf
        self.nthreads = nthreads
        self.batch_size = batch_size
        self.rescale_with_baseline = rescale_with_baseline

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
            
        # calculate bert_score
        pkp_out, hash_code = bert_score.score(pkp_preds, pkp_refs, model_type=self.model_type, 
                                              num_layers=self.num_layers,
                                              verbose=self.verbose, idf=self.idf, batch_size=self.batch_size,
                                              nthreads=self.nthreads, lang=self.lang, return_hash=True,
                                              rescale_with_baseline=self.rescale_with_baseline)
        akp_out, hash_code = bert_score.score(akp_preds, akp_refs, model_type=self.model_type, 
                                              num_layers=self.num_layers,
                                              verbose=self.verbose, idf=self.idf, batch_size=self.batch_size,
                                              nthreads=self.nthreads, lang=self.lang, return_hash=True,
                                              rescale_with_baseline=self.rescale_with_baseline)
        allkp_out, hash_code = bert_score.score(allkp_preds, allkp_refs, model_type=self.model_type, 
                                                num_layers=self.num_layers,
                                                verbose=self.verbose, idf=self.idf, batch_size=self.batch_size,
                                                nthreads=self.nthreads, lang=self.lang, return_hash=True,
                                                rescale_with_baseline=self.rescale_with_baseline)
        print('Hash code:', hash_code)
        score_dict = {'bert_score_present_precision': pkp_out[0].cpu().tolist(),
                      'bert_score_present_recall': pkp_out[1].cpu().tolist(),
                      'bert_score_present_f1': pkp_out[2].cpu().tolist(),
                      'bert_score_absent_precision': akp_out[0].cpu().tolist(),
                      'bert_score_absent_recall': akp_out[1].cpu().tolist(),
                      'bert_score_absent_f1': akp_out[2].cpu().tolist(),
                      'bert_score_all_precision': allkp_out[0].cpu().tolist(),
                      'bert_score_all_recall': allkp_out[1].cpu().tolist(),
                      'bert_score_all_f1': allkp_out[2].cpu().tolist()}
        
        return score_dict
