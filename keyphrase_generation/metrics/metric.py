import gin
from typing import List, Dict
from tqdm import tqdm 
import numpy as np
from nltk.stem.porter import *
from metrics.metric_utils import stem_word_list, stem_str_list, stem_str_2d_list


# base class
@gin.configurable
class KeyphraseMetric:
    def __init__(self, kp_sep, title_sep, unk_word, invalidate_unk):
        self.kp_sep = kp_sep
        self.title_sep = title_sep
        self.unk_word = unk_word
        self.invalidate_unk = invalidate_unk
        self.title_only = False
         
        self.stemmer = PorterStemmer()
        
    def process_single_doc(self, inp):
        src_l =  inp.strip().lower()
        if self.title_sep in src_l:
            [title, context] = src_l.strip().split(self.title_sep)
        else:
            title = ""
            context = src_l
            
        if self.title_only:    
            src_token_list = title.strip().split(' ')
        else:
            src_token_list = title.strip().split(' ') + context.strip().split(' ')
        stemmed_src_token_list = stem_word_list(src_token_list, self.stemmer)
        
        return src_token_list, stemmed_src_token_list

    def process_kp_string(self, kps):
        str_list = []
        for r in kps:
            r = r.strip()
            r = r.replace(' ##', '')
            r = r.replace('[ digit ]', '[digit]')
            str_list.append(r)
            
        token_2dlist = [kp_str.split(' ') for kp_str in str_list]
        stemmed_token_2dlist = stem_str_list(token_2dlist, self.stemmer)
        
        return str_list, token_2dlist, stemmed_token_2dlist
    
    def score_single_doc(self, preds: List[str], refs: List[str], input: str=None):
        raise NotImplementedError

    def score_corpus(self, all_preds: List[List[str]], all_refs: List[List[str]], all_inputs: List[List[str]]):
        raise NotImplementedError

    def aggregate_scores(self, all_doc_scores_dict: Dict[str, List]):
        # default: (micro) average over all cases
        aggregated_scores_dict = {k: np.mean(v) for k, v in all_doc_scores_dict.items()}        
        return all_doc_scores_dict, aggregated_scores_dict

    def evaluate(self, all_preds: List[List[str]], all_refs: List[List[str]], all_inputs: List[List[str]]):
        all_preds = [self.process_kp_string(x) for x in tqdm(all_preds, desc='Preparing predictions')]
        all_refs = [self.process_kp_string(x) for x in tqdm(all_refs, desc='Preparing references')]
        all_inputs = [self.process_single_doc(x) for x in tqdm(all_inputs, desc='Preparing inputs')]if all_inputs else None

        per_doc_scores_dict = self.score_corpus(all_preds, all_refs, all_inputs)
        per_doc_scores_dict_filtered, aggregated_scores_dict = self.aggregate_scores(per_doc_scores_dict)

        return per_doc_scores_dict_filtered, aggregated_scores_dict