# based on https://github.com/maszhongming/UniEval
import gin
import traceback
from typing import List
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from metrics import KeyphraseMetric

from nltk.stem.porter import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM


class UniEvaluator:
    def __init__(self, model_name_or_path, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up model """
        self.device = device
        self.max_length = max_length

        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.config,
                                                           cache_dir=cache_dir)

        self.model.eval()
        self.model.to(device)

        self.softmax = nn.Softmax(dim=1)

        self.pos_id = self.tokenizer("Yes")["input_ids"][0]
        self.neg_id = self.tokenizer("No")["input_ids"][0]

    def score(self, inputs, batch_size=8):
        """
            Get scores for the given samples.
            final_score = postive_score / (postive_score + negative_score)
        """

        # The implementation of "forward" in T5 still requires decoder_input_ids.
        # Therefore, we construct a random one-word target sequence.
        # The content of the target has no effect on the final scores.
        tgts = ["No" for _ in range(len(inputs))]

        pos_score_list, neg_score_list = [], []
        for i in tqdm(range(0, len(inputs), batch_size)):
            src_list = inputs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            with torch.no_grad():
                encoded_src = self.tokenizer(
                    src_list,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
                encoded_tgt = self.tokenizer(
                    tgt_list,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )

                src_tokens = encoded_src['input_ids'].to(self.device)
                src_mask = encoded_src['attention_mask'].to(self.device)
                
                tgt_tokens = encoded_tgt['input_ids'].to(self.device)[:, 0].unsqueeze(-1)
                
                output = self.model(
                    input_ids=src_tokens,
                    attention_mask=src_mask,
                    labels = tgt_tokens
                )
                logits = output.logits.view(-1, self.model.config.vocab_size)
                
                pos_score = self.softmax(logits)[:, self.pos_id] # Yes
                neg_score = self.softmax(logits)[:, self.neg_id] # No
                
                cur_pos_score = [x.item() for x in pos_score]
                cur_neg_score = [x.item() for x in neg_score]
                pos_score_list += cur_pos_score
                neg_score_list += cur_neg_score
                
        score_list = []
        for i in range(len(pos_score_list)):
            score_list.append(pos_score_list[i] / (pos_score_list[i] + neg_score_list[i]))
            
        return score_list
    

@gin.configurable
class UniEvalMetric(KeyphraseMetric):
    def __init__(self, dimensions, max_length=1024, batch_size=8, device='cuda:0'):
        """
        UniEval (https://aclanthology.org/2022.emnlp-main.131/)
        """
        super(UniEvalMetric, self).__init__()        
        self.dimensions = dimensions
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

    def add_question(self, dimension, output, src, ref):
        """
        Add questions to generate input in Bool-QA format for UniEval.
        
        dimension: specific dimension to be evaluated
        src: source input 
        output: output text 
        ref: human reference
        """
        input_with_question = []
        output_copy = copy.deepcopy(output)
        ref_copy = copy.deepcopy(ref) if ref is not None else None
        for i in range(len(output)):
            if dimension == 'naturalness':
                # cur_input = f'question: Is this a natural utterance? </s> utterance: This an article about {output_copy[i]}.'
                cur_input = f'question: Is this a natural utterance? </s> utterance: This article discusses {output_copy[i]}.'
            elif dimension == 'faithfulness-summ':
                # cur_input = f'question: Is this claim consistent with the document? </s> summary: the concept {output_copy[i]} is mentioned or described in the document. </s> document: {src[i]}'
                cur_input = f'question: Is this claim consistent with the document? </s> summary: the document discusses about {output_copy[i]}. </s> document: {src[i]}'
            elif dimension == 'faithfulness-dialog':
                # templates for the dialogue model
                cur_input = f'question: Is this response consistent with knowledge in the fact? </s> response:  {output_copy[i]} is mentioned or described in the document. </s> fact: {src[i]}'
            elif dimension == 'faithfulness-fact':
                # templates for the fact model
                cur_input = f'question: Is this claim consistent with the document? </s> summary: the concept {output_copy[i]} is mentioned or described in the document. </s> document: {src[i]}'
            elif dimension == 'saliency-summ':
                output[i][-1] = 'and ' + output_copy[i][-1]
                ref_copy[i][-1] = 'and ' + ref_copy[i][-1]
                cur_input = 'question: Is this summary relevant to the reference? </s> summary: ' \
                    + f'This document is about {", ".join(output_copy[i])}.' \
                    + ' </s> reference: ' + f'This document is about {", ".join(ref_copy[i])}.'
            elif dimension == 'coverage-summ':
                output[i][-1] = 'and ' + output_copy[i][-1]
                ref_copy[i][-1] = 'and ' + ref_copy[i][-1]
                cur_input = 'question: Is this summary relevant to the reference? </s> summary: ' \
                    + f'This document is about {", ".join(ref_copy[i])}.' \
                    + ' </s> reference: ' + f'This document is about {", ".join(output_copy[i])}.'
            elif dimension == 'diversity':
                output[i][-1] = 'and ' + output_copy[i][-1]
                cur_input = f'question: Does this utterance contain diverse concepts? </s> utterance: This is an article about {", ".join(output_copy[i])}.'
            else:
                raise NotImplementedError(f'The input format for dimension "{dimension}" is still undefined. '
                                          'Please customize it first.')
            input_with_question.append(cur_input)
            
        return input_with_question

    def get_unieval_scorer(self, dimension):
        # MingZhong/unieval-sum, MingZhong/unieval-fact, MingZhong/unieval-dialog, MingZhong/unieval-intermediate
        if dimension in ['naturalness', 'faithfulness-summ', 'saliency-summ', 'coverage-summ', 'diversity']:
            evaluator = UniEvaluator('MingZhong/unieval-sum', self.max_length, device=self.device)
        elif dimension in ['faithfulness-dialog']:
            evaluator = UniEvaluator('MingZhong/unieval-dialog', self.max_length, device=self.device)
        elif dimension in ['faithfulness-fact']:
            evaluator = UniEvaluator('MingZhong/unieval-fact', self.max_length, device=self.device)
        else:
            raise NotImplementedError(f'The UniEval model for dimension "{dimension}" is still undefined. '
                                      'Please customize it first.')
        
        return evaluator

    def score_corpus(self, all_preds, all_refs, all_inputs):
        source, target, preds = all_inputs, all_refs, all_preds
        
        # aggregate data
        src_list_for_unieval, ref_2d_list_for_unieval, hyp_2d_list_for_unieval = [], [], []
        empty_pred_mask, empty_ref_mask = [], []
        for data_idx, (src_l, trg_l, pred_l) in \
                enumerate(tqdm(zip(source, target, preds),
                            total=len(source), desc='Evaluating...')):
                        
            src_token_list, stemmed_src_token_list = src_l 
            pred_str_list, pred_token_2dlist, stemmed_pred_token_2dlist = pred_l
            trg_str_list, trg_token_2dlist, stemmed_trg_token_2dlist = trg_l
            
            src_list_for_unieval.append(' '.join(src_token_list))
            
            if len(trg_str_list) == 0:
                trg_str_list = ['null']
                empty_ref_mask.append(True)
            else:
                empty_ref_mask.append(False)
            ref_2d_list_for_unieval.append(trg_str_list)
            
            if len(pred_str_list) == 0:
                pred_str_list = ['null']
                empty_pred_mask.append(True)
            else:
                empty_pred_mask.append(False)
            hyp_2d_list_for_unieval.append(pred_str_list)
            
        # calculate UniEval scores
        score_dict = {}
        
        for dimension in self.dimensions:
            if dimension == 'naturalness':
                # dimension 1: naturalness (single-phrase, reference-free, input-free)
                unieval_scorer = self.get_unieval_scorer(dimension)
                score_dict[dimension] = []
                unieval_inputs = self.add_question(dimension, 
                                                   output=[x for y in hyp_2d_list_for_unieval for x in y],
                                                   src=None,
                                                   ref=None)
                unieval_score = unieval_scorer.score(unieval_inputs, batch_size=self.batch_size)                
                for y in hyp_2d_list_for_unieval:
                    cur_scores = []
                    for _ in y:
                        cur_scores.append(unieval_score.pop(0))
                    score_dict[dimension].append(np.mean(cur_scores))
                
            elif dimension == 'faithfulness-summ' or dimension == 'faithfulness-dialog' or dimension == 'faithfulness-fact':
                # dimension 2: faithfulness (single-phrase, reference-free, input-based)
                unieval_scorer = self.get_unieval_scorer(dimension)
                score_dict[dimension] = []
                unieval_inputs = self.add_question(dimension, 
                                                   output=[x for y in hyp_2d_list_for_unieval for x in y],
                                                   src=[src_list_for_unieval[i] for i, y in enumerate(hyp_2d_list_for_unieval) for _ in y],
                                                   ref=None)
                unieval_score = unieval_scorer.score(unieval_inputs, batch_size=self.batch_size)
                
                for y in hyp_2d_list_for_unieval:
                    cur_scores = []
                    for _ in y:
                        cur_scores.append(unieval_score.pop(0))
                    score_dict[dimension].append(np.mean(cur_scores))               
                
            elif dimension == 'saliency-summ':
                # dimension 3: saliency (whole prediction, reference-based, input-free)
                unieval_scorer = self.get_unieval_scorer(dimension)
                unieval_inputs = self.add_question(dimension, 
                                                   output=hyp_2d_list_for_unieval,
                                                   src=None,
                                                   ref=ref_2d_list_for_unieval)
                unieval_score = unieval_scorer.score(unieval_inputs, batch_size=self.batch_size)
                score_dict[dimension] = unieval_score
                
            elif dimension == 'coverage-summ':
                # dimension 4: coverage (whole prediction, reference-based, input-free)
                unieval_scorer = self.get_unieval_scorer(dimension)
                unieval_inputs = self.add_question(dimension, 
                                                   output=ref_2d_list_for_unieval,
                                                   src=None,
                                                   ref=hyp_2d_list_for_unieval)
                unieval_score = unieval_scorer.score(unieval_inputs, batch_size=self.batch_size)
                score_dict[dimension] = unieval_score
                
            elif dimension == 'diversity':
                # dimension 5: diversity (whole prediction, reference-free, input-free)
                unieval_scorer = self.get_unieval_scorer(dimension)
                unieval_inputs = self.add_question(dimension, 
                                                   output=hyp_2d_list_for_unieval,
                                                   src=None,
                                                   ref=None)
                unieval_score = unieval_scorer.score(unieval_inputs, batch_size=self.batch_size)                
                score_dict[dimension] = unieval_score
                
            else:
                raise NotImplementedError(f'The evaluation code for dimension "{dimension}" is still undefined. '
                                          'Please customize it first.')
           
           
        # correct scores for cases with empty_pred or empty_ref
        for k in score_dict.keys():
            assert len(score_dict[k]) == len(preds)
            for i in range(len(score_dict[k])):
                if empty_pred_mask[i] or empty_ref_mask[i]:
                    score_dict[k][i] = 0.0
        
        return score_dict
