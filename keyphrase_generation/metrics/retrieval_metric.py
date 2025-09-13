import os
import gin
import string
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics import KeyphraseMetric
from metrics.metric_utils import separate_present_absent_by_source, filter_prediction, find_unique_target
from sentence_transformers import util, SentenceTransformer, CrossEncoder, evaluation
from sklearn.feature_extraction import _stop_words

from nltk.stem.porter import *
from rank_bm25 import BM25Okapi


@gin.configurable
class RetrievalMetric(KeyphraseMetric):
    def __init__(self, corpus_src_file, corpus_tgt_file, query_file, 
                 do_sparse_retrieval=True, 
                 bi_encoder_name=None, cross_encoder_name=None, 
                 bi_encoder_corpus_cache_prefix=None, force_recalc_index=False,
                 ks=[1, 5], utility_score_round_limit=5, docs_for_rerank=50,
                 batch_size=32, max_seq_length=512):
        """
        Retrieval-oriented metric (based on sentence transformers)
        """
        super(RetrievalMetric, self).__init__()
        
        self.title_only = True
        
        self.corpus_src_file = corpus_src_file
        self.corpus_tgt_file = corpus_tgt_file
        self.corpus_str_list_src, self.corpus_tokens_list_src = [], []
        self.corpus_str_list_tgt, self.corpus_tokens_list_tgt = [], []
        self.corpus_tokens_list = None
        self.corpus_read = False
        self.corpus_embeddings = None
        
        self.query_file = query_file
        self.query_str_list, self.query_tokens_list = [], []
        self.query_read = False
        self.query_embeddings = None
        
        self.do_sparse_retrieval = do_sparse_retrieval
        self.ks = [int(x) for x in ks]
        self.maxk = max(self.ks)
        
        self.utility_score_round_limit = utility_score_round_limit
        
        self.docs_for_rerank = docs_for_rerank
        
        self.bi_encoder = None
        if bi_encoder_name is not None:
            self.bi_encoder = SentenceTransformer(bi_encoder_name)
            self.bi_encoder.max_seq_length = max_seq_length
            self.bi_encoder_corpus_cache_file = None
            if bi_encoder_corpus_cache_prefix:
                self.bi_encoder_corpus_cache_file = bi_encoder_corpus_cache_prefix + '_{}_features_cached.pt'.format(bi_encoder_name.split('/')[-1])
            
            if force_recalc_index:
                self.bi_encoder_corpus_cache_file = None
        
        self.cross_encoder = None
        if cross_encoder_name is not None:
            self.cross_encoder = CrossEncoder(cross_encoder_name)
            self.cross_encoder.max_seq_length = max_seq_length

        self.batch_size = batch_size

    def __read_corpus(self):
        # read and preprocessing the corpus for retrieval
        with open(self.corpus_src_file) as f:
            for line in tqdm(f.readlines(), desc='Preprocessing corpus (src)'):
                [title, context] = line.strip().split(self.title_sep)
                if self.title_only:
                    self.corpus_str_list_src.append(title)
                    self.corpus_tokens_list_src.append(self.__bm25_tokenizer(title))
                else:
                    self.corpus_str_list_src.append(title + ' ' + context)
                    self.corpus_tokens_list_src.append(self.__bm25_tokenizer(title + ' ' + context))
        with open(self.corpus_tgt_file) as f:
            for line in tqdm(f.readlines(), desc='Preprocessing corpus (tgt)'):
                line = line.strip().replace(';', '//')
                self.corpus_str_list_tgt.append(line)
                self.corpus_tokens_list_tgt.append(self.__bm25_tokenizer(line))
        assert len(self.corpus_str_list_tgt) == len(self.corpus_str_list_tgt)
        print('Read {} docs as the corpus.'.format(len(self.corpus_str_list_tgt)))
        self.corpus_read = True
        
    def __read_queries(self):
        # read and preprocessing the ad-hoc queries for retrieval
        with open(self.query_file) as f:
            for line in tqdm(f.readlines(), desc='Preprocessing query'):
                self.query_str_list.append(line.strip())
                self.query_tokens_list.append(self.__bm25_tokenizer(line.strip()))
        print('Read {} queries.'.format(len(self.query_str_list)))
        self.query_read = True
        
    def __bm25_tokenizer(self, text):
        tokenized_doc = []
        for token in text.lower().split():
            token = token.strip(string.punctuation)
            if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
                tokenized_doc.append(token)
        return tokenized_doc
    
    def process_single_doc(self, inp):
        src_l =  inp.strip().lower()
        if self.title_sep in src_l:
            [title, context] = src_l.strip().split(self.title_sep)
        else:
            title = ""
            context = src_l
            
        src_token_str = ' '.join(title.strip().split() + context.strip().split())
        stemmed_src_token_list = self.__bm25_tokenizer(src_token_str)
        
        return src_token_str, stemmed_src_token_list
    
    def score_corpus(self, all_preds, all_refs, all_inputs):
        if not self.corpus_read:
            self.__read_corpus()
        if not self.query_read:
            self.__read_queries()
        
        source, target, preds = all_inputs, all_refs, all_preds
        
        # Truncate the kp preds to the number of queries specified
        assert len(source) > len(self.query_str_list)
        source = source[:len(self.query_str_list)]
        target = target[:len(self.query_str_list)]
        preds = preds[:len(self.query_str_list)]
        print(f'Dropping the documents to keep only top {len(self.query_str_list)} that have queries.')
        
        self.corpus_str_list_src = [x[0] for x in source] + self.corpus_str_list_src
        self.corpus_tokens_list_src = [x[1] for x in source] + self.corpus_tokens_list_src
        # use preds for the instances to eval
        self.corpus_str_list_tgt = [' // '.join(x[0]) for x in preds] + self.corpus_str_list_tgt           
        self.corpus_tokens_list_tgt = [self.__bm25_tokenizer(' // '.join(x[0])) for x in preds] + self.corpus_tokens_list_tgt
        self.id2doc = {i: d for i, d in enumerate(self.corpus_str_list_src)}
        
        # index expansion by concatenating the kps with the document body
        if not self.corpus_tokens_list:
            self.corpus_tokens_list, self.corpus_str_list = [], []
            for src_tokens, tgt_tokens in tqdm(zip(self.corpus_tokens_list_src, self.corpus_tokens_list_tgt), 
                                               desc='Index expansion'):
                self.corpus_tokens_list.append(tgt_tokens + src_tokens)
            for src_str, tgt_str in tqdm(zip(self.corpus_str_list_src, self.corpus_str_list_tgt), 
                                               desc='Index expansion'):
                self.corpus_str_list.append(tgt_str + src_str)

        score_dict = {}        
        for k in self.ks:
            if self.do_sparse_retrieval:
                score_dict[f'sparse_recall_at_{k}'] = []
                score_dict[f'sparse_mrr_at_{k}'] = []
            if self.bi_encoder:            
                score_dict[f'dense_recall_at_{k}'] = []    
                score_dict[f'dense_mrr_at_{k}'] = []
            if self.cross_encoder:
                assert self.bi_encoder is not None
                score_dict[f'dense+rerank_recall_at_{k}'] = []    
                score_dict[f'dense+rerank_mrr_at_{k}'] = []
        
        # sparse retrieval
        if self.do_sparse_retrieval:
            bm25 = BM25Okapi(self.corpus_tokens_list)
            for data_idx, (src_l, trg_l, pred_l) in \
                    enumerate(tqdm(zip(source, target, preds),
                                total=len(source), desc='Evaluating sparse retrieval')):
                pred_str_list, pred_token_2dlist, stemmed_pred_token_2dlist = pred_l

                # retrieval with ad-hoc query instead of with keyphrases
                cur_query = self.query_str_list[data_idx]
                bm25_scores = bm25.get_scores(self.__bm25_tokenizer(cur_query))
                top_n = np.argpartition(bm25_scores, -self.maxk)[-self.maxk:]
                
                bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
                bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
                for k in self.ks:
                    if k == 1:
                        score_dict[f'sparse_recall_at_{k}'].append(int(data_idx in [x['corpus_id'] for x in bm25_hits if x['score'] == bm25_hits[0]['score']]))
                        score_dict[f'sparse_mrr_at_{k}'].append(float(score_dict[f'sparse_recall_at_{k}'][-1]))
                    else:
                        score_dict[f'sparse_recall_at_{k}'].append(int(data_idx in [x['corpus_id'] for x in bm25_hits[:k]]))
                        cur_rr = 0
                        for i_result, x in enumerate(bm25_hits[:k]):
                            if data_idx == x['corpus_id']:
                                cur_rr = 1 / (i_result + 1)
                                break
                        score_dict[f'sparse_mrr_at_{k}'].append(cur_rr)
                            
        # dense retrieval with bi-encoder
        if self.bi_encoder:
            corpus_embeddings = None
            if self.bi_encoder_corpus_cache_file and os.path.exists(self.bi_encoder_corpus_cache_file):
                corpus_embeddings = torch.load(self.bi_encoder_corpus_cache_file)
                if corpus_embeddings.shape[0] == len(self.corpus_str_list):
                    print('Loaded dense index from', self.bi_encoder_corpus_cache_file)
                    corpus_embeddings = corpus_embeddings.cuda()
                else:
                    corpus_embeddings = None
            if corpus_embeddings is None:
                print('Building dense index:')
                corpus_embeddings = self.bi_encoder.encode(self.corpus_str_list, convert_to_tensor=True, 
                                                           batch_size=self.batch_size, show_progress_bar=True).cuda()
                if self.bi_encoder_corpus_cache_file is not None:
                    print('Saving index to', self.bi_encoder_corpus_cache_file)
                    torch.save(corpus_embeddings.cpu(), self.bi_encoder_corpus_cache_file)
            for data_idx, (src_l, trg_l, pred_l) in \
                    enumerate(tqdm(zip(source, target, preds),
                                total=len(source), desc='Evaluating dense retrieval')):
                pred_str_list, pred_token_2dlist, stemmed_pred_token_2dlist = pred_l

                # retrieval with ad-hoc query instead of with keyphrases
                cur_query = self.query_str_list[data_idx]
                question_embedding = self.bi_encoder.encode(cur_query, convert_to_tensor=True)
                question_embedding = question_embedding.cuda()
                biencoder_hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=self.maxk)[0]
                biencoder_hits = sorted(biencoder_hits, key=lambda x: x['score'], reverse=True)
                for k in self.ks:
                    if k == 1:
                        score_dict[f'dense_recall_at_{k}'].append(int(data_idx in [x['corpus_id'] for x in biencoder_hits if x['score'] == biencoder_hits[0]['score']]))
                        score_dict[f'dense_mrr_at_{k}'].append(float(score_dict[f'dense_recall_at_{k}'][-1]))
                    else:
                        score_dict[f'dense_recall_at_{k}'].append(int(data_idx in [x['corpus_id'] for x in biencoder_hits[:k]]))
                        cur_rr = 0
                        for i_result, x in enumerate(biencoder_hits[:k]):
                            if data_idx == x['corpus_id']:
                                cur_rr = 1 / (i_result + 1)
                                break
                        score_dict[f'dense_mrr_at_{k}'].append(cur_rr)
                
        # dense retrieval with bi-encoder + rerank with cross-encoder
        if self.cross_encoder:
            for data_idx, (src_l, trg_l, pred_l) in \
                    enumerate(tqdm(zip(source, target, preds),
                                total=len(source), desc='Evaluating dense retrieval + reranking')):
                pred_str_list, pred_token_2dlist, stemmed_pred_token_2dlist = pred_l

                # retrieval with ad-hoc query instead of with keyphrases
                cur_query = self.query_str_list[data_idx]
                question_embedding = self.bi_encoder.encode(cur_query, convert_to_tensor=True)
                question_embedding = question_embedding.cuda()
                biencoder_hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=self.docs_for_rerank)[0]
                biencoder_hits = sorted(biencoder_hits, key=lambda x: x['score'], reverse=True)
                crossencoder_hits = [{'corpus_id': x['corpus_id']} for x in biencoder_hits]
                cross_inp = [[cur_query, self.id2doc[hit['corpus_id']]] for hit in biencoder_hits]
                cross_scores = self.cross_encoder.predict(cross_inp)
                for idx in range(len(cross_scores)):
                    crossencoder_hits[idx]['score'] = cross_scores[idx]
                crossencoder_hits = sorted(crossencoder_hits, key=lambda x: x['score'], reverse=True)
                for k in self.ks:
                    if k == 1:
                        score_dict[f'dense+rerank_recall_at_{k}'].append(int(data_idx in [x['corpus_id'] for x in crossencoder_hits if x['score'] == crossencoder_hits[0]['score']]))
                        score_dict[f'dense+rerank_mrr_at_{k}'].append(float(score_dict[f'dense+rerank_recall_at_{k}'][-1]))
                    else:
                        score_dict[f'dense+rerank_recall_at_{k}'].append(int(data_idx in [x['corpus_id'] for x in crossencoder_hits[:k]]))
                        cur_rr = 0
                        for i_result, x in enumerate(crossencoder_hits[:k]):
                            if data_idx == x['corpus_id']:
                                cur_rr = 1 / (i_result + 1)
                                break
                        score_dict[f'dense+rerank_mrr_at_{k}'].append(cur_rr)
        
        return score_dict

    def aggregate_scores(self, score_dict):   
        aggregated_scores_dict = {}   
        for k in self.ks:
            if self.do_sparse_retrieval:
                aggregated_scores_dict[f'sparse_recall_at_{k}'] = sum(score_dict[f'sparse_recall_at_{k}']) / len(score_dict[f'sparse_recall_at_{k}'])
                aggregated_scores_dict[f'sparse_mrr_at_{k}'] = sum(score_dict[f'sparse_mrr_at_{k}']) / len(score_dict[f'sparse_mrr_at_{k}'])
            if self.bi_encoder:
                aggregated_scores_dict[f'dense_recall_at_{k}'] = sum(score_dict[f'dense_recall_at_{k}']) / len(score_dict[f'dense_recall_at_{k}'])
                aggregated_scores_dict[f'dense_mrr_at_{k}'] = sum(score_dict[f'dense_mrr_at_{k}']) / len(score_dict[f'dense_mrr_at_{k}'])
            if self.cross_encoder:
                aggregated_scores_dict[f'dense+rerank_recall_at_{k}'] = sum(score_dict[f'dense+rerank_recall_at_{k}']) / len(score_dict[f'dense+rerank_recall_at_{k}'])
                aggregated_scores_dict[f'dense+rerank_mrr_at_{k}'] = sum(score_dict[f'dense+rerank_mrr_at_{k}']) / len(score_dict[f'dense+rerank_mrr_at_{k}'])
            
        return score_dict, aggregated_scores_dict
