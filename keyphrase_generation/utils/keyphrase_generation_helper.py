import math
import numpy as np
from utils.splade_inference import get_tokens_scores_of_docs_batch
from utils.s2orc_phrase_vocab import S2ORC_PHRASE_VOCAB
from collections import Counter
from typing import List, Dict

def batch_nounphrase_extraction(docs, CANDEXT):
    batch_candidates = [CANDEXT(doc) for doc in docs]
    candidates_scores = [Counter(item) for item in batch_candidates]
    
    return candidates_scores

    
def score_candidates_by_positions(candidates: List[str], doc: str):
    res = Counter()
    for cand in candidates:
        try:
            temp = doc.index(cand)
            position = len([item for item in doc[:temp].split(" ") if item]) + 1
            position_score = 1 + 1 / math.log2(position + 2) #(position + 1) / position
        except ValueError:
            position_score = 1
        res[cand] = position_score
    return res



def get_present_keyphrases(
        candidates: List[str], 
        candidates_tokens: List[List[int]], 
        tokens_scores: dict,
        model_name: str,
        length_penalty: int = 0,
        candidates_positions_scores: dict = {},
        candidates_phraseness_scores: dict = {},
        SPLADE_MODEL = None):
    # this is for present keyphrase generation
    # length penalization < 0 means returning longer sequence
    tokenized_candidates = [SPLADE_MODEL[model_name]["tokenizer"].convert_ids_to_tokens(item) for item in candidates_tokens]

    candidates_scores = [np.sum([tokens_scores[tok] for tok in tokenized_cand]) / (len(tokenized_cand) - length_penalty) for tokenized_cand in tokenized_candidates]

    if candidates_phraseness_scores:
        candidates_scores = [score * (candidates_phraseness_scores[candidates[i]] ** 1.5) for i, score in enumerate(candidates_scores)]

    if candidates_positions_scores:
        candidates_scores = [score * candidates_positions_scores[candidates[i]] for i, score in enumerate(candidates_scores)]
    assert len(candidates) == len(candidates_scores)
    return [(cand, score) for cand, score in zip(candidates, candidates_scores)]


def get_absent_keyphrases(tokens_scores, lower_doc, added_phrase_vocab = S2ORC_PHRASE_VOCAB):
    possible_absent_keyphrases_scores = [(tok, tokens_scores[tok]) for tok in tokens_scores if tok in S2ORC_PHRASE_VOCAB]
    absent_keyphrases_scores = [item for item in possible_absent_keyphrases_scores if item[0] not in lower_doc]

    return absent_keyphrases_scores


def _keygen_single_doc_helper(
        lower_doc,
        candidates,
        candidates_tokens,
        tokens_scores,
        informativeness_model_name,
        length_penalty,
        candidates_positions_scores,
        candidates_phraseness_scores,
        SPLADE_MODEL,
        top_k
):
    present_keyphrases = get_present_keyphrases(
        candidates=candidates,
        candidates_tokens=candidates_tokens,
        tokens_scores=tokens_scores,
        model_name=informativeness_model_name,
        length_penalty=length_penalty,
        candidates_positions_scores=candidates_positions_scores,
        candidates_phraseness_scores=candidates_phraseness_scores,
        SPLADE_MODEL=SPLADE_MODEL
    )
    present_keyphrases = list(sorted(present_keyphrases, key = lambda x: -x[1]))

    absent_keyphrases = get_absent_keyphrases(
        tokens_scores = tokens_scores, 
        lower_doc = lower_doc
    )
    absent_keyphrases = list(sorted(absent_keyphrases, key = lambda x: -x[1]))

    return {
        "present": present_keyphrases[:top_k],
        "absent": absent_keyphrases[:top_k]
    }




def keyphrase_generation_batch(
    docs: str, 
    informativeness_model_name: str,
    top_k: int = 10,
    apply_position_penalty: bool = False,
    length_penalty: int = 0,
    CANDEXT = None,
    SPLADE_MODEL = None,
):
    lower_docs = [str(doc).lower() for doc in docs]
    docs_tokens = SPLADE_MODEL[informativeness_model_name]["tokenizer"](lower_docs, return_tensors="pt", max_length = 512, padding = True, truncation = True)

    batch_tokens_scores = get_tokens_scores_of_docs_batch(
        docs_tokens = docs_tokens, 
        model_name = informativeness_model_name)
    

    batch_candidates_phraseness_scores = batch_nounphrase_extraction(docs = docs, CANDEXT=CANDEXT)

    batch_candidates = [list(candidates_phraseness_score.keys()) for candidates_phraseness_score in batch_candidates_phraseness_scores]

    batch_candidates_tokens = [[SPLADE_MODEL[informativeness_model_name]["tokenizer"](cand)["input_ids"][1:-1] for cand in candidates] for candidates in batch_candidates]

    if apply_position_penalty:
        batch_candidates_positions_scores = [score_candidates_by_positions(candidates, lower_doc) for candidates, lower_doc in zip(batch_candidates, lower_docs)]
    else:
        batch_candidates_positions_scores = [[] for _ in range(len(batch_candidates))]
    
    
    res = [{"present":[], "absent": []} for _ in range(len(docs))]

    for i in range(len(docs)):
        lower_doc = lower_docs[i]
        candidates = batch_candidates[i]
        candidates_tokens = batch_candidates_tokens[i]
        tokens_scores = batch_tokens_scores[i]
        candidates_positions_scores = batch_candidates_positions_scores[i]
        candidates_phraseness_scores = batch_candidates_phraseness_scores[i]

        temp = _keygen_single_doc_helper(
            lower_doc,
            candidates,
            candidates_tokens,
            tokens_scores,
            informativeness_model_name,
            length_penalty,
            candidates_positions_scores,
            candidates_phraseness_scores,
            SPLADE_MODEL,
            top_k
        )

        res[i] = temp

    return res