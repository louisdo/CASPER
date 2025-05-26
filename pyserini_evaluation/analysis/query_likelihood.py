import json, os, math, string, re
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from collections import Counter
from tqdm import tqdm
from nltk.corpus import stopwords
from english_words import get_english_words_set
from functools import lru_cache
from nltk.stem import PorterStemmer

WEB2LOWERSET = get_english_words_set(['web2'], lower=True)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

@lru_cache(maxsize=100000)
def word_check(word):
    if word in string.punctuation: 
        return False
    
    if word in STOPWORDS: return False
    if word not in WEB2LOWERSET: return False

    if len(word) > 1 and re.search(r"^([a-zA-Z]+-)*[a-zA-Z]+((\s|-)[0-9]+){0,1}$", word): # r"^([a-z]+-)*[a-z]+(\s[0-9]+){0,1}$"
        return True
        
    return False


@lru_cache(maxsize=100000)
def stem_word(word):
    return STEMMER.stem(word)


def query_likelihood_model(query, document, do_stemming = False):
    query_terms = [term.strip(string.punctuation) for term in query.lower().split() if word_check(term)]
    document_terms = [term.strip(string.punctuation) for term in document.lower().split() if word_check(term)]

    if do_stemming:
        query_terms = [stem_word(word) for word in query_terms]
        document_terms = [stem_word(word) for word in document_terms]

    doc_tf = Counter(document_terms)
    doc_length = len(document_terms)

    query_tf = Counter(query_terms)
    if not query_tf: return None

    log_prob = 0.0
    for term in query_tf:
        term_prob = doc_tf[term] / doc_length if term in doc_tf else 0
        if term_prob == 0:
            term_prob = 1e-10

        log_prob += query_tf[term] * math.log(term_prob)

    return log_prob / len(query_tf)


def convert_to_pytrec_eval_format(queries, all_search_results, type = "relevance"):
    """
    queries: [q1, q2, ...]
    all_search_results: [[{'docid': '22711954', 'score': 4.043900012969971}, ...]]
    """

    assert len(queries) == len(all_search_results)

    score_converter = {
        "relevance": lambda x: int(x),
        "prediction": lambda x:float(x)
    }

    res = {}
    for query, search_results in zip(queries, all_search_results):
        if query not in res:
            res[query] = {}
        

        for sr in search_results:
            docid = str(sr["docid"])
            score = score_converter[type](sr["score"])

            res[query][docid] = score

    return res


def read_qrels(qrel_path):
    _qrels = pd.read_csv(qrel_path, sep='\t').to_dict("records")
    
    metadata = {}
    for line in _qrels:
        query_id = str(line["query-id"])
        doc_id = str(line["corpus-id"])
        score = line["score"]

        if query_id not in metadata:
            metadata[query_id] = []

        metadata[query_id].append({
            "docid": doc_id,
            "score": score
        })

    queries_ids = list(metadata.keys())
    queries_all_labels = [metadata[k] for k in queries_ids]

    qrels = convert_to_pytrec_eval_format(queries = queries_ids, all_search_results=queries_all_labels)

    return qrels


def create_query_reldoc_pairs(queries, corpus, qrels):
    for query_id in qrels:
        for doc_id in qrels[query_id]:
            if qrels[query_id][doc_id] > 0: 
                query_id = str(query_id)
                doc_id = str(doc_id)
                if query_id in queries and doc_id in corpus:
                    yield [queries[query_id], corpus[doc_id]]


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--work_dir", type = str, default = "../..")
    parser.add_argument("--do_stemming", type = int, choices = [0, 1], default = 0)

    args = parser.parse_args()

    dataset_name = args.dataset_name
    work_dir = args.work_dir
    do_stemming = args.do_stemming

    dataset_name_2_relative_path = {
        "scifact": "data/beir/scifact",
        "scidocs": "data/beir/scidocs",
        "nfcorpus": "data/beir/nfcorpus",
        "arguana": "data/beir/arguana",
        "fiqa": "data/beir/fiqa",
        "trec-covid": "data/beir/trec-covid",
        "msmarco": "data/msmarco/msmarco",
        "doris_mae": "data/doris_mae/doris_mae",
        "cfscube": "data/cfscube/cfscube",
        "trec_dl_2019": "data/msmarco/trec_dl_2019",
        "trec_dl_2020": "data/msmarco/trec_dl_2020",
        "acm_cr": "data/acm_cr/acm_cr"
    }

    queries_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset_name],
        "queries.jsonl")
    
    corpus_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset_name] if dataset_name not in ["trec_dl_2019", "trec_dl_2020"] else "data/msmarco/msmarco",
        "corpus.jsonl")
    
    qrel_path = os.path.join(
        work_dir,
        dataset_name_2_relative_path[dataset_name],
        "qrels/test.tsv" if dataset_name != "msmarco" else "qrels/dev.tsv"
    )
    qrels = read_qrels(qrel_path=qrel_path)


    # load queries
    with open(queries_path) as f:
        queries_ = [json.loads(line) for line in tqdm(f)]
        queries_ = [line for line in queries_ if line["_id"] in qrels]
        queries = {str(line["_id"]): line["text"] for line in queries_}

    # load corpus
    with open(corpus_path) as f:
        corpus_ = [json.loads(line) for line in tqdm(f)]
        corpus = {str(line.get("_id", line.get("id", None))): line["title"] + " " + line["text"] for line in corpus_}

    query_reldoc_pairs = list(create_query_reldoc_pairs(queries, corpus, qrels))

    queries_loglikelihood = [query_likelihood_model(query, document, do_stemming) for query, document in query_reldoc_pairs]
    queries_loglikelihood = [item for item in queries_loglikelihood if item is not None]

    print(np.mean(queries_loglikelihood))


if __name__ == "__main__":
    main()