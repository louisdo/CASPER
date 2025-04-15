import json
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from cleantext import clean


clean_func = lambda x: clean(
    x,
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=True,                # replace all email addresses with a special token
    no_phone_numbers=True,         # replace all phone numbers with a special token
    no_numbers=False,               # replace all numbers with a special token
    no_digits=False,                # replace all digits with a special token
    no_currency_symbols=False,      # replace all currency symbols with a special token
    no_punct=False,                 # remove punctuations
    replace_with_url="<url>",
    replace_with_email="<email>",
    replace_with_phone_number="<phone>",
    lang="en"                       # set to 'de' for German special handling
)



def process_dataset_with_doct5query(dataset_name):
    if dataset_name == "trec_covid":
        processed_dataset = {}

        ds = load_dataset("BeIR/trec-covid-generated-queries")

        for line in ds["train"]:
            doc_id = line.get("_id")

            if doc_id not in processed_dataset:
                processed_dataset[doc_id] = []


            query = line.get("query").lower()
            processed_dataset[doc_id].append(query)

        return processed_dataset
    elif dataset_name == "scifact":
        processed_dataset = {}

        ds = load_dataset("BeIR/scifact-generated-queries")

        for line in ds["train"]:
            doc_id = line.get("_id")

            if doc_id not in processed_dataset:
                processed_dataset[doc_id] = []


            query = line.get("query").lower()
            processed_dataset[doc_id].append(query)

        return processed_dataset 
    elif dataset_name == "scidocs":
        processed_dataset = {}

        ds = load_dataset("BeIR/scidocs-generated-queries")

        for line in ds["train"]:
            doc_id = line.get("_id")

            if doc_id not in processed_dataset:
                processed_dataset[doc_id] = []


            query = line.get("query").lower()
            processed_dataset[doc_id].append(query)

        return processed_dataset 
    elif dataset_name == "nfcorpus":
        processed_dataset = {}

        ds = load_dataset("BeIR/nfcorpus-generated-queries")

        for line in ds["train"]:
            doc_id = line.get("_id")

            if doc_id not in processed_dataset:
                processed_dataset[doc_id] = []


            query = line.get("query").lower()
            processed_dataset[doc_id].append(query)

        return processed_dataset 
    else:
        raise NotImplementedError


def process_dataset(dataset_name):
    # if dataset_name == "duc":
    #     df = pd.read_json("hf://datasets/memray/duc/test.json", lines=True)

    #     processed_dataset = []
    #     for line in df.to_dict("records"):
    #         doc_id = line.get("name")
    #         title = line.get("title")
    #         abstract = line.get("abstract")

    #         text = f"{title.lower()}\n{abstract.lower()}"
    #         text_not_lowered = f"{title}\n{abstract}"

    #         keyphrases = line.get("keywords").split(";")
    #         keyphrases = [kw.strip().lower() for kw in keyphrases if kw]

    #         present_keyphrases = [kw for kw in keyphrases if kw in text]
    #         absent_keyphrases = [kw for kw in keyphrases if kw not in text]

    #         processed_line = {
    #             "doc_id": doc_id,
    #             "text": text_not_lowered,
    #             "present_keyphrases": present_keyphrases,
    #             "absent_keyphrases": absent_keyphrases
    #         }

    #         processed_dataset.append(processed_line)
        
    #     return processed_dataset
    
    # elif dataset_name == "stackexchange":
    #     df = pd.read_json("hf://datasets/memray/stackexchange/test.json", lines = True)

    #     processed_dataset = []
    #     for line in df.to_dict("records"):
    #         doc_id = line.get("id")
    #         title = line.get("title")
    #         question = line.get("question")
    #         accepted_answer = line.get("accepted_answer") if isinstance(line.get('accepted_answer'), str) else ""

    #         text = f"{title.lower()}\n{question.lower()}\n{accepted_answer.lower()}"

    #         keyphrases = line.get("tags").split(";") if isinstance(line.get("tags"), str) else []
    #         keyphrases = [kw.strip().lower() for kw in keyphrases if kw]

    #         present_keyphrases = [kw for kw in keyphrases if kw in text]
    #         absent_keyphrases = [kw for kw in keyphrases if kw not in text]

    #         processed_line = {
    #             "doc_id": doc_id,
    #             "text": text,
    #             "present_keyphrases": present_keyphrases,
    #             "absent_keyphrases": absent_keyphrases
    #         }

    #         processed_dataset.append(processed_line)
        
    #     return processed_dataset
    # elif dataset_name == "kptimes":
    #     df = pd.read_json("hf://datasets/memray/kptimes/test.json", lines=True)

    #     processed_dataset = []
    #     for line in df.to_dict("records"):
    #         doc_id = line.get("id")
    #         title = line.get("title")
    #         abstract = line.get("abstract")

    #         text = f"{title.lower()}\n{abstract.lower()}"

    #         keyphrases = line.get("keyword").split(";")
    #         keyphrases = [kw.strip().lower() for kw in keyphrases if kw]

    #         present_keyphrases = [kw for kw in keyphrases if kw in text]
    #         absent_keyphrases = [kw for kw in keyphrases if kw not in text]

    #         processed_line = {
    #             "doc_id": doc_id,
    #             "text": text,
    #             "present_keyphrases": present_keyphrases,
    #             "absent_keyphrases": absent_keyphrases
    #         }

    #         processed_dataset.append(processed_line)
        
    #     return processed_dataset
    # elif dataset_name == "openkp":
    #     df = pd.read_json("hf://datasets/memray/openkp/test.json", lines=True)

    #     processed_dataset = []
    #     for line in df.to_dict("records"):
    #         doc_id = line.get("url")
    #         text = line.get("text").lower()

    #         keyphrases = line.get("KeyPhrases")
    #         keyphrases = [kw.strip().lower() for kw in keyphrases if kw]

    #         present_keyphrases = [kw for kw in keyphrases if kw in text]
    #         absent_keyphrases = [kw for kw in keyphrases if kw not in text]

    #         processed_line = {
    #             "doc_id": doc_id,
    #             "text": text,
    #             "present_keyphrases": present_keyphrases,
    #             "absent_keyphrases": absent_keyphrases
    #         }

    #         processed_dataset.append(processed_line)
        
    #     return processed_dataset
    if dataset_name == "kp20k":
        df = pd.read_json("hf://datasets/memray/kp20k/test.json", lines=True)

        processed_dataset = []
        for line in df.to_dict("records"):
            doc_id = line.get("id")
            title = line.get("title")
            abstract = line.get("abstract")

            text = f"{title.lower()}\n{abstract.lower()}"
            text_not_lowered = f"{title}\n{abstract}"
            

            keyphrases = line.get("keywords").split(";")
            assert isinstance(keyphrases, list)
            keyphrases = [kw.strip().lower() for kw in keyphrases if kw]

            present_keyphrases = [kw for kw in keyphrases if kw in text]
            absent_keyphrases = [kw for kw in keyphrases if kw not in text]

            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": present_keyphrases,
                "absent_keyphrases": absent_keyphrases
            }

            processed_dataset.append(processed_line)
        
        return processed_dataset
    elif dataset_name == "nus":
        df = pd.read_json("hf://datasets/memray/nus/test.json", lines=True)

        processed_dataset = []
        for line in df.to_dict("records"):
            doc_id = line.get("name")
            title = line.get("title")
            abstract = line.get("abstract")

            text = f"{title.lower()}\n{abstract.lower()}"
            text_not_lowered = f"{title}\n{abstract}"

            keyphrases = line.get("keywords").split(";")
            keyphrases = [kw.strip().lower() for kw in keyphrases if kw]

            present_keyphrases = [kw for kw in keyphrases if kw in text]
            absent_keyphrases = [kw for kw in keyphrases if kw not in text]

            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": present_keyphrases,
                "absent_keyphrases": absent_keyphrases
            }

            processed_dataset.append(processed_line)
        
        return processed_dataset

        # with open("/scratch/lamdo/unsupervised_keyphrase_prediction_2022/data/testing_datasets/NUS/stable/test.json") as f:
        #     data = json.load(f)

        # processed_dataset = []

        # for line in data:
        #     present_keyphrases = line.get("present_keyphrases")
        #     absent_keyphrases = line.get("absent_keyphrases")
        #     text  = clean_func(line.get("text"))

        #     processed_line = {
        #         "doc_id": None,
        #         "text": text,
        #         "present_keyphrases": present_keyphrases,
        #         "absent_keyphrases": absent_keyphrases
        #     }

        #     processed_dataset.append(processed_line)

        # return processed_dataset
    
    elif dataset_name == "semeval":
        df = pd.read_json("hf://datasets/memray/semeval/test.json", lines=True)

        processed_dataset = []
        for line in df.to_dict("records"):
            doc_id = line.get("name")
            title = line.get("title")
            abstract = line.get("abstract")

            text = f"{title.lower()}\n{abstract.lower()}"
            text_not_lowered = f"{title}\n{abstract}"

            keyphrases = line.get("keywords").split(";")
            keyphrases = [kw.strip().lower() for kw in keyphrases if kw]

            present_keyphrases = [kw for kw in keyphrases if kw in text]
            absent_keyphrases = [kw for kw in keyphrases if kw not in text]

            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": present_keyphrases,
                "absent_keyphrases": absent_keyphrases
            }

            processed_dataset.append(processed_line)
        
        return processed_dataset

        # with open("/scratch/lamdo/unsupervised_keyphrase_prediction_2022/data/testing_datasets/SemEval/stable/test.json") as f:
        #     data = json.load(f)

        # processed_dataset = []

        # for line in data:
        #     present_keyphrases = line.get("present_keyphrases")
        #     absent_keyphrases = line.get("absent_keyphrases")
        #     text  = clean_func(line.get("text"))

        #     processed_line = {
        #         "doc_id": None,
        #         "text": text,
        #         "present_keyphrases": present_keyphrases,
        #         "absent_keyphrases": absent_keyphrases
        #     }

        #     processed_dataset.append(processed_line)

        # return processed_dataset

    elif dataset_name == "inspec":
        df = pd.read_json("hf://datasets/memray/inspec/test.json", lines=True)

        processed_dataset = []
        for line in df.to_dict("records"):
            doc_id = line.get("name")
            title = line.get("title")
            abstract = line.get("abstract")

            text = f"{title.lower()}\n{abstract.lower()}"
            text_not_lowered = f"{title}\n{abstract}"

            keyphrases = line.get("keywords").split(";")
            keyphrases = [kw.strip().lower() for kw in keyphrases if kw]

            present_keyphrases = [kw for kw in keyphrases if kw in text]
            absent_keyphrases = [kw for kw in keyphrases if kw not in text]

            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": present_keyphrases,
                "absent_keyphrases": absent_keyphrases
            }

            processed_dataset.append(processed_line)
        
        return processed_dataset

        # with open("/scratch/lamdo/unsupervised_keyphrase_prediction_2022/data/testing_datasets/Inspec/stable/test.json") as f:
        #     data = json.load(f)

        # processed_dataset = []

        # for line in data:
        #     present_keyphrases = line.get("present_keyphrases")
        #     absent_keyphrases = line.get("absent_keyphrases")
        #     text  = clean_func(line.get("text"))

        #     processed_line = {
        #         "doc_id": None,
        #         "text": text,
        #         "present_keyphrases": present_keyphrases,
        #         "absent_keyphrases": absent_keyphrases
        #     }

        #     processed_dataset.append(processed_line)

        # return processed_dataset
    elif dataset_name == "krapivin":
        df = pd.read_json("hf://datasets/memray/krapivin/test.json", lines=True)

        processed_dataset = []
        for line in df.to_dict("records"):
            doc_id = line.get("name")
            title = line.get("title")
            abstract = line.get("abstract")

            text = f"{title.lower()}\n{abstract.lower()}"
            text_not_lowered = f"{title}\n{abstract}"

            keyphrases = line.get("keywords").split(";")
            keyphrases = [kw.strip().lower() for kw in keyphrases if kw]

            present_keyphrases = [kw for kw in keyphrases if kw in text]
            absent_keyphrases = [kw for kw in keyphrases if kw not in text]

            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": present_keyphrases,
                "absent_keyphrases": absent_keyphrases
            }

            processed_dataset.append(processed_line)
        
        return processed_dataset

        # with open("/scratch/lamdo/unsupervised_keyphrase_prediction_2022/data/testing_datasets/Krapivin/stable/test.json") as f:
        #     data = json.load(f)

        # processed_dataset = []

        # for line in data:
        #     present_keyphrases = line.get("present_keyphrases")
        #     absent_keyphrases = line.get("absent_keyphrases")
        #     text  = clean_func(line.get("text"))

        #     processed_line = {
        #         "doc_id": None,
        #         "text": text,
        #         "present_keyphrases": present_keyphrases,
        #         "absent_keyphrases": absent_keyphrases
        #     }

        #     processed_dataset.append(processed_line)

        # return processed_dataset
    
    # elif dataset_name == "nq320k":
    #     with open("/home/lamdo/keyphrase_informativeness_test/pyserini_test/data/nq320k/nq320k/corpus_lite.json") as f:
    #         corpus = json.load(f)

    #     processed_dataset = []
    #     for i, doc in enumerate(corpus):
    #         processed_line = {
    #             "doc_id": i,
    #             "text": doc,
    #             "present_keyphrases": [],
    #             "absent_keyphrases": []
    #         }
    #         processed_dataset.append(processed_line)

    #     return processed_dataset
    elif dataset_name == "scirepeval_search":
        processed_dataset = []
        with open("/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search.jsonl") as f:
            for line in tqdm(f):
                line = json.loads(line)
                title = line["title"]
                abstract = line["abstract"]
                doc_id = line["doc_id"]

                text = f"{title}. {abstract}"
                text_not_lowered = f"{title}\n{abstract}"
                processed_line = {
                    "doc_id": doc_id,
                    "text": text,
                    "text_not_lowered": text_not_lowered,
                    "present_keyphrases": [],
                    "absent_keyphrases": []
                }

                processed_dataset.append(processed_line)
        return processed_dataset
    
    elif dataset_name == "scirepeval_search_validation_evaluation":
        processed_dataset = []
        with open("/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_validation_evaluation.jsonl") as f:
            for line in tqdm(f):
                line = json.loads(line)
                title = line["title"]
                abstract = line["abstract"]
                doc_id = line["doc_id"]

                text = f"{title}. {abstract}"
                text_not_lowered = f"{title}\n{abstract}"
                processed_line = {
                    "doc_id": doc_id,
                    "text": text,
                    "text_not_lowered": text_not_lowered,
                    "present_keyphrases": [],
                    "absent_keyphrases": []
                }

                processed_dataset.append(processed_line)
        return processed_dataset
    
    elif dataset_name == "scirepeval_search_evaluation":
        processed_dataset = []
        with open("/scratch/lamdo/pyserini_experiments/scirepeval_collections/corpus/scirepeval_search_evaluation.jsonl") as f:
            for line in tqdm(f):
                line = json.loads(line)
                title = line["title"]
                abstract = line["abstract"]
                doc_id = line["doc_id"]

                text = f"{title}. {abstract}"
                text_not_lowered = f"{title}\n{abstract}"
                processed_line = {
                    "doc_id": doc_id,
                    "text": text,
                    "text_not_lowered": text_not_lowered,
                    "present_keyphrases": [],
                    "absent_keyphrases": []
                }

                processed_dataset.append(processed_line)
        return processed_dataset
    
    elif dataset_name == "scirepeval_search_evaluation_queries":
        processed_dataset = []
        ds = load_dataset("allenai/scirepeval", "search")

        ds_evaluation = ds["evaluation"]

        for line in ds_evaluation:
            query_id = line.get("doc_id")
            query = line.get("query", "")

            processed_line = {
                "doc_id": str(query_id),
                "text": query,
                "text_not_lowered": query,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset

    elif dataset_name == "scidocs":
        processed_dataset = []

        ds = load_dataset("BeIR/scidocs", "corpus")

        for line in ds["corpus"]:
            doc_id = line.get("_id")
            title = line.get("title")
            _text = line.get("text")

            text = f"{title.lower()}. {_text.lower()}"
            text_not_lowered = f"{title}\n{_text}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    elif dataset_name == "scidocs_queries":
        processed_dataset = []

        ds = load_dataset("BeIR/scidocs", "queries")

        for line in ds["queries"]:
            doc_id = line.get("_id")
            title = line.get("title")
            _text = line.get("text")

            text = f"{title.lower()}. {_text.lower()}"
            text_not_lowered = f"{title}\n{_text}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    elif dataset_name == "scifact":
        processed_dataset = []

        ds = load_dataset("BeIR/scifact", "corpus")

        for line in ds["corpus"]:
            doc_id = line.get("_id")
            title = line.get("title")
            _text = line.get("text")

            text = f"{title.lower()}. {_text.lower()}"
            text_not_lowered = f"{title}\n{_text}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    
    elif dataset_name == "scifact_queries":
        processed_dataset = []

        ds = load_dataset("BeIR/scifact", "queries")

        for line in ds["queries"]:
            doc_id = line.get("_id")
            title = line.get("title")
            _text = line.get("text")

            text = f"{_text.lower()}"
            text_not_lowered = f"{_text}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    

    elif dataset_name == "trec_covid":
        processed_dataset = []

        ds = load_dataset("BeIR/trec-covid", "corpus")

        for line in ds["corpus"]:
            doc_id = line.get("_id")
            title = line.get("title")
            _text = line.get("text")

            text = f"{title.lower()}. {_text.lower()}"
            text_not_lowered = f"{title}\n{_text}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    
    elif dataset_name == "trec_covid_queries":
        processed_dataset = []

        ds = load_dataset("BeIR/trec-covid", "queries")

        for line in ds["queries"]:
            doc_id = line.get("_id")
            # title = line.get("title")
            _text = line.get("text")

            text = f"{_text.lower()}"
            text_not_lowered = f"{_text}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    
    elif dataset_name == "doris_mae":
        processed_dataset = []

        with open("/scratch/lamdo/doris-mae/DORIS-MAE_dataset_v1.json") as f:
            data = json.load(f)
            corpus = data["Corpus"]
            del data

        for line in corpus:
            title = line.get("title")
            abstract = line.get("original_abstract")
            doc_id = line.get("")

            # process title
            title = title.replace("_", " ")
            abstract = abstract.replace("\n", " ")

            text = f"{title.lower()}. {abstract.lower()}"

            text_not_lowered = f"{title}\n{abstract}"

            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    
    elif dataset_name == "doris_mae_queries":
        processed_dataset = []

        with open("/scratch/lamdo/doris-mae/DORIS-MAE_dataset_v1.json") as f:
            data = json.load(f)
            queries = data["Query"]
            del data

        for i, line in enumerate(queries):
            query_text = line.get("query_text", "")

            doc_id = str(i)

            text = query_text.lower()

            text_not_lowered = query_text

            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset

    elif dataset_name == "nfcorpus":
        processed_dataset = []

        ds = load_dataset("BeIR/nfcorpus", "corpus")

        for line in ds["corpus"]:
            doc_id = line.get("_id")
            title = line.get("title")
            _text = line.get("text")

            text = f"{title.lower()}. {_text.lower()}"
            text_not_lowered = f"{title}\n{_text}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    elif dataset_name == "nfcorpus_queries":
        processed_dataset = []

        ds = load_dataset("BeIR/nfcorpus", "queries")

        for line in ds["queries"]:
            doc_id = line.get("_id")
            # title = line.get("title")
            _text = line.get("text")

            text = f"{_text.lower()}"
            text_not_lowered = f"{_text}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    
    # elif dataset_name == "scirepeval_fos_100k":
    #     processed_dataset = []

    #     with open("/scratch/lamdo/scirepeval_classification/fos/train_100k.jsonl") as f:
    #         for line in f:
    #             jline = json.loads(line)

    #             doc_id = jline.get("doc_id")
    #             title = jline.get("title")
    #             abstract = jline.get("abstract")

    #             text = f"{title.lower()}. {abstract.lower()}"
    #             text_not_lowered = f"{title}\n{abstract}"

    #             processed_line = {
    #                 "doc_id": doc_id,
    #                 "text": text,
    #                 "text_not_lowered": text_not_lowered,
    #                 "present_keyphrases": [],
    #                 "absent_keyphrases": []
    #             }

    #             processed_dataset.append(processed_line)

    #     return processed_dataset
    

    # elif dataset_name == "scirepeval_mesh_descriptors_100k":
    #     processed_dataset = []

    #     with open("/scratch/lamdo/scirepeval_classification/mesh_descriptors/train_100k.jsonl") as f:
    #         for line in f:
    #             jline = json.loads(line)

    #             doc_id = jline.get("doc_id")
    #             title = jline.get("title")
    #             abstract = jline.get("abstract")

    #             text = f"{title.lower()}. {abstract.lower()}"
    #             text_not_lowered = f"{title}\n{abstract}"

    #             processed_line = {
    #                 "doc_id": doc_id,
    #                 "text": text,
    #                 "text_not_lowered": text_not_lowered,
    #                 "present_keyphrases": [],
    #                 "absent_keyphrases": []
    #             }

    #             processed_dataset.append(processed_line)

    #     return processed_dataset
    
    elif dataset_name == "scirepeval_fos_test":
        processed_dataset = []

        ds = load_dataset("allenai/scirepeval", "fos")

        for line in ds["evaluation"]:
            doc_id = line.get("doc_id")
            title = line.get("title")
            abstract = line.get("abstract")

            text = f"{title.lower()}. {abstract.lower()}"
            text_not_lowered = f"{title}. {abstract}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    
    elif dataset_name == "scirepeval_fos_test_title":
        processed_dataset = []

        ds = load_dataset("allenai/scirepeval", "fos")

        for line in ds["evaluation"]:
            doc_id = line.get("doc_id")
            title = line.get("title")
            # abstract = line.get("abstract")

            text = f"{title.lower()}"
            text_not_lowered = f"{title}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    
    elif dataset_name == "scirepeval_mesh_descriptors_test":
        processed_dataset = []

        ds = load_dataset("allenai/scirepeval", "mesh_descriptors")

        for line in ds["evaluation"]:
            doc_id = line.get("doc_id")
            title = line.get("title")
            abstract = line.get("abstract")

            text = f"{title.lower()}. {abstract.lower()}"
            text_not_lowered = f"{title}. {abstract}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    

    elif dataset_name == "acm_cr":
        processed_dataset = []

        with open("/scratch/lamdo/acm-cr/data/docs/collection.jsonl") as f:
            for line in f:
                jline = json.loads(line)

                doc_id = jline.get("id")
                title = jline.get("title")
                abstract = jline.get("abstract")

                text = f"{title.lower()}. {abstract.lower()}"
                text_not_lowered = f"{title}. {abstract}"
                processed_line = {
                    "doc_id": doc_id,
                    "text": text,
                    "text_not_lowered": text_not_lowered,
                    "present_keyphrases": [],
                    "absent_keyphrases": []
                }

                processed_dataset.append(processed_line)

        return processed_dataset

    elif dataset_name == "acm_cr_queries":
        processed_dataset = []

        with open("/scratch/lamdo/acm-cr/data/topics+qrels/sentences.jsonl") as f:
            for line in f:
                jline = json.loads(line)

                doc_id = jline.get("id")
                text = jline.get("context").lower()
                text_not_lowered = jline.get("context")
                
                processed_line = {
                    "doc_id": doc_id,
                    "text": text,
                    "text_not_lowered": text_not_lowered,
                    "present_keyphrases": [],
                    "absent_keyphrases": []
                }

                processed_dataset.append(processed_line)

        return processed_dataset
    elif dataset_name == "arxiv_classification":
        from datasets import load_from_disk

        processed_dataset = []

        ds = load_from_disk("/scratch/lamdo/arxiv_classification/arxiv_classification_20k/")

        for line in ds["evaluation"]:
            doc_id = line.get("doc_id")
            title = line.get("title")
            abstract = line.get("abstract")

            text = f"{title.lower()}. {abstract.lower()}"
            text_not_lowered = f"{title}. {abstract}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    
    elif dataset_name == "arxiv_classification_title":
        from datasets import load_from_disk

        processed_dataset = []

        ds = load_from_disk("/scratch/lamdo/arxiv_classification/arxiv_classification_20k/")

        for line in ds["evaluation"]:
            doc_id = line.get("doc_id")
            title = line.get("title")
            # abstract = line.get("abstract")

            text = f"{title.lower()}"
            text_not_lowered = f"{title}"
            processed_line = {
                "doc_id": doc_id,
                "text": text,
                "text_not_lowered": text_not_lowered,
                "present_keyphrases": [],
                "absent_keyphrases": []
            }

            processed_dataset.append(processed_line)
        return processed_dataset
    elif dataset_name == "combined_kg":
        processed_dataset = []
        for ds_name in ["semeval", "inspec", "nus", "krapivin"]:
            df = pd.read_json(f"hf://datasets/memray/{ds_name}/test.json", lines=True)

            for line in df.to_dict("records"):
                doc_id = line.get("name")
                title = line.get("title")
                abstract = line.get("abstract")

                text = f"{title.lower()}\n{abstract.lower()}"
                text_not_lowered = f"{title}\n{abstract}"

                keyphrases = line.get("keywords").split(";")
                keyphrases = [kw.strip().lower() for kw in keyphrases if kw]

                present_keyphrases = [kw for kw in keyphrases if kw in text]
                absent_keyphrases = [kw for kw in keyphrases if kw not in text]

                processed_line = {
                    "doc_id": doc_id,
                    "text": text,
                    "text_not_lowered": text_not_lowered,
                    "present_keyphrases": present_keyphrases,
                    "absent_keyphrases": absent_keyphrases
                }

                processed_dataset.append(processed_line)
        
        return processed_dataset
    else:
        raise NotImplementedError