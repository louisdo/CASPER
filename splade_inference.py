import torch, os, string
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import SpladeMaxSim, Splade
from collections import Counter
import re 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
model_type_or_dir = "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_58/debug/checkpoint/model"

# loading model and tokenizer

model = Splade(model_type_or_dir, agg="max").to("cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}


def encode_custom(tokens, model, is_q = False):
    out = model.encode_(tokens, is_q)["logits"]  # shape (bs, pad_len, voc_size)
    out = torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1)

    # mask = ~torch.isin(tokens["input_ids"], PUNCID)
    # out = out * mask.unsqueeze(-1)

    res = torch.zeros_like(out)
    res = res.to(out.device)

    out, token_indices = torch.max(out, dim = 1)


    res.scatter_(1, token_indices.unsqueeze(1), out.unsqueeze(1))
    return res


PUNCID = torch.tensor([tokenizer.vocab[punc] for punc in string.punctuation])
def encode_custom_mask_punc(tokens, model, is_q = False):
    out = model.encode_(tokens, is_q)["logits"]  # shape (bs, pad_len, voc_size)
    out = torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1)

    mask = ~torch.isin(tokens["input_ids"], PUNCID)
    out = out * mask.unsqueeze(-1)

    res = torch.zeros_like(out)
    res = res.to(out.device)

    out, token_indices = torch.max(out, dim = 1)

    return out

def main(doc): 
        doc = doc.translate(str.maketrans('', '', string.punctuation))
            
        doc_tokens = tokenizer(doc, max_length = 256, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            doc_rep = model(d_kwargs=doc_tokens)["d_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)

        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        d = {k: v for k, v in zip(col, weights)}
        sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
        bow_rep = []
        for k, v in sorted_d.items():
            if type(reverse_voc[k]) != str: continue
            if reverse_voc[k].replace('.', '').strip() == "": continue
            if v > 0:
                bow_rep.append((k, round(v, 2)))
        
        return  bow_rep[:50]

if __name__ == "__main__": 
    doc = "natural language processing in medicine"

    from elasticsearch import Elasticsearch

    # Connect to Elasticsearch
    es = Elasticsearch("")

    def query_elas(doc): 
        key_value = (main(doc))
        # key_value = [("term1", weight1), ("term2", weight2), ...]
        terms = [key[0] for key in key_value]
        query_terms = {key[0]: key[1] for key in key_value}
        min_should_match = int(min(4, len(key_value) / 5))  

        query = {
            "size": 20,
            "query": {
                "script_score": {
                    "query": {
                        "terms_set": {
                            "sparse_vector_terms": {
                                "terms": terms,
                                "minimum_should_match_script": {
                                    "source": str(min_should_match)
                                }
                            }
                        }
                    },
                    "script": {
                        "source": """
                            double score = 0;
                            List terms = params['_source']['sparse_vector_terms'];
                            List weights = params['_source']['sparse_vector_weights'];
                            Map query_terms = params['query_terms'];
                            for (int i = 0; i < terms.size(); i++) {
                                String term = terms.get(i).toString();  // Fix: convert int to string
                                if (query_terms.containsKey(term)) {
                                    score += weights.get(i) * query_terms.get(term);
                                }
                            }
                            return score;
                        """,
                        "params": {
                            "query_terms": query_terms
                        }
                    }
                }
            }
        }

        # Send the query
        response = es.search(
            index="openalex_compressed_papers_test",
            body=query
        )

        # Print results
        print(f"query: {doc}")
        print(f"took: {response['took']/1000}s")
        for hit in response["hits"]["hits"][:10]:
            print(f"ID: {hit['_id']}, Score: {hit['_score']}, Title: {hit['_source'].get('title')}")
        

    docs = [
        "Medicine Information Retrieval System", 
        "Open-domain Keyphrase generation", 
        "Graph base for name disambiguation", 
        "Scientific Document Retrieval"
    ]
    for doc in docs: 
        query_elas(doc)
    