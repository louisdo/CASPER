import torch, os, sys, traceback
sys.path.append("../../")
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import SpladeMaxSim, Splade
from collections import Counter

SPLADE_MODEL_NAME = "lamdo/eru-kg-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLADE_MODEL = {}

def init_splade_model(model_name = SPLADE_MODEL_NAME):
    if model_name == "lamdo/eru-kg-base":
        model_type_or_dir = "lamdo/eru-kg-base"

    else: raise NotImplementedError

    print(f"Using {model_type_or_dir}")

    # if model_name in ["custom_trained_combined_references_v9"]:
    #     model = SpladeSoftPlus(model_type_or_dir, agg="max")
    # else:
    
    model = Splade(model_type_or_dir, agg="max")
    model = model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

    SPLADE_MODEL[model_name] = {
        "model": model,
        "tokenizer": tokenizer,
        "reverse_voc": reverse_voc
    }

def get_tokens_scores_of_docs_batch(docs_tokens, model_name = SPLADE_MODEL_NAME):
    # try:
    # Compute the document representations for the batch
    with torch.no_grad():
        docs_rep = SPLADE_MODEL[model_name]["model"](d_kwargs=docs_tokens.to(DEVICE))["d_rep"].cpu()  # shape (batch_size, 30522)

    batch_results = []
    
    for doc_rep in docs_rep:
        try:
            # Get the number of non-zero dimensions in the rep:
            col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
            
            # Get the weights for non-zero dimensions
            weights = doc_rep[col].cpu().tolist()

            if not isinstance(weights, list) and not isinstance(col, list):
                weights = [weights]
                col = [col]
            
            # Create a dictionary of dimension indices and their weights
            d = {k: v for k, v in zip(col, weights)}
            sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
            
            # Create the BOW representation
            bow_rep = []
            for k, v in sorted_d.items():
                bow_rep.append((SPLADE_MODEL[model_name]["reverse_voc"][k], round(v, 2)))
            
            # Add the Counter object to the batch_results
            batch_results.append(Counter({line[0]: line[1] for line in bow_rep}))
        except Exception as e:
            print("Error in getting token score (informativeness module)", traceback.format_exc(), col, weights)
            batch_results.append(Counter())

    return batch_results


def scores_candidates(candidates, tokens_scores, model_name = SPLADE_MODEL_NAME, length_penalty = -0.25):
    tokenized_candidates = [SPLADE_MODEL[model_name]["tokenizer"].tokenize(candidate) for candidate in candidates]
    scores = [sum(tokens_scores[tok] for tok in tokenized_cand) / (len(tokenized_cand) - length_penalty) for tokenized_cand in tokenized_candidates]

    return scores

if __name__ == "__main__":
    init_splade_model()

    docs_tokens = SPLADE_MODEL[SPLADE_MODEL_NAME]["tokenizer"](["this is a test", "this is another test"], return_tensors="pt", max_length = 512, padding = True, truncation = True)
    batch_tokens_scores = get_tokens_scores_of_docs_batch(docs_tokens, model_name = SPLADE_MODEL_NAME)

    candidates = ["this is", "a test"]
    scores = scores_candidates(candidates, tokens_scores = batch_tokens_scores[0])

    print([[c,s] for c,s in zip(candidates, scores)])
    print(batch_tokens_scores)