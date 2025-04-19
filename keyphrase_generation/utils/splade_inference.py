import torch, sys, traceback
sys.path.append("/home/lamdo/keyphrase_informativeness_test/splade")
from transformers import AutoModelForMaskedLM, AutoTokenizer
from collections import Counter
from splade.models.transformer_rep import Splade


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLADE_MODEL = {}

def init_splade_model(model_name):
    if SPLADE_MODEL.get(model_name) is not None:
        return 
    else:
        print(f"Init splade model {model_name}. This will be done only once")

    if model_name == "custom_trained_pubmedqa+specter":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_pubmedqa+specter/debug/checkpoint/model"

    elif model_name == "custom_trained_msmarco":
        model_type_or_dir = "/home/lamdo/keyphrase_informativeness_test/splade/experiments_msmarco/debug/checkpoint/model"

    elif model_name == "splade-cocondenser-ensembledistil":
        model_type_or_dir = "naver/splade-cocondenser-ensembledistil"

    elif model_name == "splade-cocondenser-selfdistil":
        model_type_or_dir = "naver/splade-cocondenser-selfdistil"
    
    elif model_name == "phrase_splade": 
        model_type_or_dir = "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_25/debug/checkpoint/model"

    elif model_name == "phrase_splade_26": 
        model_type_or_dir = "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_26/debug/checkpoint/model"

    elif model_name == "phrase_splade_27": 
        model_type_or_dir = "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_27/debug/checkpoint/model"

    elif model_name == "phrase_splade_28": 
        model_type_or_dir = "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_28/debug/checkpoint/model"
    elif model_name == "phrase_splade_29": 
        model_type_or_dir = "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_29/debug/checkpoint/model"
    elif model_name == "phrase_splade_30": 
        model_type_or_dir = "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_30/debug/checkpoint/model"

    elif model_name == "phrase_splade_24": 
        model_type_or_dir = "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_24/debug/checkpoint/model"

    elif model_name == "phrase_splade_12": 
        model_type_or_dir = "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_12/debug/checkpoint/model"

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


def get_tokens_scores_of_docs_batch(docs_tokens, model_name):
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
    # except Exception as e:
    #     print("Error in getting token score (informativeness module)", traceback.format_exc(), col, weights, )
    #     return [Counter() for _ in range(len(docs_tokens.input_ids))]