import json, torch, string
from collections import Counter
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from nltk.corpus import stopwords
import os 

STOPWORDS = stopwords.words('english')

def create_phrase_vocab(path, threshold=32): 
    keywords = json.load(open(path))
    pre_process_phrases_to_add_embeddings = {item['keyword']:item['embedding'] for item in keywords if len(item.get("doc_ids", [])) >= threshold}
    return pre_process_phrases_to_add_embeddings

def check_exist_in_vocab(phrase, model_vocab):
    if phrase in model_vocab: return True

    for item in model_vocab:
        if phrase in item: return True

    return False

def check_exist_in_phrase_vocab(phrase, phrase_vocab):
    if phrase in phrase_vocab: return True

    for item in phrase_vocab:
        if item in phrase: return True

    return False


def create_model_with_added_phrase_vocab(phrase_vocab, pre_process_phrases_to_add_embeddings, model_name = "distilbert/distilbert-base-uncased", max_added_phrases = 20000):
    assert model_name == "distilbert/distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)   

    vocab = tokenizer.vocab
    to_skip_tokens = [vocab.get(punc) for punc in string.punctuation]  + [vocab.get(word) for word in STOPWORDS]
    to_skip_tokens = set([tok for tok in to_skip_tokens if tok is not None])

    with torch.no_grad():

        model_vocab = tokenizer.vocab
        phrases_to_add = [phrase for phrase in tqdm(phrase_vocab, desc = "Filtering phrases (check with vocab)") if not check_exist_in_vocab(phrase, model_vocab)]
        phrases_to_add_ = [phrases_to_add[i] for i in tqdm(range(len(phrases_to_add)), desc = "Filtering phrases (check with self)") if not check_exist_in_phrase_vocab(phrases_to_add[i], phrases_to_add[:i])][:max_added_phrases]
        phrases_to_add = phrases_to_add_
        with open("test_gitig_.json", "w") as f:
            json.dump(phrases_to_add, f, indent = 4)


        num_added_toks = tokenizer.add_tokens(phrases_to_add)
        print(f"Added {num_added_toks} tokens")

        model.resize_token_embeddings(len(tokenizer))

        model_vocab = tokenizer.vocab

        for phrase in tqdm(phrases_to_add):
            index = model_vocab[phrase]

            model.distilbert.embeddings.word_embeddings.weight.data[index] = torch.tensor(pre_process_phrases_to_add_embeddings[phrase])
        
        return tokenizer, model


if __name__ == "__main__":
    # keywords_with_embeddings_path = os.environ['keywords_with_embeddings_path']
    keywords_with_embeddings_path = "/home/lvnguyen/research/Efficient-Storage-for-Large-IR-Systems/create_concept_splade/_gitig_data/keyphrases_with_embeddings_cooc_all.json"
    pre_process_phrases_to_add_embeddings = create_phrase_vocab(keywords_with_embeddings_path, threshold=0)

    tokenizer, model = create_model_with_added_phrase_vocab(
        phrase_vocab=list(pre_process_phrases_to_add_embeddings.keys()), 
        pre_process_phrases_to_add_embeddings=pre_process_phrases_to_add_embeddings, 
        model_name = os.environ['based_bert_path']
    )

    print(model.distilbert.embeddings.word_embeddings.weight.shape)

    model_name_on_hf = "linh101201/distilbert-base-uncased-phrase-v2"
    tokenizer.push_to_hub(model_name_on_hf)
    model.push_to_hub(model_name_on_hf)