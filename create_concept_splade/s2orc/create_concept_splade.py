import json, torch, string, re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from nltk.corpus import stopwords
from functools import lru_cache

STOPWORDS = stopwords.words('english')

@lru_cache(maxsize=16384)
def simple_tokenize(text: str, lower = False, remove_title = True):
    """
    Tokenizes the given text into a list of tokens.
    
    Args:
        text (str): The input text to be tokenized.
        lower (bool, optional): Whether to convert the tokens to lowercase. Defaults to False.
        remove_title (bool, optional): Whether to remove academic titles from the tokens. Defaults to True.
    
    Returns:
        list: A list of tokenized tokens extracted from the input text.
    """
    # Function implementation goes here
    if not text: return []
    if not isinstance(text, str): text = str(text)
    if lower:
        res = [
            tok.strip(string.punctuation).strip("\n").lower() for tok in re.split(r"[-\,\(\)\s]+", text)
        ]
    else:
        res = [tok.strip(string.punctuation).strip("\n") for tok in re.split(r"[-\,\(\)\s]+", text)]

    return [tok for tok in res if tok]


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


def create_model_with_added_phrase_vocab(phrase_vocab, model_name = "distilbert/distilbert-base-uncased", max_added_phrases = 20000):
    assert model_name in ["distilbert/distilbert-base-uncased", "bert-base-uncased"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)   

    with torch.no_grad():

        model_vocab = tokenizer.vocab
        phrases_to_add = [phrase for phrase in tqdm(phrase_vocab, desc = "Filtering phrases (check with vocab)") if not check_exist_in_vocab(phrase, model_vocab)]
        # phrases_to_add_ = [phrases_to_add[i] for i in tqdm(range(len(phrases_to_add)), desc = "Filtering phrases (check with self)") if not check_exist_in_phrase_vocab(phrases_to_add[i], phrases_to_add[:i])][:max_added_phrases]
        # phrases_to_add = phrases_to_add
        # with open("created_vocab.json", "w") as f:
        #     json.dump(phrases_to_add, f, indent = 4)

        num_added_toks = tokenizer.add_tokens(phrases_to_add)
        print(f"Added {num_added_toks} tokens")

        model.resize_token_embeddings(len(tokenizer))
        
        return tokenizer, model



if __name__ == "__main__":
    # vocab_path = "/home/lamdo/splade/create_concept_splade/vocab_create/phrase_vocab_s2orc_gitig_.json"
    # vocab_path = "/home/lamdo/splade/create_concept_splade/vocab_create/word_vocab_s2orc_gitig_.json"
    vocab_path = "/home/lamdo/splade/create_concept_splade/s2orc/phrase_vocab_30k.json"

    with open(vocab_path) as f:
        phrase_counter = json.load(f)

    if isinstance(phrase_counter, dict):
        phrase_vocab = list(phrase_counter.keys())
    else:
        phrase_vocab = phrase_counter

    tokenizer, model = create_model_with_added_phrase_vocab(
        phrase_vocab=phrase_vocab,
        max_added_phrases=60000,
        model_name="distilbert/distilbert-base-uncased"
    )

    print(model.distilbert.embeddings.word_embeddings.weight.shape)

    model_name_on_hf = "lamdo/distilbert-base-uncased-phrase-30kaddedphrasesfroms2orcfreqbased"
    tokenizer.push_to_hub(model_name_on_hf)
    model.push_to_hub(model_name_on_hf)

    # model_name = "/scratch/lamdo/distilbert-base-uncased-word-30kaddedwordsfroms2orc/"
    # tokenizer.save_pretrained(model_name)
    # model.save_pretrained(model_name)