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


def create_embedding_and_mlm_for_added_phrases(phrases_to_add, model, tokenizer, to_skip_tokens):
    tokenized_phrases_to_add = [tokenizer(phrase, add_special_tokens=False) for phrase in phrases_to_add]

    phrases_to_add_embeddings = []

    word_embeddings = model.distilbert.embeddings.word_embeddings.weight.data.detach()

    for i in range(len(phrases_to_add)):
        input_ids = tokenized_phrases_to_add[i]["input_ids"]
        input_ids = [inpid for inpid in input_ids if inpid not in to_skip_tokens]

        phrase_embedding = torch.mean(word_embeddings[input_ids], dim = 0)

        phrases_to_add_embeddings.append(phrase_embedding)


    return {
        "embeddings": phrases_to_add_embeddings
    }


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
        with open("created_vocab.json", "w") as f:
            json.dump(phrases_to_add, f, indent = 4)

        num_added_toks = tokenizer.add_tokens(phrases_to_add)
        print(f"Added {num_added_toks} tokens")

        model.resize_token_embeddings(len(tokenizer))
        
        return tokenizer, model



if __name__ == "__main__":
    vocab_path = "/home/lamdo/splade/create_concept_splade/s2orc/phrase_vocab.json"

    with open(vocab_path) as f:
        phrase_counter = json.load(f)

    phrase_vocab = list(phrase_counter.keys())

    tokenizer, model = create_model_with_added_phrase_vocab(
        phrase_vocab=phrase_vocab
    )

    print(model.distilbert.embeddings.word_embeddings.weight.shape)

    model_name_on_hf = "lamdo/distilbert-base-uncased-phrase-16kaddedphrasesfroms2orc"
    tokenizer.push_to_hub(model_name_on_hf)
    model.push_to_hub(model_name_on_hf)
