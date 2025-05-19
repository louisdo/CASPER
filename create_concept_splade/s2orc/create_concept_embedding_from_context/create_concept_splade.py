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


def create_model_with_added_phrase_vocab(phrase_vocab, phrase2embs, model_name = "distilbert/distilbert-base-uncased", max_added_phrases = 20000):
    assert model_name in ["distilbert/distilbert-base-uncased", "bert-base-uncased"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)   

    with torch.no_grad():

        model_vocab = tokenizer.vocab
        phrases_to_add = [phrase for phrase in tqdm(phrase_vocab, desc = "Filtering phrases (check with vocab)") if not check_exist_in_vocab(phrase, model_vocab)]
        phrases_to_add_ = [phrases_to_add[i] for i in tqdm(range(len(phrases_to_add)), desc = "Filtering phrases (check with self)") if not check_exist_in_phrase_vocab(phrases_to_add[i], phrases_to_add[:i])][:max_added_phrases]
        phrases_to_add = phrases_to_add_

        num_added_toks = tokenizer.add_tokens(phrases_to_add)
        print(f"Added {num_added_toks} tokens")


        avg_embedding_norm = torch.norm(model.distilbert.embeddings.word_embeddings.weight.data, dim = -1).mean()
        avg_vocab_projector_bias = model.vocab_projector.weight.data.mean()

        model.resize_token_embeddings(len(tokenizer))


        model_vocab = tokenizer.vocab

        for phrase in phrases_to_add:
            if phrase not in model_vocab: continue
            emb = torch.tensor(phrase2embs[phrase])
            emb /= torch.norm(emb)
            emb *= avg_embedding_norm
            phrase_idx = model_vocab[phrase]
            model.distilbert.embeddings.word_embeddings.weight.data[phrase_idx] = emb
            model.vocab_projector.weight.data[phrase_idx] = avg_vocab_projector_bias
        
        return tokenizer, model



if __name__ == "__main__":
    vocab_path = "/scratch/lamdo/phrase_mask_embeddings_.jsonl"
    
    phrase_metadata = []
    with open(vocab_path) as f:
        for line in tqdm(f, desc = "reading input"):
            jline = json.loads(line)
            phrase_metadata.append(jline)

    phrase_vocab = [item["phrase"] for item in phrase_metadata]
    phrase2embs = {item["phrase"]: item["emb"] for item in phrase_metadata}

    tokenizer, model = create_model_with_added_phrase_vocab(
        phrase_vocab=phrase_vocab,
        phrase2embs=phrase2embs,
        max_added_phrases=60000,
        model_name="distilbert/distilbert-base-uncased"
    )

    print(model.distilbert.embeddings.word_embeddings.weight.shape)

    # model_name_on_hf = "lamdo/bert-base-uncased-phrase-60kaddedphrasesfroms2orc"
    # tokenizer.push_to_hub(model_name_on_hf)
    # model.push_to_hub(model_name_on_hf)


    model.save_pretrained("model_gitig_")
    tokenizer.save_pretrained("model_gitig_")