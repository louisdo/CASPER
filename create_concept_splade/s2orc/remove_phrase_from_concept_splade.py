import json, torch, string, re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from nltk.corpus import stopwords


ORIGINAL_BERT_VOCAB_SIZE=30522

def select_single_word_phrases_from_vocab(huggingface_vocab):
    selected_tokens_string = []
    selected_tokens_index = []
    for token, index in huggingface_vocab.items():
        if index >= ORIGINAL_BERT_VOCAB_SIZE and " " in token:
            selected_tokens_string.append(token)
            selected_tokens_index.append(index)

    return selected_tokens_string, selected_tokens_index


def remove_added_phrases_from_model(to_modify_model_name):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained(to_modify_model_name)

    model.resize_token_embeddings(ORIGINAL_BERT_VOCAB_SIZE)

    return tokenizer, model






def create_model_by_deleting_single_word_phrases(to_modify_model_name = "lamdo/distilbert-base-uncased-phrase-16kaddedphrasesfroms2orc-mlm-80000steps"):

    tokenizer, model = remove_added_phrases_from_model(to_modify_model_name)


    tokenizer_to_modify = AutoTokenizer.from_pretrained(to_modify_model_name)
    model_to_modify = AutoModelForMaskedLM.from_pretrained(to_modify_model_name)


    selected_tokens_string, selected_tokens_index = select_single_word_phrases_from_vocab(tokenizer_to_modify.vocab)

    with torch.no_grad():

        

        num_added_toks = tokenizer.add_tokens(selected_tokens_string)
        print(f"Added {num_added_toks} tokens")

        model.resize_token_embeddings(len(tokenizer))
        new_model_vocab = tokenizer.vocab

        for phrase, phrase_index_in_model_to_modify in zip(selected_tokens_string, selected_tokens_index):
            index = new_model_vocab[phrase]
            model.distilbert.embeddings.word_embeddings.weight.data[index] = model_to_modify.distilbert.embeddings.word_embeddings.weight.data[phrase_index_in_model_to_modify]

        
        return tokenizer, model


def create_model_by_deleting_all_phrases(to_modify_model_name = "lamdo/distilbert-base-uncased-phrase-16kaddedphrasesfroms2orc-mlm-80000steps"):
    tokenizer, model = remove_added_phrases_from_model(to_modify_model_name)

    return tokenizer, model

    

if __name__ == "__main__":
    # tokenizer, model = create_model_by_deleting_single_word_phrases(to_modify_model_name="/scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_16kaddedphrasesfroms2orc/")

    # model_name_on_hf = "lamdo/distilbert-base-uncased-phrase-16kaddedphrasesfroms2orc-mlm-150000steps-multiwords"
    # tokenizer.push_to_hub(model_name_on_hf)
    # model.push_to_hub(model_name_on_hf)

    tokenizer, model = create_model_by_deleting_all_phrases(to_modify_model_name="/scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_16kaddedphrasesfroms2orc/")
    model_name_on_hf = "lamdo/distilbert-base-uncased-phrase-16kaddedphrasesfroms2orc-mlm-150000steps-nophrases"
    tokenizer.push_to_hub(model_name_on_hf)
    model.push_to_hub(model_name_on_hf)