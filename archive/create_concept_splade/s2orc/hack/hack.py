import os
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/scratch/lamdo/distilbert-base-uncased-word-30kaddedwordsfroms2orc/")

print("Vocab_size:", len(tokenizer.vocab))

with open('/scratch/lamdo/distilbert-base-uncased-word-30kaddedwordsfroms2orc/added_tokens.json') as json_file:
    added_tokens = json.load(json_file)
    sorted_tokens = dict(sorted(added_tokens.items(), key=lambda item: item[1]))
    for tk in sorted_tokens.keys():
        print(tk)
