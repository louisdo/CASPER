import json, os
import torch
from pyserini.search.lucene import LuceneSearcher
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm 
import  string
from nltk.corpus import stopwords
from collections import Counter
STOPWORDS = stopwords.words('english')

# Lucene index
index_path='/home/lvnguyen/research/eru-kg-base'
  # Lucene index path
based_bert_path='distilbert/distilbert-base-uncased'

DEVICE = "cuda"
searcher = LuceneSearcher(index_path)
tokenizer = DistilBertTokenizer.from_pretrained(based_bert_path)
model = DistilBertModel.from_pretrained(based_bert_path).to(DEVICE)
vocab = tokenizer.vocab
to_skip_tokens = [vocab.get(punc) for punc in string.punctuation]  + [vocab.get(word) for word in STOPWORDS]
to_skip_tokens = set([tok for tok in to_skip_tokens if tok is not None])

def remove_stopwords(texts):
    cleaned_texts = []
    
    for text in texts:
        words = text.split()

        cleaned_words = [word for word in words if word.lower() not in STOPWORDS]
        
        cleaned_texts.append(' '.join(cleaned_words))
    
    return cleaned_texts

# embeddings for a batch of documents
def compute_embeddings_batch(texts):
    texts = remove_stopwords(texts)
    inputs = tokenizer(texts, return_tensors='pt', add_special_tokens=False, padding=True, truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        input_ids = inputs['input_ids']
        # print(input_ids.shape)

        # #option1: mean tokens embeddings - mean of model.embeddings.word_embeddings(input_ids)
        # print(model.embeddings.word_embeddings(torch.tensor(input_ids)).shape)
        embeddings = torch.mean(model.embeddings.word_embeddings(input_ids), dim=0)
        # print(embeddings.shape)
    
    return embeddings.squeeze().tolist()[0]

if __name__ == "__main__": 
    keywords_to_run = json.load(open("test_gitig_.json"))
    keywords_to_run = {item:[] for item in keywords_to_run}

    total_docs = searcher.num_docs
    for doc_id in tqdm(range(total_docs)): 
        lucene_doc = searcher.doc(doc_id).lucene_document()
        lucene_doc_raw_content = json.loads(lucene_doc.get('raw'))
        for keyword in keywords_to_run: 
            if keyword in lucene_doc_raw_content['present_keyphrases']:
                keywords_to_run[keyword].extend(lucene_doc_raw_content['present_keyphrases'])

    for keyword in keywords_to_run: 
        most_common_keywords = Counter(keywords_to_run[keyword]).most_common(5)
        text = " ".join([item[0] for item in most_common_keywords])
        keywords_to_run[keyword] = {
            "keyphrase": keyword, 
            "related keyphrases":  most_common_keywords, 
            "text": text,
            "embedding": compute_embeddings_batch([text])
        }

    json.dump(list(keywords_to_run.values()), open("_gitig_data/keyphrases_with_embeddings_cooc_all.json", 'w'), indent=4)