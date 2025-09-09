import json, os
import torch
from pyserini.search.lucene import LuceneSearcher
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm 
import  string
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')

# Lucene index
index_path = os.environ['index_path']  # Lucene index path
based_bert_path = os.environ['based_bert_path']
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
    
    return embeddings.squeeze().tolist()

#main function: Compute embeddings for all documents by batch
def process_all_documents(batch_size=256, saved_path="_gitig_data/docs_with_embeddings_all.jsonl"):
    
    total_docs = searcher.num_docs # total number of documents in the index

    for start_idx in tqdm(range(0, total_docs, batch_size)):
        docs_with_embeddings = []
        end_idx = min(start_idx + batch_size, total_docs)
        
        # Collect the batch of raw texts and their IDs
        batch_texts = []
        doc_ids = []

        for doc_id in range(start_idx, end_idx):
            lucene_doc = searcher.doc(doc_id).lucene_document()
            lucene_doc_raw_content = lucene_doc.get('raw')  # Extract raw text from the document
            if lucene_doc_raw_content:
                batch_texts.append(json.loads(lucene_doc_raw_content)['contents'])
                doc_ids.append(json.loads(lucene_doc_raw_content)['id'])

        # Compute embeddings for the batch of documents
        embeddings = compute_embeddings_batch(batch_texts)

        for doc_id, lucene_doc_raw_content, embedding in zip(doc_ids, batch_texts, embeddings):
            docs_with_embeddings.append({
                'id': doc_id, 
                'content': lucene_doc_raw_content,
                'embedding': embedding
            })
        
        # Save the documents with embeddings
        with open(saved_path, 'a+') as f:
            for line in docs_with_embeddings: 
                f.write(json.dumps(line) + "\n")
    

if __name__ == "__main__": 
    # Process all documents in batches and save results
    batch_size = 256
    saved_path=os.environ['saved_path']
    docs_with_embeddings = process_all_documents(batch_size, saved_path)


