import json, os
from pyserini.search.lucene import LuceneSearcher
from collections import Counter 
import torch
from tqdm import tqdm 


def choose_and_embed_phrase(collection, docs_embeddings, num_docs_threshold, saved_path): 

    """ 
    Input:
        collection: A list of dictionaries, each containing the following keys:
            - 'id' (str)
            - 'contents' (str)
            - 'present_keyphrases' (List[str])
        num_docs_threshold: An integer representing the threshold for the number of document occurrences (int)
    Output:
        results: A dictionary where the keys are keyphrases (str) and the values are lists of document ids (List[str])
            Example: {'key_phrase_1': ['doc_id_1', 'doc_id_2', etc.], 'key_phrase_2': ['doc_id_3', 'doc_id_4', etc.]}"
    """

    #Select Phrase to run
    list_present_keyphrases = [kp for sublist in collection for kp in sublist["present_keyphrases"]]
    keyphrases_counter = Counter(list_present_keyphrases)
    keyphrase_to_run = [keyphrase for keyphrase, count in keyphrases_counter.items() if count >= num_docs_threshold]

    #Embed phrase by mean of documents
    results = []
    for keyphrase in tqdm(keyphrase_to_run): 
        doc_ids = [doc['id'] for doc in collection if keyphrase in doc['present_keyphrases']]
        results.append(
            {
                "keyphrase": keyphrase, 
                "doc_ids": doc_ids, 
                "embedding": torch.mean(torch.tensor([docs_embeddings[doc_id] for doc_id in doc_ids]), dim=0).tolist()
            }
        )
    
    json.dump(results, open(saved_path, 'w'), indent=4)

    return results


if __name__ == "__main__": 
    index_path = os.environ['index_path']
    docs_with_embeddings_path = os.environ['docs_with_embeddings_path']
    saved_path = os.environ['saved_path']
    num_docs_threshold = 50
    searcher = LuceneSearcher(index_path)


    print("Loading embeddings ...")
    embeddings = [json.loads(line) for line in open(docs_with_embeddings_path).readlines()]
    # Save for test
    # json.dump(embeddings[:10000], open("test.json", 'w'))
    # embeddings = json.load(open("test.json"))

    docs_embeddings = {item['id']: item['embedding'] for item in embeddings}
    
    collection= [json.loads(searcher.doc(doc_id).lucene_document().get("raw")) for doc_id in range(searcher.num_docs) if json.loads(searcher.doc(doc_id).lucene_document().get("raw"))['id'] in docs_embeddings]


    print("Phrase embeddings ...")
    choose_and_embed_phrase(collection, docs_embeddings, num_docs_threshold = num_docs_threshold, saved_path = saved_path)