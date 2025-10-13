import json
import os
import shutil
from pyserini.search.lucene import LuceneSearcher
import subprocess
from tqdm import tqdm
import random


def load_s2orc_documents_from_metadata_file(metadata_file):

    """Load S2ORC metadata from JSONL file"""
    print(f"Loading metadata from: {metadata_file}")
    
    documents = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading documents"):
            line = line.strip()
            if not line or 'corpusId' not in line:
                continue
            
            try:
                data = json.loads(line)
                
                # Check required fields
                if 'corpusId' not in data or not data.get('abstract'):
                    continue
                
                # Filter by abstract length (> 10 words)
                abstract = data['abstract']
                if len(abstract.split()) <= 10:
                    continue
                
                # Combine title and abstract
                title = data.get('title', '')
                content = f"{title}. {abstract}"
                
                corpus_id = str(data['corpusId'])
                documents.append((corpus_id, content))
                
            except Exception as ve:
                continue
    
    print(f"Completed Loaded {len(documents)} documents")
    return documents


class PyseriniLuceneBM25:
    def __init__(self, index_path="/scratch/academic_online/s2orc/pyserini_index"):
        """
        Initialize Pyserini 
        index_path: path to store Lucene index
        """
        self.index_path = index_path
        self.docs_path = "/scratch/academic_online/s2orc/documents.jsonl"
        self.all_indexed_ids_path = "/scratch/academic_online/s2orc/all_indexed_ids.txt"
        self.searcher = None
    
    def clean_content(self, content): 
        return " ".join(content.split())

    def build_and_save_index(self, documents):
        """
        Build BM25 index using Pyserini/Lucene and save to disk
        documents: list of tuples [(id, content), ...]
        """

        # Remove existing index and docs if they exist
        if os.path.exists(self.index_path):
            confirmation = input(f"continue to remove {self.index_path} (y/n): ")
            if confirmation.lower().strip() == "y":
                shutil.rmtree(self.index_path)
        
        if os.path.exists(self.docs_path):
            confirmation = input(f"continue to remove {self.docs_path} (y/n): ")
            if confirmation.lower().strip() == "y":
                os.remove(self.docs_path)
        

        # Prepare documents in JSONL format for Pyserini
        print(f"Preparing {len(documents)} documents...")
        all_indexed_ids = []
        with open(self.docs_path, 'w', encoding='utf-8') as f, open(self.all_indexed_ids_path, "w") as all_ids_f:
            for doc_id, content in documents:
                all_indexed_ids.append(str(doc_id))
                doc = {
                    "id": str(doc_id),
                    "contents": self.clean_content(content)
                }
                f.write(json.dumps(doc) + '\n')
            
            all_ids_f.write("\n".join(all_indexed_ids))
        
        print(f"Documents saved to: {self.docs_path} -- all ids: {self.all_indexed_ids_path}")
        
        print(f"\nBuilding Lucene index with BM25...")
        
        docs_dir = os.path.dirname(os.path.abspath(self.docs_path))
        
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", docs_dir,
            "--index", self.index_path,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"\nIndex built and saved to: {self.index_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error building index: {e}")
            raise
        
        # Verify
        self.get_index_stats()
    
    def load(self): 
        # Load the index and Set BM25 similarity
        self.searcher = LuceneSearcher(self.index_path)
        self.searcher.set_bm25(k1=0.9, b=0.4)
        self.all_indexed_ids = open(self.all_indexed_ids_path).readlines()

    def search_by_bm25(self, query_text, excluded_ids=[], top_k=5):
        """
        Load Lucene index and perform BM25 search
        query_text: search query string
        excluded_ids: avoid exact match with pos doc
        top_k: number of top results to return
        """
        assert self.searcher, "Load Searcher before query"

        # Clean query_text in the same way with build corpus
        query_text = self.clean_content(query_text)
        excluded_ids = [str(id) for id in excluded_ids]

        hits = self.searcher.search(query_text, k=top_k+len(excluded_ids))
           
        results = []
        
        for hit in hits:
            
            doc_id = hit.docid
            if str(doc_id) in excluded_ids: continue

            score = hit.score
            doc = self.searcher.doc(doc_id)
            content = json.loads(doc.raw())['contents']

            results.append({
                'id': doc_id,
                'content': content,
                'score': score
            })

            if len(results) >= top_k:
                break
            
        return results
    
    def random_search(self, excluded_ids=""): 
        """
        Random query context 
        excluded_id: avoid exact match with pos doc
        """
        excluded_ids = [str(id) for id in excluded_ids]
        for attempt in range(5): 
            random_id = random.choice(self.all_indexed_ids).strip()
            if str(random_id) not in excluded_ids: 
                return json.loads(self.searcher.doc(random_id).raw())['contents']

    def get_index_stats(self):
        """Get statistics about the Index"""
        
        searcher = LuceneSearcher(self.index_path)
        num_docs = searcher.num_docs
            
        print(f"Index path: {self.index_path} | Total documents: {num_docs}")


def main():
    # Load documents from S2ORC metadata
    metadata_file = "/scratch/academic_online/s2orc/metadata_from_api/metadata_from_api.3.jsonl"
    documents = load_s2orc_documents_from_metadata_file(metadata_file)[:2500000]

    # Create index instance
    indexer = PyseriniLuceneBM25(index_path="/scratch/academic_online/s2orc/pyserini_index")
    
    print("=" * 70)
    print("Building Pyserini/Lucene BM25 Index")
    print("=" * 70)
    
    # Build and save index
    indexer.build_and_save_index(documents)
  

if __name__ == "__main__":
    main()