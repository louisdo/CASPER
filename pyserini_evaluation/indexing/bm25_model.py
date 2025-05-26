import math, pickle
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from nltk.corpus import stopwords

class BM25:
    def __init__(
            self, 
            corpus, 
            k1=0.9, 
            b=0.4,
            splade_model_dir = None):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.nd = {}
        self.corpus_length = len(corpus)

        self.tokenizer = AutoTokenizer.from_pretrained(splade_model_dir)

        self.STOPWORDS = set(list(stopwords.words('english')))
        
        self._initialize()


    def tokenize(self, text):
        res = self.tokenizer.tokenize(text)
        return [tok for tok in res if tok not in self.STOPWORDS]


    def save_model(self, filename):
        """Serialize BM25 model state without the original corpus"""
        state = {
            'k1': self.k1,
            'b': self.b,
            'doc_lengths': self.doc_lengths,
            'avgdl': self.avgdl,
            'nd': self.nd,
            'idf': self.idf,
            'corpus_length': self.corpus_length,
            'splade_model_dir': self.tokenizer.name_or_path  # Save tokenizer path
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load_model(cls, filename):
        """Deserialize BM25 model from file"""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        # Create dummy instance and populate state
        instance = cls.__new__(cls)
        instance.__dict__.update(state)
        
        # Reinitialize tokenizer from saved path
        instance.tokenizer = AutoTokenizer.from_pretrained(state['splade_model_dir'])

        instance.STOPWORDS = set(list(stopwords.words('english')))
        
        return instance

    def _initialize(self):
        """Calculate document frequencies and lengths"""
        nd = {}  # Number of docs containing each term
        num_docs = len(self.corpus)
        
        for doc_ in tqdm(self.corpus, desc = "Initializing BM25 model"):
            doc = self.tokenize(doc_)
            self.doc_lengths.append(len(doc))
            frequencies = {}
            
            for term in doc:
                frequencies[term] = frequencies.get(term, 0) + 1
            
            self.doc_freqs.append(frequencies)
            
            # Update document frequency counts
            for term in set(doc):
                nd[term] = nd.get(term, 0) + 1

        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        self.nd = nd
        self._calc_idf(num_docs)

        del self.corpus



    def _calc_idf(self, num_docs):
        """Calculate inverse document frequencies"""
        for term, freq in self.nd.items():
            self.idf[term] = math.log(
                (num_docs - freq + 0.5) / (freq + 0.5) + 1
            )

    def get_term_scores(self, text_string):
        """Calculate BM25 scores for each term in input text against corpus"""
        scores = {}

        text = self.tokenize(text_string)

        doc_len = len(text)

        for term in set(text):
            if term not in self.idf:
                continue
                
            tf = text.count(term)
            idf = self.idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avgdl
            )
            
            scores[term] = int(100 * idf * numerator / denominator )
    
        scores = {k.replace(" ", "-"): v for k,v in scores.items()}

        return scores




if __name__ == "__main__":
    corpus = [
        "hello there good man",
        "it is windy in london",
        "how is the weather today"
    ]

    # Initialize BM25
    bm25 = BM25(corpus, splade_model_name = "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_55/debug/checkpoint/model")

    # Process query
    query = "windy london"

    # Get scores
    scores = bm25.get_term_scores(query)
    print(f"Scores: {scores}")