import math
import os
import pickle
from collections import Counter

from helpers.constants import BM25_B, BM25_K1, CACHE_DIR
from helpers.tokenizer import tokenize


class InvertedIndex:
    def __init__(self, stopwords: set[str] | None = None):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.stopwords = stopwords or set()
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text, self.stopwords)
        # track doc length
        self.doc_lengths[doc_id] = len(tokens)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)


    def get_documents(self, term: str) -> list[int]:
        tokens = tokenize(term, self.stopwords)
        if not tokens:
            return []
        token = tokens[0]
        return sorted(self.index.get(token, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize(term, self.stopwords)
        if len(tokens) == 0:
            return 0
        if len(tokens) > 1:
            raise ValueError(f"Expected single token, got {len(tokens)}: {tokens}")
        
        token = tokens[0]
        if doc_id not in self.term_frequencies:
            return 0
        
        return self.term_frequencies[doc_id].get(token, 0)

    def build(self, movies: list[dict]) -> None:
        for i, m in enumerate(movies, 1):
            if i % 1000 == 0:
                print(f"Processed {i} movies...")
            doc_id = m["id"]
            self.docmap[doc_id] = m
            self.__add_document(doc_id, f"{m['title']} {m['description']}")

    def save(
        self,
        index_path: str = os.path.join(CACHE_DIR, "index.pkl"),
        docmap_path: str = os.path.join(CACHE_DIR, "docmap.pkl"),
        tf_path: str = os.path.join(CACHE_DIR, "term_frequencies.pkl"),
        doc_lengths_path: str = os.path.join(CACHE_DIR, "doc_lengths.pkl"),
    ) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    @classmethod
    def load(
        cls,
        stopwords: set[str] | None = None,
        index_path: str = os.path.join(CACHE_DIR, "index.pkl"),
        docmap_path: str = os.path.join(CACHE_DIR, "docmap.pkl"),
        tf_path: str = os.path.join(CACHE_DIR, "term_frequencies.pkl"),
        doc_lengths_path: str = os.path.join(CACHE_DIR, "doc_lengths.pkl"),
    ) -> "InvertedIndex":
        if not (os.path.exists(index_path) and os.path.exists(docmap_path) and os.path.exists(tf_path) and os.path.exists(doc_lengths_path)):
            raise FileNotFoundError("Index files not found. Please run 'build' command first.")
        
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            docmap = pickle.load(f)
        with open(tf_path, "rb") as f:
            term_frequencies = pickle.load(f)
        with open(doc_lengths_path, "rb") as f:
            doc_lengths = pickle.load(f)
        
        inst = cls(stopwords or set())
        inst.index = index
        inst.docmap = docmap
        inst.term_frequencies = term_frequencies
        inst.doc_lengths = doc_lengths
        return inst
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize(term, self.stopwords)
        if len(tokens) == 0:
            return 0.0
        if len(tokens) > 1:
            raise ValueError(f"Expected single token, got {len(tokens)}: {tokens}")
        
        token = tokens[0]
        total_docs = len(self.docmap)
        doc_matches = len(self.index.get(token, set()))
        idf = math.log((total_docs-doc_matches+0.5)/(doc_matches+0.5)+1)
        return idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        if tf == 0:
            return 0.0

        avg_len = self.__get_avg_doc_length()
        doc_len = self.doc_lengths.get(doc_id, 0)
        length_norm = 1.0 if avg_len == 0 else (1 - b) + b * (doc_len / avg_len)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        """Calculate BM25 score for a single term in a document."""
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[int, float]]:
        """Search documents using BM25 scoring."""
        # Tokenize the query
        query_tokens = tokenize(query, self.stopwords)
        
        if not query_tokens:
            return []
        
        # Initialize scores dictionary
        scores: dict[int, float] = {}
        
        # Calculate BM25 score for each document
        for doc_id in self.docmap.keys():
            total_score = 0.0
            for token in query_tokens:
                total_score += self.bm25(doc_id, token)
            
            if total_score > 0:
                scores[doc_id] = total_score
        
        # Sort by score (descending) and return top limit
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]