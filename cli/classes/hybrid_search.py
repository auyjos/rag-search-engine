import os

from helpers.constants import CACHE_DIR
from helpers.tokenizer import load_stopwords

from .chunk_search import ChunkSemanticSearch
from .invert_index import InvertedIndex


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        # Load or build inverted index
        self.stopwords = load_stopwords()
        index_path = os.path.join(CACHE_DIR, "index.pkl")
        
        if os.path.exists(index_path):
            self.idx = InvertedIndex.load(self.stopwords)
        else:
            self.idx = InvertedIndex(self.stopwords)
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query, limit):
        """Perform BM25 search and return results in standardized format."""
        raw_results = self.idx.bm25_search(query, limit)
        
        # Convert from list[tuple[int, float]] to list[dict]
        results = []
        for doc_id, score in raw_results:
            results.append({
                "doc_id": doc_id,
                "score": score,
                "document": self.idx.docmap[doc_id]
            })
        return results

    @staticmethod
    def normalize(scores):
        """Normalize scores using min-max normalization to 0-1 range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if min_score == max_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]

    @staticmethod
    def hybrid_score(bm25_score, semantic_score, alpha=0.5):
        """Calculate hybrid score combining BM25 and semantic scores."""
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def weighted_search(self, query, alpha, limit=5):
        """Perform weighted hybrid search combining BM25 and semantic search."""
        # Get results from both searches (500x limit to ensure enough results)
        search_limit = limit * 500
        bm25_results = self._bm25_search(query, search_limit)
        semantic_results = self.semantic_search.search_chunks(query, search_limit)
        
        # Extract scores for normalization
        bm25_scores = [result["score"] for result in bm25_results]
        semantic_scores = [result["score"] for result in semantic_results]
        
        # Normalize scores
        normalized_bm25 = self.normalize(bm25_scores)
        normalized_semantic = self.normalize(semantic_scores)
        
        # Create mappings from doc_id to normalized scores
        bm25_score_map = {result["doc_id"]: normalized_bm25[i] for i, result in enumerate(bm25_results)}
        # semantic_results uses 'id' not 'doc_id'
        semantic_score_map = {result["id"]: normalized_semantic[i] for i, result in enumerate(semantic_results)}
        
        # Create a document lookup map for quick access
        doc_map = {doc["id"]: doc for doc in self.documents}
        
        # Combine results
        all_doc_ids = set(bm25_score_map.keys()) | set(semantic_score_map.keys())
        combined_results = {}
        
        for doc_id in all_doc_ids:
            bm25_norm = bm25_score_map.get(doc_id, 0.0)
            semantic_norm = semantic_score_map.get(doc_id, 0.0)
            hybrid = self.hybrid_score(bm25_norm, semantic_norm, alpha)
            
            # Get the full document
            doc = doc_map.get(doc_id)
            
            if doc is not None:
                combined_results[doc_id] = {
                    "doc_id": doc_id,
                    "document": doc,
                    "bm25_score": bm25_norm,
                    "semantic_score": semantic_norm,
                    "hybrid_score": hybrid
                }
        
        # Sort by hybrid score in descending order
        sorted_results = sorted(combined_results.values(), key=lambda x: x["hybrid_score"], reverse=True)
        
        return sorted_results
    
    @staticmethod
    def rrf_score(rank, k=60):
        """Calculate RRF score for a given rank"""
        return 1/ (k+rank)

    def rrf_search(self, query, k, limit=10):
        """Perform hybrid search using Reciprocal Rank Fusion."""
        search_limit = limit * 500
        bm25_result = self._bm25_search(query, search_limit)
        semantic_results = self.semantic_search.search_chunks(query, search_limit)

        bm25_rank_map = {result["doc_id"]: i +1 for i, result in enumerate(bm25_result)}
        semantic_rank_map = {result["id"]: i +1 for i, result in enumerate(semantic_results)}

        doc_map = {doc["id"]: doc for doc in self.documents}

        all_docs_ids = set(bm25_rank_map.keys()) | set(semantic_rank_map.keys())
        combined_results = {}

        for doc_id in all_docs_ids:
            bm25_rank = bm25_rank_map.get(doc_id)
            semantic_rank = semantic_rank_map.get(doc_id)

            rrf_score_total = 0.0
            if bm25_rank is not None:
                rrf_score_total += self.rrf_score(bm25_rank, k)
            if semantic_rank is not None:
                rrf_score_total += self.rrf_score(semantic_rank, k)

            doc = doc_map.get(doc_id)

            if doc is not None:
                combined_results[doc_id] = {
                    "doc_id": doc_id,
                    "document": doc,
                    "bm25_rank": bm25_rank,
                    "semantic_rank": semantic_rank,
                    "rrf_score": rrf_score_total
                }
        # Sort by RRF score in descending order
        sorted_results = sorted(combined_results.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_results
        
        