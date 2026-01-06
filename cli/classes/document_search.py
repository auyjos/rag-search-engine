"""Document-level semantic search."""
from typing import Any

import numpy as np
from cli.classes.base_search import BaseSemanticSearch
from cli.config import DEFAULT_MODEL, DEFAULT_SEARCH_LIMIT, SCORE_PRECISION
from cli.utils.cache import load_movie_embeddings, save_movie_embeddings
from cli.utils.similarity import cosine_similarity


class DocumentSemanticSearch(BaseSemanticSearch):
    """Semantic search at the document level."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """
        Initialize document-level semantic search.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        super().__init__(model_name)
    
    def build_embeddings(self, documents: list[dict[str, Any]]) -> np.ndarray:
        """
        Build embeddings for all documents and save them to disk.
        
        Args:
            documents: List of dictionaries, each representing a movie
        
        Returns:
            Numpy array of embeddings
        """
        self._populate_document_map(documents)
        
        # Create string representations
        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        
        # Generate embeddings with progress bar
        print("Generating embeddings for all documents...")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        
        # Save embeddings to disk
        save_movie_embeddings(self.embeddings)
        
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict[str, Any]]) -> np.ndarray:
        """
        Load embeddings from cache or create them if they don't exist.
        
        Args:
            documents: List of dictionaries, each representing a movie
        
        Returns:
            Numpy array of embeddings
        """
        # Always populate documents and document_map
        self._populate_document_map(documents)
        
        # Try to load cached embeddings
        cached_embeddings = load_movie_embeddings()
        
        if cached_embeddings is not None:
            # Verify that cached embeddings match document count
            if len(cached_embeddings) == len(documents):
                self.embeddings = cached_embeddings
                return self.embeddings
            else:
                print(f"Warning: Cached embeddings count ({len(cached_embeddings)}) doesn't match documents count ({len(documents)})")
                print("Rebuilding embeddings...")
        
        # Build embeddings from scratch
        return self.build_embeddings(documents)
    
    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[str, Any]]:
        """
        Search for documents semantically similar to the query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        
        Returns:
            List of dictionaries containing score, title, and description
        
        Raises:
            ValueError: If embeddings are not loaded
            
        Example:
            >>> search = DocumentSemanticSearch()
            >>> search.load_or_create_embeddings(documents)
            >>> results = search.search("action movie", limit=5)
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        if self.documents is None:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        similarities = []
        
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            document = self.documents[i]
            similarities.append((score, document))
        
        # Sort by score in descending order and limit results
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:limit]
        
        # Format results
        results = []
        for score, doc in top_results:
            results.append({
                'score': round(score, SCORE_PRECISION),
                'title': doc['title'],
                'description': doc['description']
            })
        
        return results
