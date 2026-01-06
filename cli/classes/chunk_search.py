"""Chunk-level semantic search."""
from typing import Any

import numpy as np
from cli.classes.base_search import BaseSemanticSearch
from cli.config import (DEFAULT_CHUNK_SIZE, DEFAULT_MODEL, DEFAULT_OVERLAP,
                        DEFAULT_SEARCH_LIMIT, DESCRIPTION_PREVIEW_LENGTH,
                        SCORE_PRECISION)
from cli.utils.cache import load_chunk_embeddings, save_chunk_embeddings
from cli.utils.chunking import semantic_chunk
from cli.utils.similarity import cosine_similarity


class ChunkSemanticSearch(BaseSemanticSearch):
    """Semantic search at the chunk level with document aggregation."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """
        Initialize chunk-level semantic search.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(
        self,
        documents: list[dict[str, Any]],
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP
    ) -> np.ndarray:
        """
        Build embeddings for document chunks and save them to disk.
        
        Args:
            documents: List of dictionaries, each representing a movie
            max_chunk_size: Maximum number of sentences per chunk
            overlap: Number of sentences to overlap between chunks
        
        Returns:
            Numpy array of chunk embeddings
        """
        # Populate documents and document_map
        self._populate_document_map(documents)
        
        # Create lists to hold chunks and metadata
        all_chunks = []
        chunk_metadata = []
        
        # Process each document
        for movie_idx, doc in enumerate(documents):
            description = doc.get('description', '').strip()
            
            # Skip empty descriptions
            if not description:
                continue
            
            # Chunk the description
            chunks = semantic_chunk(description, max_chunk_size=max_chunk_size, overlap=overlap)
            
            # Add each chunk and its metadata
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'movie_idx': movie_idx,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(chunks)
                })
        
        # Generate embeddings for all chunks
        print("Generating embeddings for all chunks...")
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        
        # Save to cache
        save_chunk_embeddings(self.chunk_embeddings, chunk_metadata)
        
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(
        self,
        documents: list[dict[str, Any]],
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP
    ) -> np.ndarray:
        """
        Load chunk embeddings from cache or create them if they don't exist.
        
        Args:
            documents: List of dictionaries, each representing a movie
            max_chunk_size: Maximum number of sentences per chunk
            overlap: Number of sentences to overlap between chunks
        
        Returns:
            Numpy array of chunk embeddings
        """
        # Always populate documents and document_map
        self._populate_document_map(documents)
        
        # Try to load cached embeddings
        cached_embeddings, cached_metadata = load_chunk_embeddings()
        
        if cached_embeddings is not None and cached_metadata is not None:
            self.chunk_embeddings = cached_embeddings
            self.chunk_metadata = cached_metadata
            return self.chunk_embeddings
        
        # Build embeddings from scratch
        return self.build_chunk_embeddings(documents, max_chunk_size, overlap)
    
    def search_chunks(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[str, Any]]:
        """
        Search for documents by finding the most relevant chunks and aggregating scores.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        
        Returns:
            List of dictionaries containing id, title, document, score, and metadata
        
        Raises:
            ValueError: If chunk embeddings are not loaded
            
        Example:
            >>> search = ChunkSemanticSearch()
            >>> search.load_or_create_chunk_embeddings(documents)
            >>> results = search.search_chunks("space adventure", limit=5)
        """
        if self.chunk_embeddings is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        
        if self.chunk_metadata is None:
            raise ValueError("No chunk metadata loaded. Call `load_or_create_chunk_embeddings` first.")
        
        if self.documents is None:
            raise ValueError("No documents loaded. Call `load_or_create_chunk_embeddings` first.")
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Calculate cosine similarity for each chunk
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            metadata = self.chunk_metadata[i]
            chunk_scores.append({
                'chunk_idx': metadata['chunk_idx'],
                'movie_idx': metadata['movie_idx'],
                'score': score
            })
        
        # Aggregate scores at document level (keep max score per document)
        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score['movie_idx']
            score = chunk_score['score']
            
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score
        
        # Sort by score in descending order
        sorted_movie_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top results
        top_movie_scores = sorted_movie_scores[:limit]
        
        # Format results
        results = []
        for movie_idx, score in top_movie_scores:
            doc = self.documents[movie_idx]
            description = doc.get('description', '')
            
            # Truncate description to preview length
            truncated_desc = description[:DESCRIPTION_PREVIEW_LENGTH] if len(description) > DESCRIPTION_PREVIEW_LENGTH else description
            
            results.append({
                'id': doc['id'],
                'title': doc['title'],
                'document': truncated_desc,
                'score': round(score, SCORE_PRECISION),
                'metadata': doc.get('metadata', {})
            })
        
        return results
