"""
Legacy semantic search module for backward compatibility.

This module re-exports classes and functions from the refactored modules.
For new code, import from the specific modules instead:
- cli.classes.document_search for DocumentSemanticSearch
- cli.classes.chunk_search for ChunkSemanticSearch
- cli.utils.similarity for cosine_similarity
- cli.utils.chunking for semantic_chunk
"""

import json
from pathlib import Path

# Import from new modules
from cli.classes.chunk_search import \
    ChunkSemanticSearch as ChunkedSemanticSearch
from cli.classes.document_search import \
    DocumentSemanticSearch as SemanticSearch
from cli.config import MOVIES_FILE
from cli.utils.chunking import semantic_chunk
from cli.utils.similarity import cosine_similarity

# Re-export for backward compatibility
__all__ = [
    'SemanticSearch',
    'ChunkedSemanticSearch', 
    'cosine_similarity',
    'semantic_chunk',
    'embed_text',
    'embed_query_text',
    'verify_model',
    'verify_embeddings'
]


def embed_text(text: str) -> None:
    """Generate and display an embedding for the given text."""
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    """Generate and display an embedding for a query."""
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_model() -> None:
    """Verify that the model loads correctly and print its information."""
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def verify_embeddings() -> None:
    """Verify that embeddings are generated correctly for the movie dataset."""
    semantic_search = SemanticSearch()
    
    # Load movies from JSON file
    with open(MOVIES_FILE, 'r') as f:
        data = json.load(f)
    
    # Extract the movies array from the JSON structure
    documents = data['movies']
    
    # Load or create embeddings
    embeddings = semantic_search.load_or_create_embeddings(documents)
    
    # Print verification information
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
