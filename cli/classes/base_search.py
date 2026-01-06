"""Base semantic search functionality using sentence transformers."""
from typing import Any

import numpy as np
from cli.config import DEFAULT_MODEL
from sentence_transformers import SentenceTransformer


class BaseSemanticSearch:
    """Base class for semantic search operations."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """
        Initialize the semantic search with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"Loading model '{model_name}'... This may take a moment on first run.")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text input.
        
        Args:
            text: Input text string to embed
        
        Returns:
            Numpy array representing the text embedding
        
        Raises:
            ValueError: If text is empty or contains only whitespace
        
        Example:
            >>> search = BaseSemanticSearch()
            >>> embedding = search.generate_embedding("Hello world")
            >>> embedding.shape
            (384,)
        """
        if not text or text.strip() == "":
            raise ValueError("Input text cannot be empty or contain only whitespace")
        
        # encode expects a list and returns a list, we take the first element
        embedding = self.model.encode([text])[0]
        return embedding
    
    def _populate_document_map(self, documents: list[dict[str, Any]]) -> None:
        """
        Populate the internal document map for quick lookup.
        
        Args:
            documents: List of document dictionaries
        """
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
