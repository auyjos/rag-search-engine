import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 =  np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0 
    
    return dot_product / (norm1*norm2)

class SemanticSearch:
    def __init__(self):
        """Initialize the semantic search with the all-MiniLM-L6-v2 model."""
        print("Loading model 'all-MiniLM-L6-v2'... This may take a moment on first run.")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text):
        """
        Generate an embedding for a single text input.
        
        Args:
            text: Input text string to embed
        
        Returns:
            numpy array representing the text embedding
        
        Raises:
            ValueError: If text is empty or contains only whitespace
        """
        if not text or text.strip() == "":
            raise ValueError("Input text cannot be empty or contain only whitespace")
        
        # encode expects a list and returns a list, we take the first element
        embedding = self.model.encode([text])[0]
        return embedding

    def build_embeddings(self, documents):
        """
        Build embeddings for all documents and save them to disk.
    
        Args:
            documents: List of dictionaries, each representing a movie
    
        Returns:
            numpy array of embeddings
        """
        self.documents = documents

        # Build document map
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        # Create string representations
        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        
        # Generate embeddings with progress bar
        print("Generating embeddings for all documents...")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        
        # Save embeddings to disk
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        embeddings_path = cache_dir / "movie_embeddings.npy"
        np.save(embeddings_path, self.embeddings)
        print(f"Embeddings saved to {embeddings_path}")
        
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        """
        Load embeddings from cache or create them if they don't exist.
        
        Args:
            documents: List of dictionaries, each representing a movie
        
        Returns:
            numpy array of embeddings
        """
        # Always populate documents and document_map
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        # Check if cached embeddings exist
        embeddings_path = Path("cache") / "movie_embeddings.npy"
        
        if embeddings_path.exists():
            print(f"Loading cached embeddings from {embeddings_path}")
            self.embeddings = np.load(embeddings_path)
            
            # Verify that cached embeddings match document count
            if len(self.embeddings) == len(documents):
                print("Cached embeddings loaded successfully!")
                return self.embeddings
            else:
                print(f"Warning: Cached embeddings count ({len(self.embeddings)}) doesn't match documents count ({len(documents)})")
                print("Rebuilding embeddings...")
        
        # Build embeddings from scratch
        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        """
        Search for documents semantically similar to the query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        
        Returns:
            List of dictionaries containing score, title, and description
        
        Raises:
            ValueError: If embeddings are not loaded
        """
        if self.embeddings is None: 
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        if self.documents is None:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding =  self.generate_embedding(query)
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
                'score': score,
                'title': doc['title'],
                'description': doc['description']
            })
        
        return results


def embed_text(text):
    """Generate and display an embedding for the given text."""
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query):
    """Generate and display an embedding for a query."""
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_model():
    """Verify that the model loads correctly and print its information."""
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def verify_embeddings():
    """Verify that embeddings are generated correctly for the movie dataset."""
    semantic_search = SemanticSearch()
    
    # Load movies from JSON file
    movies_path = Path("data/movies.json")
    with open(movies_path, 'r') as f:
        data = json.load(f)
    
    # Extract the movies array from the JSON structure
    documents = data['movies']
    
    # Load or create embeddings
    embeddings = semantic_search.load_or_create_embeddings(documents)
    
    # Print verification information
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")