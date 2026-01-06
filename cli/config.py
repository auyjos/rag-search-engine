"""Configuration settings for the RAG search engine."""
from pathlib import Path

# Directory paths
CACHE_DIR = Path("cache")
DATA_DIR = Path("data")

# File paths
MOVIES_FILE = DATA_DIR / "movies.json"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"
MOVIE_EMBEDDINGS_FILE = CACHE_DIR / "movie_embeddings.npy"
CHUNK_EMBEDDINGS_FILE = CACHE_DIR / "chunk_embeddings.npy"
CHUNK_METADATA_FILE = CACHE_DIR / "chunk_metadata.json"

# Model settings
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Chunking defaults
DEFAULT_CHUNK_SIZE = 4
DEFAULT_OVERLAP = 1

# Search defaults
DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 4
DESCRIPTION_PREVIEW_LENGTH = 100
