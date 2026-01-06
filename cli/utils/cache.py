"""Cache management utilities for embeddings."""
import json
from pathlib import Path
from typing import Any

import numpy as np
from cli.config import (CACHE_DIR, CHUNK_EMBEDDINGS_FILE, CHUNK_METADATA_FILE,
                        MOVIE_EMBEDDINGS_FILE)


def ensure_cache_dir() -> None:
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(exist_ok=True)


def save_embeddings(embeddings: np.ndarray, filepath: Path) -> None:
    """
    Save embeddings to disk.
    
    Args:
        embeddings: Numpy array of embeddings
        filepath: Path to save the embeddings
    """
    ensure_cache_dir()
    np.save(filepath, embeddings)
    print(f"Embeddings saved to {filepath}")


def load_embeddings(filepath: Path) -> np.ndarray | None:
    """
    Load embeddings from disk.
    
    Args:
        filepath: Path to the embeddings file
    
    Returns:
        Numpy array of embeddings, or None if file doesn't exist
    """
    if filepath.exists():
        print(f"Loading cached embeddings from {filepath}")
        return np.load(filepath)
    return None


def save_metadata(metadata: dict[str, Any], filepath: Path) -> None:
    """
    Save metadata to disk as JSON.
    
    Args:
        metadata: Dictionary containing metadata
        filepath: Path to save the metadata
    """
    ensure_cache_dir()
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {filepath}")


def load_metadata(filepath: Path) -> dict[str, Any] | None:
    """
    Load metadata from disk.
    
    Args:
        filepath: Path to the metadata file
    
    Returns:
        Dictionary containing metadata, or None if file doesn't exist
    """
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def save_movie_embeddings(embeddings: np.ndarray) -> None:
    """Save movie embeddings to the default cache location."""
    save_embeddings(embeddings, MOVIE_EMBEDDINGS_FILE)


def load_movie_embeddings() -> np.ndarray | None:
    """Load movie embeddings from the default cache location."""
    return load_embeddings(MOVIE_EMBEDDINGS_FILE)


def save_chunk_embeddings(embeddings: np.ndarray, metadata: list[dict]) -> None:
    """
    Save chunk embeddings and metadata to the default cache location.
    
    Args:
        embeddings: Numpy array of chunk embeddings
        metadata: List of metadata dictionaries for each chunk
    """
    save_embeddings(embeddings, CHUNK_EMBEDDINGS_FILE)
    metadata_dict = {
        "chunks": metadata,
        "total_chunks": len(metadata)
    }
    save_metadata(metadata_dict, CHUNK_METADATA_FILE)


def load_chunk_embeddings() -> tuple[np.ndarray | None, list[dict] | None]:
    """
    Load chunk embeddings and metadata from the default cache location.
    
    Returns:
        Tuple of (embeddings array, metadata list), or (None, None) if not found
    """
    embeddings = load_embeddings(CHUNK_EMBEDDINGS_FILE)
    metadata_dict = load_metadata(CHUNK_METADATA_FILE)
    
    if embeddings is not None and metadata_dict is not None:
        print("Cached chunk embeddings loaded successfully!")
        return embeddings, metadata_dict['chunks']
    
    return None, None
