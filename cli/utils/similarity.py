"""Similarity calculation functions."""
import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score between 0 and 1
    
    Example:
        >>> vec1 = np.array([1, 0, 0])
        >>> vec2 = np.array([1, 0, 0])
        >>> cosine_similarity(vec1, vec2)
        1.0
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)
