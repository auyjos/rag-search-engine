"""Text chunking utilities."""
import re


def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    """
    Split text into semantic chunks based on sentence boundaries.
    
    Args:
        text: The text to chunk
        max_chunk_size: Maximum number of sentences per chunk (default: 4)
        overlap: Number of sentences to overlap between chunks (default: 0)
    
    Returns:
        List of chunk strings
    
    Example:
        >>> text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        >>> chunks = semantic_chunk(text, max_chunk_size=2, overlap=0)
        >>> len(chunks)
        2
    """

    text = text.strip()

    if not text:
        return []


    # Split text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not re.search(r"[.!?]$", sentences[0].strip()):
        stripped_text = text.strip()
        return [stripped_text]  if stripped_text else []
    
    if not sentences:
        return []
    
    
    chunks = []
    i = 0
    
    while i < len(sentences):
        # Get up to max_chunk_size sentences
        chunk_sentences = sentences[i:i + max_chunk_size]
        chunk = " ".join(chunk_sentences)
        chunks.append(chunk)
        
        # Move forward by (max_chunk_size - overlap) sentences
        if overlap > 0 and i + max_chunk_size < len(sentences):
            i += max_chunk_size - overlap
        else:
            i += max_chunk_size
    
    return chunks
