"""Multimodal query rewriting utilities using Gemini API."""

import mimetypes

from google import genai
from google.genai import types


def rewrite_query_with_image(image_path: str, text_query: str, api_key: str) -> dict:
    """
    Rewrite a text query based on an image using Gemini's multimodal capabilities.
    
    Args:
        image_path: Path to the image file
        text_query: Original text query to rewrite
        api_key: Gemini API key
        
    Returns:
        Dictionary with 'rewritten_query' and 'token_count' keys
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: For other errors during processing
    """
    # Determine MIME type
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    
    # Read image file
    with open(image_path, "rb") as f:
        img = f.read()
    
    # Set up Gemini client
    client = genai.Client(api_key=api_key)
    
    # System prompt
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""
    
    # Build parts for the request
    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        text_query.strip(),
    ]
    
    # Send request to Gemini
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=parts
    )
    
    # Return results
    return {
        "rewritten_query": response.text.strip(),
        "token_count": response.usage_metadata.total_token_count if response.usage_metadata else None
    }