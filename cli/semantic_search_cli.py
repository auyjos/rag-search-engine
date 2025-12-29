#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classes.semantic_search import (SemanticSearch, embed_query_text,
                                     embed_text, verify_embeddings,
                                     verify_model)


def search_movies(query, limit):
    """Search for movies using semantic search."""
    semantic_search = SemanticSearch()
    
    # Load movies from JSON file
    movies_path = Path("data/movies.json")
    with open(movies_path, 'r') as f:
        data = json.load(f)
    
    # Extract the movies array from the JSON structure
    documents = data['movies']
    
    # Load or create embeddings
    semantic_search.load_or_create_embeddings(documents)
    
    # Perform search
    results = semantic_search.search(query, limit)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        # Truncate description to first 100 characters if needed
        desc = result['description']
        if len(desc) > 100:
            desc = desc[:97] + "..."
        print(f"   {desc}")
        print()

def chunk_text(text, chunk_size, overlap=0):
    """
    Split text into fixed-size chunks based on word count with optional overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks (default: 0)
    """
    # Split text into words
    words = text.split()
    
    # Create chunks with overlap
    chunks = []
    i = 0
    while i < len(words):
        # Get chunk_size words starting at position i
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        # Move forward by (chunk_size - overlap) words
        # This creates overlap between chunks
        i += chunk_size - overlap
    
    # Print results
    total_chars = len(text)
    print(f"Chunking {total_chars} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the semantic search model is loaded correctly")
    
    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for the given text")
    embed_parser.add_argument("text", type=str, help="Text to embed")
    
    subparsers.add_parser("verify_embeddings", help="Verify embeddings for the movie dataset")
    
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate an embedding for a search query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results (default: 5)")

    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk (default: 200)")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of words to overlap between chunks (default: 0)")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_movies(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()