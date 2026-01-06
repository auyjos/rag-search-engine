#!/usr/bin/env python3
"""
Semantic Search CLI - Main entry point for search and embedding commands.

This CLI provides commands for:
- Document-level semantic search
- Chunk-level semantic search
- Text chunking
- Embedding generation
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli.commands.embedding_commands import (add_embedding_commands,
                                             handle_embed_chunks,
                                             handle_semantic_chunk)
from cli.commands.search_commands import (add_search_commands, handle_search,
                                          handle_search_chunked)
from cli.config import MOVIES_FILE


def load_movies() -> list[dict]:
    """
    Load movies from the JSON file.
    
    Returns:
        List of movie dictionaries
    """
    with open(MOVIES_FILE, 'r') as f:
        data = json.load(f)
    return data['movies']


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic Search CLI for RAG Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for movies (document-level)
  python cli/semantic_search_cli.py search "space adventure" --limit 5
  
  # Search using chunks
  python cli/semantic_search_cli.py search_chunked "romantic comedy" --limit 5
  
  # Generate chunk embeddings
  python cli/semantic_search_cli.py embed_chunks
  
  # Chunk text
  python cli/semantic_search_cli.py semantic_chunk "First. Second. Third." --max-chunk-size 2
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add all commands
    add_embedding_commands(subparsers)
    add_search_commands(subparsers)
    
    args = parser.parse_args()
    
    # Route to appropriate handler
    match args.command:
        case "semantic_chunk":
            handle_semantic_chunk(args)
        
        case "embed_chunks":
            handle_embed_chunks(args, load_movies)
        
        case "search":
            handle_search(args, load_movies)
        
        case "search_chunked":
            handle_search_chunked(args, load_movies)
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()