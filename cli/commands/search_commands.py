"""Search-related CLI commands."""
import argparse

from cli.classes.chunk_search import ChunkSemanticSearch
from cli.classes.document_search import DocumentSemanticSearch
from cli.config import DEFAULT_SEARCH_LIMIT, DESCRIPTION_PREVIEW_LENGTH


def add_search_commands(subparsers: argparse._SubParsersAction) -> None:
    """
    Add search-related commands to the CLI.
    
    Args:
        subparsers: Argument parser subparsers object
    """
    # Search command (document-level)
    search_parser = subparsers.add_parser(
        "search",
        help="Search for movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to return"
    )
    
    # Search chunked command
    search_chunked_parser = subparsers.add_parser(
        "search_chunked",
        help="Search for movies using chunked semantic search"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to return"
    )


def handle_search(args: argparse.Namespace, load_movies_func) -> None:
    """
    Handle the search command (document-level semantic search).
    
    Args:
        args: Parsed command-line arguments
        load_movies_func: Function to load movie documents
    """
    # Load movie documents
    documents = load_movies_func()
    
    # Initialize DocumentSemanticSearch
    semantic_search = DocumentSemanticSearch()
    
    # Load or create embeddings
    semantic_search.load_or_create_embeddings(documents)
    
    # Perform search
    results = semantic_search.search(args.query, args.limit)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        # Truncate description to first 100 characters if needed
        desc = result['description']
        if len(desc) > DESCRIPTION_PREVIEW_LENGTH:
            desc = desc[:DESCRIPTION_PREVIEW_LENGTH - 3] + "..."
        print(f"   {desc}")
        print()


def handle_search_chunked(args: argparse.Namespace, load_movies_func) -> None:
    """
    Handle the search_chunked command (chunk-level semantic search).
    
    Args:
        args: Parsed command-line arguments
        load_movies_func: Function to load movie documents
    """
    # Load movie documents
    documents = load_movies_func()
    
    # Initialize ChunkSemanticSearch
    chunked_search = ChunkSemanticSearch()
    
    # Load or create chunk embeddings
    chunked_search.load_or_create_chunk_embeddings(documents)
    
    # Perform search
    results = chunked_search.search_chunks(args.query, args.limit)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
