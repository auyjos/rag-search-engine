"""Embedding-related CLI commands."""
import argparse

from cli.classes.chunk_search import ChunkSemanticSearch
from cli.classes.document_search import DocumentSemanticSearch
from cli.utils.chunking import semantic_chunk


def add_embedding_commands(subparsers: argparse._SubParsersAction) -> None:
    """
    Add embedding-related commands to the CLI.
    
    Args:
        subparsers: Argument parser subparsers object
    """
    # Semantic chunk command
    chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Split text into semantic chunks"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences to overlap between chunks"
    )
    
    # Embed chunks command
    subparsers.add_parser("embed_chunks", help="Generate embeddings for document chunks")


def handle_semantic_chunk(args: argparse.Namespace) -> None:
    """
    Handle the semantic_chunk command.
    
    Args:
        args: Parsed command-line arguments
    """
    chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
    
    print(f"Semantically chunking {len(args.text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def handle_embed_chunks(args: argparse.Namespace, load_movies_func) -> None:
    """
    Handle the embed_chunks command.
    
    Args:
        args: Parsed command-line arguments
        load_movies_func: Function to load movie documents
    """
    # Load movie documents
    documents = load_movies_func()
    
    # Initialize ChunkSemanticSearch
    chunked_search = ChunkSemanticSearch()
    
    # Load or build chunk embeddings
    embeddings = chunked_search.load_or_create_chunk_embeddings(documents)
    
    # Print info
    print(f"Generated {len(embeddings)} chunked embeddings")
