import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.classes.hybrid_search import HybridSearch


def load_movies():
    """Load movies from the JSON file."""
    with open("data/movies.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["movies"]


def enhance_query_spelling(query: str, api_key: str) -> str:
    """
    Enhance query by correcting spelling errors using Gemini API.
    
    Args:
        query: Original search query
        api_key: Gemini API key
        
    Returns:
        Enhanced query with corrected spelling
    """
    client = genai.Client(api_key=api_key)
    
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )
    
    # Extract and clean the enhanced query
    enhanced_query = response.text.strip()
    
    # Remove any quotation marks that might be in the response
    enhanced_query = enhanced_query.strip('"').strip("'")
    
    return enhanced_query

def enhance_query_rewrite(query: str, api_key: str) -> str:
    """
    Rewrite vague user queries into more specific, searchable terms using Gemini API.
    
    Args:
        query: Original vague search query
        api_key: Gemini API key
        
    Returns:
        Rewritten query optimized for search
    """
    client = genai.Client(api_key=api_key)
    
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )
    
    # Extract and clean the rewritten query
    rewritten_query = response.text.strip()
    
    # Remove any quotation marks that might be in the response
    rewritten_query = rewritten_query.strip('"').strip("'")
    
    return rewritten_query

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add normalize subcommand
    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores using min-max normalization")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="List of scores to normalize")

    # Add weighted-search subcommand
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 vs semantic (default: 0.5)")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results (default: 5)")

      # Add rrf-search subcommand
    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform hybrid search using Reciprocal Rank Fusion")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="RRF k parameter (default: 60)")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results (default: 5)")
    rrf_search_parser.add_argument(
            "--enhance",
            type=str,
            choices=["spell","rewrite"],
            help="Query enhancement method",
        )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            # Use the normalize method from HybridSearch class
            normalized = HybridSearch.normalize(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            # Load documents and initialize hybrid search
            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            
            # Perform weighted search
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            
            # Display results
            for i, result in enumerate(results[:args.limit], 1):
                doc = result["document"]
                print(f"{i}. {doc['title']}")
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"   BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                # Truncate description to first 100 characters
                description = doc.get('description', '')
                desc_preview = description[:100] + "..." if len(description) > 100 else description
                print(f"   {desc_preview}")
                if i < len(results[:args.limit]):
                    print()
        case "rrf-search":
            original_query = args.query
            query_to_search = original_query
            
            # Apply query enhancement if requested
            if args.enhance:
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                
                if not api_key:
                    print("Error: GEMINI_API_KEY not found in environment variables")
                    return
                
                # Choose enhancement method
                if args.enhance == "spell":
                    enhanced_query = enhance_query_spelling(original_query, api_key)
                elif args.enhance == "rewrite":
                    enhanced_query = enhance_query_rewrite(original_query, api_key)
                else:
                    enhanced_query = original_query
                
                # Only print enhancement message if query actually changed
                if enhanced_query != original_query:
                    print(f"Enhanced query ({args.enhance}): '{original_query}' -> '{enhanced_query}'\n")
                
                query_to_search = enhanced_query
            
            # Load documents and perform search
            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query_to_search, args.k, args.limit)

            # Display results
            for i, result in enumerate(results[:args.limit], 1):
                doc = result["document"]
                print(f"{i}. {doc['title']}")
                print(f"   RRF Score: {result['rrf_score']:.3f}")
                
                # Display ranks
                bm25_rank = result['bm25_rank'] if result['bm25_rank'] is not None else "-"
                semantic_rank = result['semantic_rank'] if result['semantic_rank'] is not None else "-"
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                
                # Truncate description to first 100 characters
                description = doc.get('description', '')
                desc_preview = description[:100] + "..." if len(description) > 100 else description
                print(f"   {desc_preview}")
                if i < len(results[:args.limit]):
                    print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()