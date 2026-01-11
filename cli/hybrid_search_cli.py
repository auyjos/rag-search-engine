import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cli.classes.hybrid_search import HybridSearch
from cli.utils.gemini_functions import (enhance_query_expand,
                                        enhance_query_rewrite,
                                        enhance_query_spelling, rerank_batch,
                                        rerank_cross_encoder,
                                        rerank_individual)
from dotenv import load_dotenv
from google import genai

# Add parent directory to path to allow imports



def load_movies():
    """Load movies from the JSON file."""
    with open("data/movies.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["movies"]



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
            choices=["spell","rewrite", "expand"],
            help="Query enhancement method",
        )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-ranking method to use after initial search",
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
                elif args.enhance == "expand":
                    enhanced_query = enhance_query_expand(original_query, api_key)
                else:
                    enhanced_query = original_query
                
                # Only print enhancement message if query actually changed
                if enhanced_query != original_query:
                    print(f"Enhanced query ({args.enhance}): '{original_query}' -> '{enhanced_query}'\n")
                
                query_to_search = enhanced_query
            
            # Determine the search limit (5x if reranking)
            search_limit = args.limit * 5 if args.rerank_method else args.limit
            
            # Load documents and perform search
            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query_to_search, args.k, search_limit)
            
            # Apply re-ranking if requested
            if args.rerank_method == "individual":
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                
                if not api_key:
                    print("Error: GEMINI_API_KEY not found in environment variables")
                    return
                
                # Only rerank the top 'limit' results, not all search results
                top_results = results[:args.limit]
                print(f"Reranking top {len(top_results)} results using {args.rerank_method} method...")
                results = rerank_individual(query_to_search, top_results, api_key)
            elif args.rerank_method == "batch":
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                
                if not api_key:
                    print("Error: GEMINI_API_KEY not found in environment variables")
                    return
                
                # Rerank using batch method (single API call)
                top_results = results[:search_limit]
                print(f"Reranking top {args.limit} results using {args.rerank_method} method...")
                results = rerank_batch(query_to_search, top_results, api_key)
            elif args.rerank_method == "cross_encoder":
                # Rerank using cross-encoder method (no API key needed)
                top_results = results[:search_limit]
                print(f"Reranking top {search_limit} results using {args.rerank_method} method...")
                results = rerank_cross_encoder(query_to_search, top_results)
            
            # Print header
            print(f"Reciprocal Rank Fusion Results for '{query_to_search}' (k={args.k}):\n")

            # Display results (truncated to limit)
            for i, result in enumerate(results[:args.limit], 1):
                doc = result["document"]
                print(f"{i}. {doc['title']}")
                
                # Show cross encoder score if available
                if "cross_encoder_score" in result:
                    print(f"   Cross Encoder Score: {result['cross_encoder_score']:.3f}")
                
                # Show rerank rank if available (batch method)
                if "rerank_rank" in result:
                    print(f"   Rerank Rank: {result['rerank_rank']}")
                
                # Show rerank score if available (individual method)
                if "rerank_score" in result:
                    print(f"   Rerank Score: {result['rerank_score']:.3f}/10")
                
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