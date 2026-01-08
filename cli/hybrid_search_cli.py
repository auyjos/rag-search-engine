import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.classes.hybrid_search import HybridSearch


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
            documents= load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(args.query, args.k, args.limit)

            for i, result in enumerate(results[:args.limit],1):
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