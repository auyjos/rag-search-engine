import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cli.classes.hybrid_search import HybridSearch
from dotenv import load_dotenv
from google import genai
from helpers.constants import GEMINI_API_KEY_ERROR, api_key
from helpers.load_movies import load_movies
from utils.augmented_generation import (generate_citations_answer,
                                        generate_question_answer,
                                        generate_rag_response,
                                        generate_summary)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
        
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to summarize (default: 5)"
    )
        
    citations_parser = subparsers.add_parser(
        "citations", help="Answer query with citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to use (default: 5)"
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a question conversationally"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to use (default: 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            
            if not api_key:
                print("Error: GEMINI_API_KEY not found in environment variables")
                return
            
            # Load movies and perform RRF search
            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query, k=60, limit=5)
            
            # Print search results
            print("Search Results:")
            for result in results:
                title = result["document"]["title"]
                print(f"  - {title}")
            
            print()
            
            # Generate RAG response
            rag_response = generate_rag_response(query, results, api_key)
            
            print("RAG Response:")
            print(rag_response)
        case "summarize":
            query = args.query
            limit = args.limit
            
            
            if not api_key:
                print(GEMINI_API_KEY_ERROR)
                return
            
            # Load movies and perform RRF search
            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query, k=60, limit=limit)
            
            # Print search results
            print("Search Results:")
            for result in results:
                title = result["document"]["title"]
                print(f"  - {title}")
            
            print()
            
            # Generate summary
            summary = generate_summary(query, results, api_key)
            
            print("LLM Summary:")
            print(summary)
        case "citations":
            query = args.query
            limit = args.limit
            
            if not api_key:
                print(GEMINI_API_KEY_ERROR)
                return
            
            # Load movies and perform RRF search
            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query, k=60, limit=limit)
            
            # Print search results
            print("Search Results:")
            for result in results:
                title = result["document"]["title"]
                print(f"  - {title}")
            
            print()
            
            # Generate answer with citations
            answer = generate_citations_answer(query, results, api_key)
            
            print("LLM Answer:")
            print(answer)
        case "question":
            question = args.question
            limit = args.limit
            
        
            if not api_key:
                print("Error: GEMINI_API_KEY not found in environment variables")
                return
            
            # Load movies and perform RRF search
            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(question, k=60, limit=limit)
            
            # Print search results
            print("Search Results:")
            for result in results:
                title = result["document"]["title"]
                print(f"  - {title}")
            
            print()
            
            # Generate answer
            answer = generate_question_answer(question, results, api_key)
            
            print("Answer:")
            print(answer)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()