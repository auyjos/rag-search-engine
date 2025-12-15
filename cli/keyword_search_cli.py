#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.invert_index import InvertedIndex
from helpers.tokenizer import load_stopwords, tokenize


def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["movies"]

def search_movies(query: str) -> list[dict]:
    stopwords = load_stopwords()
    idx = InvertedIndex.load(stopwords)
    doc_ids: set[int] = set()
    for q_token in tokenize(query, stopwords):
        doc_ids.update(idx.get_documents(q_token))
    ordered_ids = sorted(doc_ids)[:5]
    return [idx.docmap[doc_id] for doc_id in ordered_ids]

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build and cache inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_movies(args.query)
            for i, movie in enumerate(results, 1):
                print(f"{i}. {movie['title']}")
        case "build":
            stopwords = load_stopwords()
            movies = load_movies()
            idx = InvertedIndex(stopwords)
            idx.build(movies)
            idx.save()
            docs = idx.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()