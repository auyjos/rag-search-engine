#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classes.invert_index import InvertedIndex
from helpers.constants import BM25_B, BM25_K1
from helpers.tokenizer import load_stopwords, tokenize


def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["movies"]

def search_movies(query: str) -> list[dict]:
    stopwords = load_stopwords()
    
    try:
        idx = InvertedIndex.load(stopwords)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    doc_ids: set[int] = set()
    for q_token in tokenize(query, stopwords):
        doc_ids.update(idx.get_documents(q_token))
    
    ordered_ids = sorted(doc_ids)[:5]
    return [idx.docmap[doc_id] for doc_id in ordered_ids]

def bm25_idf_command(term: str) -> float:
    stopwords = load_stopwords()
    idx = InvertedIndex.load(stopwords)
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    stopwords = load_stopwords()
    idx = InvertedIndex.load(stopwords)
    return idx.get_bm25_tf(doc_id, term, k1, b)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    
    subparsers.add_parser("build", help="Build and cache inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to compute IDF for")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to compute TF-IDF for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")


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

            print("Index built and saved successfully.")

        case "tf":
            stopwords = load_stopwords()
            try:
                idx = InvertedIndex.load(stopwords)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
            
            tf = idx.get_tf(args.doc_id, args.term)
            print(tf)
        case "idf":
            stopwords = load_stopwords()
            try:
                idx = InvertedIndex.load(stopwords)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            total_docs = len(idx.docmap)
            doc_matches = len(idx.get_documents(args.term))
            idf = math.log((total_docs + 1) / (doc_matches + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            stopwords = load_stopwords()
            try:
                idx = InvertedIndex.load(stopwords)
            except FileNotFoundError as e:
                print(f"Error {e}")
                sys.exit(1)
            
            tf = idx.get_tf(args.doc_id, args.term)
            total_docs = len(idx.docmap)
            doc_matches = len(idx.get_documents(args.term))
            idf = math.log((total_docs + 1) / (doc_matches + 1))
            tfidf = tf * idf
            print(f"TF-IDF of '{args.term}' in document {args.doc_id}: {tfidf:.2f}")
        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        
        case "bm25search":
            stopwords = load_stopwords()
            try:
                idx = InvertedIndex.load(stopwords)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
            
            results = idx.bm25_search(args.query, args.limit)
            
            for i, (doc_id, score) in enumerate(results, 1):
                movie = idx.docmap[doc_id]
                print(f"{i}. ({doc_id}) {movie['title']} - Score: {score:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()