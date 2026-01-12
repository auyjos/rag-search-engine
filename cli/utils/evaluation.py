"""Evaluation metrics for search systems."""
import argparse
import json
import sys
from pathlib import Path

from cli.classes.hybrid_search import HybridSearch

# Add parent directory to path to import search functions
sys.path.insert(0, str(Path(__file__).parent.parent))
# Import from the classes directory




def load_golden_dataset():
    """Load movies from the JSON file."""
    with open("data/golden_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


def load_movies():
    """Load movies from the JSON file."""
    with open("data/movies.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["movies"]


def calculate_precision(retrieved_titles: list, relevant_titles: list) -> float:
    """
    Calculate precision: how many retrieved results are relevant.
    
    Args:
        retrieved_titles: List of titles returned by search
        relevant_titles: List of titles marked as relevant in golden dataset
        
    Returns:
        Precision score between 0 and 1
    """
    if not retrieved_titles:
        return 0.0
    
    relevant_retrieved = sum(1 for title in retrieved_titles if title in relevant_titles)
    precision = relevant_retrieved / len(retrieved_titles)

    return precision


def calculate_recall(retrieved_titles: list, relevant_titles: list) -> float:
    """
    Calculate recall: how many relevant results were retrieved.
    
    Args:
        retrieved_titles: List of titles returned by search
        relevant_titles: List of titles marked as relevant in golden dataset
        
    Returns:
        Recall score between 0 and 1
    """
    if not relevant_titles:
        return 0.0
    
    relevant_retrieved = sum(1 for title in retrieved_titles if title in relevant_titles)
    recall = relevant_retrieved / len(relevant_titles)
    
    return recall



def run_evaluation(limit:int):
    """
    Run evaluation on golden dataset and calculate precision@k.
    
    Args:
        limit: Number of results to retrieve (k value)
    """
    
    test_cases =  load_golden_dataset()
    documents = load_movies()
    search_engine = HybridSearch(documents)

    print(f"k={limit}\n")

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    for test_case in test_cases:
        query = test_case["query"]
        relevant_titles = test_case["relevant_docs"]
        
        # Run RRF search with k=60 and get top 'limit' results
        results = search_engine.rrf_search(query, k=60, limit=limit)
        
        # Extract titles from results (only the top 'limit' results)
        retrieved_titles = [result["document"]["title"] for result in results[:limit]]
        
        # Calculate precision and recall
        precision = calculate_precision(retrieved_titles, relevant_titles)
        recall = calculate_recall(retrieved_titles, relevant_titles)
        f1 = calculate_f1(precision, recall)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        
        # Print results
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}")
        print()
    
    # Print average precision and recall
    avg_precision = total_precision / len(test_cases)
    avg_recall = total_recall / len(test_cases)
    avg_f1 = total_f1 / len(test_cases)
    print(f"Average Precision@{limit}: {avg_precision:.4f}")
    print(f"Average Recall@{limit}: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

def calculate_f1(precision: float, recall:float) -> float: 
    """
    Calculate F1 score: harmonic mean of precision and recall.
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score between 0 and 1
    """
    if precision + recall ==0:
        return 0.0
    
    f1 = 2 * (precision*recall) / (precision + recall)
    return f1