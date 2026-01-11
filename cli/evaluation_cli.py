import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to import search functions
sys.path.insert(0, str(Path(__file__).parent.parent))
# Import from the classes directory

from cli.utils.evaluation import (calculate_precision, calculate_recall,
                                  run_evaluation)


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


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    run_evaluation(limit)

if __name__ == "__main__":
    main()