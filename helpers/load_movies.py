import json


def load_movies():
    """Load movies from the JSON file."""
    with open("data/movies.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["movies"]
