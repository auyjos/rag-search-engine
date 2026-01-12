import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cli.lib.multimodal_search import verify_image_embedding
from cli.utils.image_description import rewrite_query_with_image
from dotenv import load_dotenv
from helpers.constants import GEMINI_API_KEY_ERROR, api_key


def main():
    parser = argparse.ArgumentParser(description="Multimodal Query Rewriting CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--query", type=str, required=True, help="Text query to rewrite")
    
    args = parser.parse_args()

    
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables")
        return
    
    try:
        # Rewrite query using image
        result = rewrite_query_with_image(args.image, args.query, api_key)
        
        # Print results
        print(f"Rewritten query: {result['rewritten_query']}")
        if result['token_count'] is not None:
            print(f"Total tokens:    {result['token_count']}")
    except FileNotFoundError:
        print(f"Error: Image file not found: {args.image}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()