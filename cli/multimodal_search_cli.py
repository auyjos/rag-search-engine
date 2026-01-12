import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cli.lib.multimodal_search import (image_search_command,
                                       verify_image_embedding)


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
     # Add verify_image_embedding subcommand
    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Verify image embedding generation"
    )
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file"
    )
    
    # Add image_search subcommand
    search_parser = subparsers.add_parser(
        "image_search",
        help="Search for movies using an image"
    )
    search_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file"
    )
    
    args = parser.parse_args()
    
    match args.command:
        case "verify_image_embedding":
            try:
                verify_image_embedding(args.image_path)
            except FileNotFoundError:
                print(f"Error: Image file not found: {args.image_path}")
            except Exception as e:
                print(f"Error: {e}")
        case "image_search":
            try:
                results= image_search_command(args.image_path)
                # Print results
                for i, result in enumerate(results, 1):
                    title = result["title"]
                    similarity = result["similarity"]
                    description = result["description"]
                    
                    # Truncate description to first 100 characters
                    desc_preview = description[:100] + "..." if len(description) > 100 else description
                    
                    print(f"{i}. {title} (similarity: {similarity:.3f})")
                    print(f"   {desc_preview}")
                    if i < len(results):
                        print()
            except FileNotFoundError:
                print(f"Error: Image file not found: {args.image_path}")
            except Exception as e:
                print(f"Error: {e}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()