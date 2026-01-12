import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cli.lib.multimodal_search import verify_image_embedding


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
    
    args = parser.parse_args()
    
    match args.command:
        case "verify_image_embedding":
            try:
                verify_image_embedding(args.image_path)
            except FileNotFoundError:
                print(f"Error: Image file not found: {args.image_path}")
            except Exception as e:
                print(f"Error: {e}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()