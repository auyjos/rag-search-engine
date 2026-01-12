import argparse
import mimetypes
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
from google import genai
from google.genai import types

BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "cache"
# Constants
GEMINI_API_KEY_ERROR = "Error: GEMINI_API_KEY not found in environment variables"
# Load environment variables
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

