# RAG Search Engine

A comprehensive information retrieval system implementing multiple search methodologies including keyword-based search (BM25), semantic search, hybrid search, and Retrieval-Augmented Generation (RAG) with multimodal capabilities. Built as a learning project to understand the fundamentals of modern search and RAG systems.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Keyword Search](#keyword-search)
  - [Semantic Search](#semantic-search)
  - [Hybrid Search](#hybrid-search)
  - [Search Evaluation](#search-evaluation)
  - [RAG Pipeline](#rag-pipeline)
  - [Multimodal Search](#multimodal-search)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

## 🎯 Overview

This project demonstrates the core concepts behind modern search systems and Retrieval-Augmented Generation (RAG) by implementing:

- **Keyword Search**: Traditional BM25 (Best Matching 25) algorithm for exact and fuzzy keyword matching
- **Semantic Search**: Neural embedding-based search that understands meaning and context
- **Hybrid Search**: Combines keyword and semantic approaches with configurable weighting (RRF)
- **RAG Pipeline**: Context-aware answer generation with citations and summarization
- **Multimodal Search**: Image and text-based search using CLIP embeddings
- **Search Evaluation**: Automated evaluation with Precision@k, Recall@k, F1 Score, and LLM-as-a-judge

The system is built around a dataset of ~5,000 movies, providing a rich corpus for search and RAG experimentation.

## ✨ Features

### Search Methods

1. **BM25 Keyword Search**
   - Inverted index construction
   - TF-IDF scoring with length normalization
   - Stopword filtering
   - Configurable k1 and b parameters

2. **Semantic Search**
   - Document-level embeddings using Sentence Transformers
   - Chunk-level search for long documents
   - Cosine similarity scoring
   - Sentence-aware chunking with overlap

3. **Reciprocal Rank Fusion (RRF) for combining rankings
   - Query enhancement (spelling correction, rewriting, expansion)
   - Re-ranking methods (LLM-based, batch, cross-encoder)
   - Best of both worlds: handles exact matches and conceptual queries

4. **Search Evaluation**
   - Precision@k and Recall@k metrics
   - F1 Score calculation
   - LLM-as-a-judge evaluation (0-3 relevance scoring)
   - Golden dataset testing

5. **RAG Pipeline**
   - Context-aware answer generation
   - Multi-document summarization
   - Citation-aware responses with source references
   - Conversational question answering
   - Integration with Google Gemini API

6. **Multimodal Search**
- Google Gemini API key (for RAG features)

### Setup

```bash
# Clone the repository
git clone https://github.com/auyjos/rag-search-engine.git
cd rag-search-engine

# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

The first time you run semantic search, it will download the sentence-transformer model (~80MB).
The first time you run multimodal search, it will download the CLIP model (~60
## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/auyjos/rag-search-engine.git
cd rag-search-engine

# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt
```

The first time you run semantic search, it will download the sentence-transformer model (~80MB).

## 💻 Usage

### Keyword Search

Search using BM25 algorithm for exact keyword matching:

```bash
# Build the inverted index (first time only)
uv run cli/keyword_search_cli.py build

# Search for movies
uv run cli/keyword_search_cli.py search "space adventure"

# Search with limit
uv run cli/keyword_search_cli.py search "detective mystery" --limit 10

# Get BM25 IDF score for a term
uv run cli/keyword_search_cli.py idf "detective"

# Get BM25 TF score for a document and term
uv run cli/keyword_search_cli.py tf 123 "detective"

# Run BM25 search with custom parameters
uv run cli/keyword_search_cli.py bm25 "space adventure" --limit 5
```

### Semantic Search

Search by meaning using neural embeddings:

```bash
# Generate embeddings (first time or when data changes)
uv run cli/semantic_search_cli.py embed-chunks

# Document-level semantic search
uv run cli/semantic_search_cli.py search "movies about family relationships"

# Chunk-level semantic search (better for long documents)
uv run cli/semantic_search_cli.py search-chunked "robot falls in love" --limit 10

# Create semantic chunks from text
uv run cli/semantic_search_cli.py chunk "Long text..." --max-chunk-size 4 --overlap 1
```

### Hybrid Search

Combine keyword and semantic approaches:

```bash
# Normalize scores (utility function)
uv run cli/hybrid_search_cli.py normalize 0.5 2.3 1.2 0.5 0.1

# Weighted hybrid search with default alpha (0.5)
uv run cli/hybrid_search_cli.py weighted-search "British detective"

# Emphasize keywords (alpha=0.8 means 80% keyword, 20% semantic)
uv run cli/hybrid_search_cli.py weighted-search "The Revenant" --alpha 0.8 --limit 10

# Reciprocal Rank Fusion (RRF) search
uv run cli/hybrid_search_cli.py rrf-search "bear attack" -k 60 --limit 5

# RRF with query enhancement
uv run cli/hybrid_search_cli.py rrf-search "beear movie" --enhance spell
uv run cli/hybrid_search_cli.py rrf-search "that bear movie" --enhance rewrite
uv run cli/hybrid_search_cli.py rrf-search "grizzly film" --enhance expand

# RRF with re-ranking
uv run cli/hybrid_search_cli.py rrf-search "intense thriller" --rerank-method cross_encoder
uv run cli/hybrid_search_cli.py rrf-search "family comedy" --rerank-method batch
uv run cli/hybrid_search_cli.py rrf-search "action movie" --rerank-method individual

# RRF with evaluation
uv run cli/hybrid_search_cli.py rrf-search "space adventure" --evaluate

# Emphasize semantics (alpha=0.2 means 20% keyword, 80% semantic)
uv run cli/hybrid_search_cli.py weighted-search "family movies" --alpha 0.2 --limit 10

# # Search Evaluation

Evaluate search quality using various metrics:

```bash
# Run evaluation with golden dataset (default k=5)
uv run cli/evaluation_cli.py --limit 5

# Output includes:
# - Precision@k: How many retrieved results are relevant
# - Recall@k: How many relevant results were retrieved
# - F1 Score: Harmonic mean of precision and recall
```

### RAG Pipeline

Generate context-aware answers using retrieved documents:

```bash
# Basic RAG: Search + Generate answer
uv run cli/augmented_generation_cli.py rag "what are good dinosaur movies"

# Multi-document summarization
uv run cli/augmented_generation_cli.py summarize "bear movies" --limit 5

# Answer with citations [1], [2], etc.
uv run cli/augmented_generation_cli.py citations "intense thriller movies" --limit 5

# Conversational question answering
uv run cli/augmented_generation_cli.py question "What year was The Revenant released?"
```

### Multimodal Search

Search using images and text:

```bash
# Verify image embedding generation
uv runRF**: Alternative fusion using Reciprocal Rank Fusion: `score = Σ 1/(k + rank)`
5. **Ranking**: Returns documents sorted by hybrid score

### RAG Pipeline

1. **Retrieval**: Uses hybrid search (RRF) to find relevant documents
2. **Context Building**: Formats d    # Command-line interfaces
│   ├── keyword_search_cli.py         # BM25 search CLI
│   ├── semantic_search_cli.py        # Semantic search CLI
│   ├── hybrid_search_cli.py          # Hybrid search CLI (RRF, reranking)
│   ├── evaluation_cli.py             # Search evaluation CLI
│   ├── augmented_generation_cli.py   # RAG pipeline CLI
│   ├── describe_image_cli.py         # Multimodal query rewriting CLI
│   ├── multimodal_search_cli.py      # Image-based search CLI
│   ├── classes/                      # Core search implementations
│   │   ├── base_search.py            # Base semantic search class
│   │   ├── chunk_search.py           # Chunk-level semantic search
│   │   ├── document_search.py        # Document-level semantic search
│   │   ├── hybrid_search.py          # Hybrid search (weighted + RRF)
│   │   ├── invert_index.py           # Inverted index & BM25
│   │   └── semantic_search.py        # Legacy compatibility layer
│   ├── lib/                          # Library modules
│   │   └── multimodal_search.py      # CLIP-based multimodal search
│   ├── commands/                     # Command handlers
│   │   ├── embedding_commands.py     # Embedding generation commands
│   │   └── search_commands.py        # Search commands
│   ├── utils/                        # Utility functions
│   │   ├── cache.py                  # Caching for embeddings
│   │   ├── chunking.py               # Text chunking utilities
│   │   ├── similarity.py             # Similarity calculations
│   │   ├── evaluation.py             # Evaluation metrics
│   │   ├── gemini_functions.py       # Gemini API functions
│   │   ├── augmented_generation.py   # RAG helper functions
│   │   └── image_description.py      # Multimodal query rewriting
│   └── config.py                     # Configuration settings
├── data/                             # Data files
│   ├── movies.json                   # Movie dataset (~5,000 movies)
│   ├── golden_dataset.json           # Test cases for evaluation
│   ├── stopwords.txt                 # Stopwords for keyword search
│   └── paddington.jpeg               # Sample image for multimodal search
├── helpers/                          # Helper modules
│   ├── constants.py                  # BM25 and API constants
│   ├── tokenizer.py                  # Text tokenization
│   └── load_movies.py                # Movie dataset loader
├── cache/                            # Generated cache files
│   ├── chunk_embeddings.npy          # Cached chunk embeddings
│   ├── chunk_metadata.json           # Chunk metadata
│   ├── index.pkl                     # Inverted index
│   └── ...                           # Other cache files
├── .env                              # Environment variables (API keys)
└── pyproject.toml    d, 20% Semantic | Specific terms with some context | "Leonardo DiCaprio survival" |
| 0.5 | 50/50 Split | Balanced queries | "2015 adventure films" |
| 0.2 | 20% Keyword, 80% Semantic | Conceptual searches | "movies about redemption" |
| 0.0 | 100% Semantic | Abstract concepts | "finding yourself" |

## 🔧 How It Works

### BM25 Keyword Search

1. **Indexing**: Builds an inverted index mapping terms to documents
2. **Scoring**: Uses BM25 formula: `score = IDF(term) × TF(term, doc)`
   - **IDF**: Inverse Document Frequency - rarer terms score higher
   - **TF**: Term Frequency with length normalization
3. **Ranking**: Returns top documents by BM25 score

### Semantic Search

1. **Embedding Generation**: Converts text to 384-dimensional vectors using `all-MiniLM-L6-v2`
2. **Chunking**: Splits long documents into overlapping chunks at sentence boundaries
3. **Similarity**: Computes cosine similarity between query and document embeddings
4. **Aggregation**: For chunk-based search, keeps the maximum score per document

### Hybrid Search

1. **Parallel Search**: Runs both BM25 and semantic search simultaneously
2. **Normalization**: Applies min-max normalization to make scores comparable
3. **Combination**: Calculates weighted score: `α × BM25_norm + (1-α) × semantic_norm`
4. **Ranking**: Returns documents sorted by hybrid score

## 📁 Project Structure

```
rag-search-engine/
├── cli/                          # Command-line interfaces
│   ├── keyword_search_cli.py     # BM25 search CLI
│   ├── semantic_search_cli.py    # Semantic search CLI
│   ├── hybrid_search_cli.py      # Hybrid search CLI
│   ├── classes/                  # Core search implementations
│   │   ├── base_search.py        # Base semantic search class
│   │   ├── chunk_search.py       # Chunk-level semantic search
│   │   ├── document_search.py    # Document-level semantic search
│   │  s
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
MULTIMODAL_MODEL = "clip-ViT-B-32"  # CLIP model for images

# Chunking
DEFAULT_CHUNK_SIZE = 4              # Sentences per chunk
DEFAULT_OVERLAP = 1                 # Overlapping sentences

# Search
DEFAULT_SEARCH_LIMIT = 5            # Default number of results
SCORE_PRECISION = 4                 # Decimal places for scores

# BM25 (in helpers/constants.py)
BM25_K1 = 1.5                       # Term saturation parameter
BM25_B = 0.75                       # Length normalization parameter

# RRF
RRF_K = 60                          # Default k parameter for RRF

# API
GEMINI_MODEL = "gemini-2.0-flash-001"  # Gemini model for RAG
MAX_RETRIES = 3                     # API retry attempts
RETRY_DELAY = 30                    # Initial retry delay (seconds)
```

### Environment Variables

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
├── helpers/                      # Helper modules
│   ├── constants.py              # BM25 constants
│   └── tokenizer.py              # Text tokenization
├── cache/                        # Generated cache files
│   ├── chunk_embeddings.npy      # Cached chunk embeddings
│   ├── chunk_metadata.json       # Chunk metadata
│   ├── index.pkl                 # Inverted index
│   └── ...                       # Other cache files, RRF
- **RAG Implementation**: Context-aware generation, prompt engineering, citations
- **Multimodal AI**: Cross-modal embeddings, image-text similarity (CLIP)
- **Search Evaluation**: Precision/Recall/F1, LLM-as-a-judge, golden datasets
- **API Integration**: Gemini API, retry logic, rate limiting, error handling
- **Query Enhancement**: Spelling correction, query rewriting, expansion
- **Re-ranking**: LLM-based, cross-encoder, batch processing
```

## ⚙️ Configuration

Key configuration values in [`cli/config.py`](cli/config.py):

```python
# Model
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model

# Chunking
DEFAULT_CHUNK_SIZE = 4              # Sentences per chunk
DEFAULT_OVERLAP = 1                 # Overlapping sentences

# Search
DEFAULT_SEARCH_LIMIT = 5            # Default number of results
SCORE_PRECISION = 4                 # Decimal places for scores

# BM25 (in helpers/constants.py)
BM25_K1 = 1.5                       # Term saturation parameter
BM25_B = 0.75                       # Length normalization parameter
```

## 🎓 Learning Objectives

This project demonstrates:

- **Information Retrieval**: Traditional IR techniques (inverted index, TF-IDF, BM25)
- **Neural Search**: Embedding generation, vector similarity, semantic understanding
- **Hybrid Systems**: Combining multiple approaches, score normalization
- **RAG Foundations**: Core concepts needed for Retrieval-Augmented Generation
- **Python Best Practices**: Modular design, CLI tools, caching strategies

## 🤝 Contributing

This is a learning project, but contributions are welcome! Feel free to:

- Report bugs or issues
- Suggest improvements
- Add new search methods
- Improve documentation

## 📝 License

This project is available for educational purposes.

## 🙏 Acknowledgments

- Dataset: Movie descriptions from various sources
- Model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) by Sentence Transformers
- Inspiration: Modern RAG systems and information retrieval techniques
