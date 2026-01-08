# RAG Search Engine

A comprehensive information retrieval system implementing multiple search methodologies including keyword-based search (BM25), semantic search, and hybrid search approaches. Built as a learning project to understand the fundamentals of Retrieval-Augmented Generation (RAG) systems.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Keyword Search](#keyword-search)
  - [Semantic Search](#semantic-search)
  - [Hybrid Search](#hybrid-search)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

## 🎯 Overview

This project demonstrates the core concepts behind modern search systems by implementing and comparing different search approaches:

- **Keyword Search**: Traditional BM25 (Best Matching 25) algorithm for exact and fuzzy keyword matching
- **Semantic Search**: Neural embedding-based search that understands meaning and context
- **Hybrid Search**: Combines keyword and semantic approaches with configurable weighting

The system is built around a dataset of ~5,000 movies, providing a rich corpus for search experimentation.

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

3. **Hybrid Search**
   - Min-max score normalization
   - Weighted combination of BM25 and semantic scores
   - Configurable alpha parameter (0.0 = pure semantic, 1.0 = pure keyword)
   - Best of both worlds: handles exact matches and conceptual queries

### Additional Features

- **Caching**: Embeddings and indexes are cached for performance
- **Chunking**: Smart text chunking with sentence boundaries and overlap
- **CLI Tools**: Easy-to-use command-line interfaces for all search methods
- **Extensible**: Modular design for adding new search methods

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

# Emphasize semantics (alpha=0.2 means 20% keyword, 80% semantic)
uv run cli/hybrid_search_cli.py weighted-search "family movies" --alpha 0.2 --limit 10

# Balanced approach
uv run cli/hybrid_search_cli.py weighted-search "2015 comedies" --alpha 0.5 --limit 10
```

### Alpha Parameter Guide

The alpha (α) parameter controls the balance between keyword and semantic search:

| Alpha | Distribution | Best For | Example |
|-------|--------------|----------|---------|
| 1.0 | 100% Keyword | Exact titles, names, IDs | "The Revenant" |
| 0.8 | 80% Keyword, 20% Semantic | Specific terms with some context | "Leonardo DiCaprio survival" |
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
│   │   ├── hybrid_search.py      # Hybrid search implementation
│   │   ├── invert_index.py       # Inverted index & BM25
│   │   └── semantic_search.py    # Legacy compatibility layer
│   ├── commands/                 # Command handlers
│   │   ├── embedding_commands.py # Embedding generation commands
│   │   └── search_commands.py    # Search commands
│   ├── utils/                    # Utility functions
│   │   ├── cache.py              # Caching for embeddings
│   │   ├── chunking.py           # Text chunking utilities
│   │   └── similarity.py         # Similarity calculations
│   └── config.py                 # Configuration settings
├── data/                         # Data files
│   ├── movies.json               # Movie dataset (~5,000 movies)
│   └── stopwords.txt             # Stopwords for keyword search
├── helpers/                      # Helper modules
│   ├── constants.py              # BM25 constants
│   └── tokenizer.py              # Text tokenization
├── cache/                        # Generated cache files
│   ├── chunk_embeddings.npy      # Cached chunk embeddings
│   ├── chunk_metadata.json       # Chunk metadata
│   ├── index.pkl                 # Inverted index
│   └── ...                       # Other cache files
└── pyproject.toml                # Project configuration
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
