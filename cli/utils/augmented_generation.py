import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import time

from cli.classes.hybrid_search import HybridSearch
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError


def generate_rag_response(query: str, results: list, api_key: str) -> str:
    """
    Generate a RAG response using retrieved documents and LLM.
    
    Args:
        query: The user's search query
        results: List of search results
        api_key: Gemini API key
        
    Returns:
        Generated response from LLM
    """
    client = genai.Client(api_key=api_key)
    
    # Format documents for the prompt - TRUNCATE descriptions to reduce tokens
    docs = []
    for i, result in enumerate(results, 1):
        doc = result["document"]
        title = doc["title"]
        description = doc.get("description", "")[:150]  # Limit to 150 characters
        docs.append(f"{i}. {title}: {description}")
    
    docs_text = "\n".join(docs)
    
    # Create the RAG prompt
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs_text}

Provide a comprehensive answer that addresses the query:"""
    
    # Call the LLM with retry logic
    max_retries = 3
    retry_delay = 30  # Start with 30 seconds
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt
            )
            return response.text.strip()
        except ClientError as e:
            # Check if it's a rate limit error (429)
            error_message = str(e)
            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                if attempt < max_retries - 1:
                    print(f"Rate limit hit. Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Max retries reached. Please try again later.")
                    raise
            else:
                raise

def generate_summary(query: str, results: list, api_key: str) -> str:
    """
    Generate a multi-document summary using retrieved documents and LLM.
    
    Args:
        query: The user's search query
        results: List of search results
        api_key: Gemini API key
        
    Returns:
        Generated summary from LLM
    """
    client = genai.Client(api_key=api_key)
    
    # Format results for the prompt - include title and truncated description
    formatted_results = []
    for i, result in enumerate(results, 1):
        doc = result["document"]
        title = doc["title"]
        description = doc.get("description", "")[:200]  # Limit to 200 characters
        formatted_results.append(f"{i}. {title}: {description}")
    
    results_text = "\n".join(formatted_results)
    
    # Create the summarization prompt
    prompt = f"""Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search Results:
{results_text}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    
    # Call the LLM with retry logic
    max_retries = 3
    retry_delay = 30  # Start with 30 seconds
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt
            )
            return response.text.strip()
        except ClientError as e:
            # Check if it's a rate limit error (429)
            error_message = str(e)
            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                if attempt < max_retries - 1:
                    print(f"Rate limit hit. Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Max retries reached. Please try again later.")
                    raise
            else:
                raise

def generate_citations_answer(query: str, results: list, api_key: str) -> str:
    """
    Generate an answer with citations using retrieved documents and LLM.
    
    Args:
        query: The user's search query
        results: List of search results
        api_key: Gemini API key
        
    Returns:
        Generated answer with citations from LLM
    """
    client = genai.Client(api_key=api_key)
    
    # Format documents for the prompt with numbers for citation
    documents = []
    for i, result in enumerate(results, 1):
        doc = result["document"]
        title = doc["title"]
        description = doc.get("description", "")[:200]  # Limit to 200 characters
        documents.append(f"[{i}] {title}: {description}")
    
    documents_text = "\n".join(documents)
    
    # Create the citations prompt
    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{documents_text}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    
    # Call the LLM with retry logic
    max_retries = 3
    retry_delay = 30  # Start with 30 seconds
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt
            )
            return response.text.strip()
        except ClientError as e:
            # Check if it's a rate limit error (429)
            error_message = str(e)
            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                if attempt < max_retries - 1:
                    print(f"Rate limit hit. Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Max retries reached. Please try again later.")
                    raise
            else:
                raise

def generate_question_answer(question: str, results: list, api_key: str) -> str:
    """
    Generate a conversational answer to a question using retrieved documents and LLM.
    
    Args:
        question: The user's question
        results: List of search results
        api_key: Gemini API key
        
    Returns:
        Generated answer from LLM
    """
    client = genai.Client(api_key=api_key)
    
    # Format documents for the prompt
    context = []
    for i, result in enumerate(results, 1):
        doc = result["document"]
        title = doc["title"]
        description = doc.get("description", "")[:200]  # Limit to 200 characters
        context.append(f"{i}. {title}: {description}")
    
    context_text = "\n".join(context)
    
    # Create the question answering prompt
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{context_text}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    
    # Call the LLM with retry logic
    max_retries = 3
    retry_delay = 30  # Start with 30 seconds
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt
            )
            return response.text.strip()
        except ClientError as e:
            # Check if it's a rate limit error (429)
            error_message = str(e)
            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                if attempt < max_retries - 1:
                    print(f"Rate limit hit. Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Max retries reached. Please try again later.")
                    raise
            else:
                raise