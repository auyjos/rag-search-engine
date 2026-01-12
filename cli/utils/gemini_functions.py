"""LLM-based query enhancement utilities using Gemini API."""

import json
import re
import time

from google import genai
from google.genai.errors import ClientError
from sentence_transformers import CrossEncoder  # Parse the JSON response


def enhance_query_spelling(query: str, api_key: str) -> str:
    """
    Enhance query by correcting spelling errors using Gemini API.
    
    Args:
        query: Original search query
        api_key: Gemini API key
        
    Returns:
        Enhanced query with corrected spelling
    """
    client = genai.Client(api_key=api_key)
    
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )
    
    # Extract and clean the enhanced query
    enhanced_query = response.text.strip()
    
    # Remove any quotation marks that might be in the response
    enhanced_query = enhanced_query.strip('"').strip("'")
    
    return enhanced_query


def enhance_query_rewrite(query: str, api_key: str) -> str:
    """
    Rewrite vague user queries into more specific, searchable terms using Gemini API.
    
    Args:
        query: Original vague search query
        api_key: Gemini API key
        
    Returns:
        Rewritten query optimized for search
    """
    client = genai.Client(api_key=api_key)
    
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )
    
    # Extract and clean the rewritten query
    rewritten_query = response.text.strip()
    
    # Remove any quotation marks that might be in the response
    rewritten_query = rewritten_query.strip('"').strip("'")
    
    return rewritten_query


def enhance_query_expand(query: str, api_key: str) -> str:
    """
    Expand query with related terms, synonyms, and concepts using Gemini API.
    
    Args:
        query: Original search query
        api_key: Gemini API key
        
    Returns:
        Expanded query with additional relevant terms
    """
    client = genai.Client(api_key=api_key)
    
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )
    
    # Extract and clean the expanded query
    expanded_query = response.text.strip()
    
    # Remove any quotation marks that might be in the response
    expanded_query = expanded_query.strip('"').strip("'")
    
    return expanded_query


def rerank_individual(query: str, results: list, api_key: str) -> list:
    """
    Re-rank search results using individual LLM scoring for each document.
    
    Args:
        query: The search query
        results: List of search results to re-rank
        api_key: Gemini API key
        
    Returns:
        Re-ranked list of results with new scores
    """
    client = genai.Client(api_key=api_key)
    
    reranked_results = []
    total = len(results)
    
    for idx, result in enumerate(results, 1):
        doc = result["document"]
        
        # Print progress
        print(f"Reranking {idx}/{total}: {doc.get('title', 'Unknown')[:50]}...", flush=True)
        
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=prompt
                )
                
                # Check if response and response.text are valid
                if response is None or not hasattr(response, 'text') or response.text is None:
                    print(f"  Empty response from API, using score 0.0", flush=True)
                    result["rerank_score"] = 0.0
                    reranked_results.append(result)
                    break
                
                # Extract the score from the response
                score_text = response.text.strip()
                # Try to parse as float, default to 0 if it fails
                try:
                    rerank_score = float(score_text)
                except ValueError:
                    # If the response contains extra text, try to extract the number
                    numbers = re.findall(r'\d+\.?\d*', score_text)
                    rerank_score = float(numbers[0]) if numbers else 0.0
                
                # Add the rerank score to the result
                result["rerank_score"] = rerank_score
                reranked_results.append(result)
                
                # Success - break retry loop
                break
                
            except ClientError as e:
                if e.status_code == 429:  # Rate limit
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 5 * retry_count  # Exponential backoff
                        print(f"  Rate limit hit, waiting {wait_time}s...", flush=True)
                        time.sleep(wait_time)
                    else:
                        print(f"  Failed after {max_retries} retries, using score 0.0", flush=True)
                        result["rerank_score"] = 0.0
                        reranked_results.append(result)
                else:
                    print(f"  API error: {e}, using score 0.0", flush=True)
                    result["rerank_score"] = 0.0
                    reranked_results.append(result)
                    break
                    
            except Exception as e:
                print(f"  Unexpected error: {e}, using score 0.0", flush=True)
                result["rerank_score"] = 0.0
                reranked_results.append(result)
                break
        
        # Sleep between successful requests to avoid rate limiting
        if retry_count == 0 and idx < total:
            time.sleep(1)  # Reduced from 3 to 1 second
    
    print()  # Add newline after progress messages
    
    # Sort by rerank score in descending order
    reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked_results


def rerank_batch(query: str, results: list, api_key: str) -> list:
    """
    Rerank search results using a single batch LLM call.
    
    Args:
        query: The search query
        results: List of search results to rerank
        api_key: Gemini API key
        
    Returns:
        Reranked results with 'rerank_rank' field added
    """
    import json
    
    if not results:
        return []
    
    client = genai.Client(api_key=api_key)
    
    # Build the document list string
    doc_list_lines = []
    for idx, result in enumerate(results):
        doc = result['document']
        doc_id = result['doc_id']
        title = doc.get('title', 'Unknown')
        description = doc.get('description', '')  # Use full description for best context
        bm25_rank = result.get('bm25_rank', '-')
        semantic_rank = result.get('semantic_rank', '-')
        doc_list_lines.append(f"{doc_id}. {title} [BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}]: {description}")
    
    doc_list_str = "\n".join(doc_list_lines)
    
    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt
            )
            
            # Check if response is valid
            if response is None or not hasattr(response, 'text') or response.text is None:
                print(f"Empty response from API, using original order")
                # Return original order with ranks
                for idx, result in enumerate(results):
                    result['rerank_rank'] = idx + 1
                return results
            
            # Extract and parse JSON response
            response_text = response.text.strip()
            
            # Try to extract JSON array from response (handles cases with extra text)
            json_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
            if json_match:
                json_str = '[' + json_match.group(1) + ']'
                ranked_ids = json.loads(json_str)
            else:
                # Try parsing the whole response as JSON
                ranked_ids = json.loads(response_text)
            
            # Create a mapping from doc_id to rank
            rank_map = {doc_id: idx + 1 for idx, doc_id in enumerate(ranked_ids)}
            
            # Add rerank_rank to each result
            for result in results:
                doc_id = result['doc_id']
                result['rerank_rank'] = rank_map.get(doc_id, len(ranked_ids) + 1)
            
            # Sort by rerank rank
            results.sort(key=lambda x: x['rerank_rank'])
            
            return results
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response was: {response_text[:200]}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying... ({retry_count}/{max_retries})")
                time.sleep(2)
            else:
                print("Max retries reached, using original order")
                for idx, result in enumerate(results):
                    result['rerank_rank'] = idx + 1
                return results
                
        except ClientError as e:
            if e.status_code == 429:  # Rate limit
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 5 * retry_count
                    print(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} retries, using original order")
                    for idx, result in enumerate(results):
                        result['rerank_rank'] = idx + 1
                    return results
            else:
                print(f"API error: {e}, using original order")
                for idx, result in enumerate(results):
                    result['rerank_rank'] = idx + 1
                return results
                
        except Exception as e:
            print(f"Unexpected error: {e}, using original order")
            for idx, result in enumerate(results):
                result['rerank_rank'] = idx + 1
            return results
    
    # Fallback: return original order
    for idx, result in enumerate(results):
        result['rerank_rank'] = idx + 1
    return results

def rerank_cross_encoder(query:str, results:list)->list:
    """
        Rerank search results using a cross-encoder model.
        
        Args:
            query: The search query
            results: List of search results to rerank
            
        Returns:
            Reranked results with 'cross_encoder_score' field added
        """
    if not results:
        return []
    
    pairs = []
    for result in results:
        doc = result['document']
        doc_str = f"{doc.get('title', '')} - {doc.get('description', '')}"
        pairs.append([query, doc_str])
     # Initialize cross-encoder model (only once per query)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    # Compute scores for all pairs
    scores = cross_encoder.predict(pairs)
    
    # Add scores to results
    for idx, result in enumerate(results):
        result['cross_encoder_score'] = float(scores[idx])
    
    # Sort by cross-encoder score in descending order
    results.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
    
    return results

def evaluate_search_results(query:str, results:list, api_key:str) -> None:
    """
    Evaluate search results using LLM to rate relevance on a 0-3 scale.
    
    Args:
        query: The search query
        results: List of search results
        api_key: Gemini API key
    """

    client= genai.Client(api_key=api_key)
    formatted_results = []
    for i, result in enumerate(results,1):
        doc = result["document"]
        title = doc["title"]
        description = doc.get("description", "")
        formatted_results.append(f"{i}. {title}: {description}")
    
     # Create the evaluation prompt
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers out than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]"""
        
    # Call the LLM
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )

    
    if response is None or not hasattr(response, 'text') or response.text is None:
        print("Empty response from API, cannot evaluate results")
        return
    
    response_text = response.text.strip()
    
    # Remove markdown code blocks if present
    if response_text.startswith("```"):
        # Extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
    
    try:
        scores = json.loads(response_text)
        
        # Print evaluation report
        print("Evaluation Report:")
        for i, (result, score) in enumerate(zip(results, scores), 1):
            title = result["document"]["title"]
            print(f"{i}. {title}: {score}/3")
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response was: {response_text}")