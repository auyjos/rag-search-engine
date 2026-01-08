import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

# Load environment variables from .env file
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

# Create a Gemini client
client = genai.Client(api_key=api_key)

# Try to generate content with retry logic
max_retries = 3
retry_delay = 60  # seconds

for attempt in range(max_retries):
    try:
        # Generate content using the Gemini model
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
        )
        
        # Print the response text
        print("\nResponse:")
        print(response.text)
        
        # Print token usage
        print(f"\nPrompt Tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")
        
        # Success - exit the loop
        break
        
    except ClientError as e:
        if e.status_code == 429:  # Rate limit exceeded
            if attempt < max_retries - 1:
                print(f"\nRate limit exceeded. Waiting {retry_delay} seconds before retry {attempt + 2}/{max_retries}...")
                time.sleep(retry_delay)
            else:
                print(f"\nRate limit exceeded after {max_retries} attempts.")
                print("Please wait a few minutes and try again.")
                raise
        else:
            # Other API errors
            print(f"\nAPI Error: {e}")
            raise