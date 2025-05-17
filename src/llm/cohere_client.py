# src/llm/cohere_client.py

import os
from dotenv import load_dotenv
import cohere

load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(cohere_api_key)
