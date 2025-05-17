# src/llm/openai_llm.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4")

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.0,
    api_key=OPENAI_API_KEY
)
