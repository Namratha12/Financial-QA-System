# src/common/utils.py

import os
import re
from pathlib import Path
from dotenv import load_dotenv
import csv
from typing import List

def load_env():
    load_dotenv()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_env_variable(name: str, default: str = None) -> str:
    value = os.getenv(name)
    if value is None:
        if default is not None:
            return default
        raise EnvironmentError(f"Environment variable '{name}' not set.")
    return value


def normalize_id(doc_id: str) -> str:
    """Removes row suffixes like ::row_1 from document IDs."""
    return doc_id.split("::")[0]


def extract_years_from_text(text: str) -> list[str]:
    """Extracts all 4-digit years (1900–2099) from a string."""
    return re.findall(r"\b(19\d{2}|20\d{2})\b", text)


def try_parse_float(val: str) -> float | None:
    try:
        val = val.replace('%', 'e-2').replace('$', '').replace(',', '').strip()
        return float(val)
    except Exception:
        return None


def is_blank(val: str) -> bool:
    return not val or not val.strip()

def format_prompt(prompt: str) -> str:
    """Clean and format prompt before sending to LLM."""
    return prompt.strip().replace("\n\n", "\n")



def load_csv_data(path: str, limit: int = None) -> List[dict]:
    """
    Load evaluation examples from a CSV file.
    Assumes CSV has 'id', 'question', and 'answer' columns.
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("question", "").strip()]
    return rows[:limit] if limit else rows
