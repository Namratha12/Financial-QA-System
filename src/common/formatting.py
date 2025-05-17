# src/common/formatting.py

from typing import List

def clean_text(text: str) -> str:
    return text.strip().replace("\xa0", " ").replace("\u200b", " ")

def extract_table_rows(table_markdown: str) -> List[str]:
    """
    Extracts meaningful rows from a markdown table.
    """
    rows = table_markdown.strip().split("\n")
    return [r for r in rows if "|" in r and not r.strip().startswith("| ---")]

def normalize_id(raw_id: str) -> str:
    """Strip row/chunk identifiers to retain just the original document ID."""
    return raw_id.split("::")[0]
