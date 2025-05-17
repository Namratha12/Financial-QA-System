from src.common.scoring import exact_match, numeric_match
from src.common.utils import normalize_id
from typing import List

def compute_accuracy(question: str, predicted: str, expected: str) -> float:
    """
    Computes answer accuracy using exact and numeric matching.
    """
    if not predicted and expected:
        return 0.0
    if exact_match(predicted, expected):
        return 1.0
    return numeric_match(predicted, expected)

def compute_precision(predicted_ids: List[str], expected_id: str) -> float:
    """
    Precision = correct retrievals / total retrieved
    """
    normalized_ids = [normalize_id(i) for i in predicted_ids]
    return float(expected_id in normalized_ids) / len(predicted_ids) if predicted_ids else 0.0

def compute_recall(predicted_ids: List[str], expected_id: str) -> float:
    """
    Recall = correct retrievals / total relevant
    (Assumes only 1 relevant doc for now)
    """
    normalized_ids = [normalize_id(i) for i in predicted_ids]
    return float(expected_id in normalized_ids) if predicted_ids else 0.0
