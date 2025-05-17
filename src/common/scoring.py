# src/common/scoring.py

from typing import List


def relative_score(a: float, b: float, power: int = 2) -> float:
    """
    Computes a relative score between two numbers, penalizing larger errors.
    Returns a value in [0.0, 1.0].
    """
    if a == b:
        return 1.0
    try:
        return 1 - ((abs(a - b) / max(abs(a), abs(b))) ** power)
    except ZeroDivisionError:
        return 0.0


def exact_match(predicted: str, expected: str) -> bool:
    return predicted.strip().lower() == expected.strip().lower()


def numeric_match(predicted: str, expected: str) -> float:
    """
    Attempts to match numbers even with format variation (%, commas, etc.).
    Returns score in [0.0, 1.0].
    """
    def parse_number(val: str) -> float:
        return float(val.replace('%', 'e-2').replace('$', '').replace(',', '').strip())

    try:
        parsed_pred = parse_number(predicted)
        parsed_exp = parse_number(expected)
        return relative_score(parsed_pred, parsed_exp)
    except Exception:
        return 0.0


def precision(predicted_ids: List[str], expected_id: str) -> float:
    return float(expected_id in predicted_ids) / len(predicted_ids) if predicted_ids else 0.0


def recall(predicted_ids: List[str], expected_id: str) -> float:
    return float(expected_id in predicted_ids) if predicted_ids else 0.0
