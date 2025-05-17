# src/common/types.py

from typing import TypedDict, List


class EvaluationResult(TypedDict):
    id: str
    question: str
    expected_answer: str
    predicted_answer: str
    generation: str
    accuracy: float
    retrieved_doc_ids: str
    retrieval_precision: float
    retrieval_recall: float
    reranked_doc_ids: str
    reranker_precision: float
    reranker_recall: float
    latency: float
    prompt: str


class DocumentMetadata(TypedDict):
    id: str
    question: str
    answer: str
    table_markdown: str
    context: str

class DocumentChunk(TypedDict):
    id: str
    text: str

class RerankChunk(TypedDict):
    text: str
    id: str
