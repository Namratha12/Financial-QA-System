# src/evaluation/runner.py

import time
import csv
import pandas as pd
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import config
from src.agent.pipeline import run_agent_pipeline
from src.common.types import EvaluationResult
from src.common.utils import load_csv_data, normalize_id
from src.evaluation.metrics import compute_accuracy, compute_precision, compute_recall


def evaluate_single_example(row: dict) -> EvaluationResult:
    question = row["question"]
    expected = row["answer"]
    expected_doc_id = row["id"]
    ex_id = row["id"]

    start_time = time.time()
    result = run_agent_pipeline(question)
    latency = time.time() - start_time

    answer = result.answer
    generation = result.generation
    doc_ids = [doc.metadata.get("id", "") for doc in result.documents]
    reranked_ids = [doc.metadata.get("id", "") for doc in result.reranked_documents]
    normalized_doc_ids = [normalize_id(doc_id) for doc_id in doc_ids]
    normalized_reranked_ids = [normalize_id(doc_id) for doc_id in reranked_ids]
    accuracy = compute_accuracy(question, answer, expected)
    retrieval_precision = compute_precision(normalized_doc_ids, expected_doc_id)
    retrieval_recall = compute_recall(normalized_doc_ids, expected_doc_id)
    reranker_precision = compute_precision(normalized_reranked_ids, expected_doc_id)
    reranker_recall = compute_recall(normalized_reranked_ids, expected_doc_id)
    return EvaluationResult(
        id=ex_id,
        question=question,
        expected_answer=expected,
        predicted_answer=answer,
        generation=generation,
        accuracy=accuracy,
        retrieved_doc_ids=", ".join(doc_ids),
        retrieval_precision=retrieval_precision,
        retrieval_recall=retrieval_recall,
        reranked_doc_ids=", ".join(reranked_ids),
        reranker_precision=reranker_precision,
        reranker_recall=reranker_recall,
        latency=latency,
        prompt=result.prompt,
    )


def run_evaluation(save_path: str = "eval_local.csv", limit: int = 500):
    data = load_csv_data(config.data_path, limit=limit)
    results: List[EvaluationResult] = []

    print(f"[INFO] Evaluating {len(data)} examples...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(evaluate_single_example, row) for row in data]
        for future in as_completed(futures):
            results.append(future.result())

    df = pd.DataFrame(results)
    df.to_csv(save_path, quoting=csv.QUOTE_NONNUMERIC, index=False)

    # Summary
    print("\n[RESULTS SUMMARY]")
    print(f"Average Accuracy: {df['accuracy'].mean():.2%}")
    print(f"Average Retrieval Precision: {df['retrieval_precision'].mean():.2%}")
    print(f"Average Retrieval Recall: {df['retrieval_recall'].mean():.2%}")
    print(f"Average Rerank Precision: {df['reranker_precision'].mean():.2%}")
    print(f"Average Rerank Recall: {df['reranker_recall'].mean():.2%}")
    print(f"Average Latency: {df['latency'].mean():.2f}s")
    print(f"Results saved to {save_path}")
