# src/config.py

from pydantic import BaseModel
from pathlib import Path


class Config(BaseModel):
    vector_store_dir: Path = Path("src//vector_database")
    data_path: Path = Path("data/parsed_convfinqa.csv")
    faiss_index_path: Path = vector_store_dir/"faiss_index.bin"
    metadata_path: Path = vector_store_dir/"faiss_metadata.pkl"

    # Evaluation + retrieval
    evaluation_sample_limit: int = 500
    use_ground_truth_retrieval: bool = False

    # Embedding + reranking
    embedding_model_name: str = "intfloat/e5-base-v2"
    reranker_model_name: str = "rerank-english-v3.0"
    top_k_retrieval: int = 10
    top_k_rerank: int = 5

    # LLM generation
    disable_llm_generation: bool = False
    temperature: float = 0.0
    top_p: float = 0.95


# Singleton instance
config = Config()
