# src/vector_store/embedding_model.py

from sentence_transformers import SentenceTransformer
from typing import List
from src.config import config

class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for consistent embedding logic.
    """
    def __init__(self, model_name: str = None):
        self.model_name = config.embedding_model_name
        self.model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of document texts. Prefix for E5 models to improve retrieval relevance
        """
        prefixed = [f"passage: {text.strip()}" for text in texts]
        return self.model.encode(prefixed, convert_to_numpy=True).tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a user query. Prefix added.
        """
        formatted = f"query: {query.strip()}"
        return self.model.encode([formatted])[0].tolist()

embedding_model = EmbeddingModel()
