# src/vector_store/retriever.py

import faiss
import pickle
import numpy as np
from pathlib import Path
from langchain_core.documents import Document

from src.config import config
from .embedding_model import embedding_model


class VectorRetriever:
    def __init__(self):
        self.index = faiss.read_index(str(config.faiss_index_path))
        self.metadata = self._load_metadata(config.metadata_path)

    def _load_metadata(self, metadata_path: Path) -> list[dict]:
        with open(metadata_path, "rb") as f:
            return pickle.load(f)

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """
        Perform FAISS similarity search using the embedding model.
        """
        embedding = embedding_model.embed_query(query)
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                meta = self.metadata[idx]
                content = f"passage: {meta.get('table_markdown', '')}\n\n{meta.get('context', '')}"
                results.append(Document(page_content=content.strip(), metadata={"id": meta.get("id", "")}))

        return results
