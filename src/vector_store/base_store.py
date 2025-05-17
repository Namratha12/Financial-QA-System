# src/vector_store/base_store.py

import faiss
import pickle
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from src.config import config
from src.vector_store.embedding_model import EmbeddingModel

class FaissVectorStore:
    def __init__(self, 
                 index_path: Path,
                 metadata_path: Path,
                 embedding_model_name: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model = EmbeddingModel(embedding_model_name)

        # Load index and metadata
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        query_vector = self.embedding_model.embed_query(query)
        query_vector = [query_vector] 

        _, indices = self.index.search(query_vector, k)
        return [self._convert_to_document(self.metadata[i]) for i in indices[0] if i < len(self.metadata)]

    def _convert_to_document(self, meta: dict) -> Document:
        page_content = f"passage: {meta.get('table_markdown', '')}\n\n{meta.get('context', '')}"
        return Document(
            page_content=page_content.strip(),
            metadata={"id": meta.get("id", "")}
        )


# Global instance (used in agent)
vector_store = FaissVectorStore(
    index_path=config.faiss_index_path,
    metadata_path=config.metadata_path,
    embedding_model_name=config.embedding_model_name
)
