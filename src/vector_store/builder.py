# src/vector_store/builder.py

import pandas as pd
import faiss
import pickle
from pathlib import Path
from typing import List
from src.config import config
from src.common.types import DocumentChunk
from src.vector_store.embedding_model import embedding_model
from src.common.utils import ensure_dir  

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class IndexBuilder:
    def __init__(self):
        self.index_path = config.faiss_index_path
        self.metadata_path = config.metadata_path
        self.data_path = config.data_path

    def load_data(self) -> List[DocumentChunk]:
        df = pd.read_csv(self.data_path).fillna("")
        chunks = []
        for _, row in df.iterrows():
            table_rows = row["table_markdown"].strip().split("\n")
            for i, table_row in enumerate(table_rows):
                chunk_id = f"{row['id']}::row_{i}"
                combined = f"{table_row.strip()}\n\n{row['context'].strip()}"
                chunks.append(DocumentChunk(id=chunk_id, text=combined))
        return chunks

    def build_faiss_index(self, embeddings: List[List[float]]) -> faiss.IndexFlatL2:
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))
        return index

    def run(self):
        print("[INFO] Loading and chunking data...")
        chunks = self.load_data()

        print("[INFO] Embedding chunks...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_model.embed_documents(texts)

        print("[INFO] Building FAISS index...")
        index = self.build_faiss_index(embeddings)
        ensure_dir(self.index_path.parent) 

        print(f"[INFO] Saving index to {self.index_path}...")
        faiss.write_index(index, str(self.index_path))

        print(f"[INFO] Saving metadata to {self.metadata_path}...")
        with open(self.metadata_path, "wb") as f:
            pickle.dump(chunks, f)


        print(" Index build complete.")


if __name__ == "__main__":
    import numpy as np
    builder = IndexBuilder()
    builder.run()
