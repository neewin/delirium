"""
Module 2: Embedding Generation and Semantic Retrieval.

Takes chunks from Module 1, embeds them with a Sentence Transformer,
indexes them in FAISS (cosine similarity via normalized inner product),
and lets you semantically search for the most relevant chunks.
"""

import os
import json
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticRetriever:
    """Embeds document chunks, builds a FAISS index, and runs semantic search."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []

    # -- Part 1: Load model (once) -----------------------------------------
    def load_model(self) -> None:
        """Load the Sentence Transformer model if not already loaded.

        Loading is slow (reads model weights from disk into memory), so we
        do it once and reuse it for every embedding call instead of
        reloading per chunk/query, which would be very wasteful.
        """
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    # -- Part 2: Embed chunks -------------------------------------------
    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Generate normalized embeddings for a list of Module 1 chunks.

        Order is preserved: embeddings[i] corresponds to chunks[i].
        """
        if not chunks:
            raise ValueError("No chunks provided for embedding.")

        self.load_model()
        texts = [c["text"] for c in chunks]
        if any(not t.strip() for t in texts):
            raise ValueError("One or more chunks contain empty text.")

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")  # FAISS requires float32
        faiss.normalize_L2(embeddings)  # normalize for cosine similarity
        return embeddings

    # -- Part 3: Build FAISS index ----------------------------------------
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create a flat FAISS index using inner product.

        IndexFlatIP computes the dot product between vectors. When vectors
        are L2-normalized (unit length), the dot product is mathematically
        equivalent to cosine similarity. This keeps the index simple and
        exact (no approximation) — ideal for a small/medium mini-project.
        """
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    # -- Part 9 helper: build everything in one call -----------------------
    def build_index(self, chunks: List[Dict]) -> None:
        """Embed chunks, build the FAISS index, and store metadata."""
        embeddings = self.generate_embeddings(chunks)
        self.index = self.create_faiss_index(embeddings)
        # Metadata position i must match FAISS vector position i.
        self.metadata = [
            {
                "chunk_id": c["chunk_id"],
                "source": c["source"],
                "page": c["page"],
                "text": c["text"],
            }
            for c in chunks
        ]

    # -- Part 5: Embed a query ----------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        """Embed and normalize a user question using the same model."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only.")

        self.load_model()
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding)
        return query_embedding

    # -- Part 6 & 7: Search ---------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return the top_k chunks most similar to the query, with scores.

        Score is a cosine similarity in [-1, 1] (in practice usually 0-1
        for natural text): higher means more semantically similar. There
        is no universal "good" threshold — what counts as relevant depends
        on the embedding model and the dataset, so treat scores as a
        relative ranking signal, not an absolute guarantee of correctness.
        """
        if self.index is None:
            raise RuntimeError("Index has not been built or loaded yet.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_vector = self.embed_query(query)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 when fewer than k results exist
                continue
            chunk_meta = self.metadata[idx]
            results.append({
                "chunk_id": chunk_meta["chunk_id"],
                "source": chunk_meta["source"],
                "page": chunk_meta["page"],
                "text": chunk_meta["text"],
                "score": float(score),
            })
        return results

    # -- Part 8: Save / Load -------------------------------------------
    def save(self, index_path: str, metadata_path: str) -> None:
        """Persist the FAISS index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Build the index first.")

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str, metadata_path: str) -> None:
        """Load a previously saved FAISS index and metadata from disk."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        try:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to load index/metadata (possibly corrupted): {exc}")

        if self.index.ntotal != len(self.metadata):
            raise RuntimeError("Index and metadata are out of sync (counts differ).")
