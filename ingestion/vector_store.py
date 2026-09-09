"""
Lightweight numpy-based vector store.

Replaces chromadb (which requires tokenizers<=0.20.3, incompatible with Python 3.14).
Provides an identical public API to the previous ChromaVectorStore so that all
existing imports in agent/ and frontend/ continue to work without changes.

Persistence: data is stored as a single pickle file under persist_dir.
On Streamlit Cloud the filesystem is ephemeral, so the store starts fresh on
each redeployment — same behaviour as the previous ChromaDB setup.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any
from pathlib import Path


class ChromaVectorStore:
    """
    Numpy cosine-similarity vector store with drop-in compatibility for the
    previous ChromaDB-backed implementation.
    """

    def __init__(
        self,
        persist_dir: str = "./data/vector_store",
        collection_name: str = "research_pilot_docs",
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._store_path = self.persist_dir / f"{collection_name}.pkl"

        # In-memory storage
        self._ids: List[str] = []
        self._embeddings: np.ndarray = np.empty((0,), dtype=np.float32)
        self._texts: List[str] = []
        self._metadatas: List[Dict] = []

        self._embedder = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self):
        """Load persisted data if available."""
        if self._store_path.exists():
            try:
                with open(self._store_path, "rb") as f:
                    data = pickle.load(f)
                self._ids = data.get("ids", [])
                self._embeddings = data.get("embeddings", np.empty((0,), dtype=np.float32))
                self._texts = data.get("texts", [])
                self._metadatas = data.get("metadatas", [])
            except Exception as e:
                print(f"[VectorStore] Could not load persisted data: {e}")

    def _save(self):
        """Persist current data to disk."""
        try:
            with open(self._store_path, "wb") as f:
                pickle.dump(
                    {
                        "ids": self._ids,
                        "embeddings": self._embeddings,
                        "texts": self._texts,
                        "metadatas": self._metadatas,
                    },
                    f,
                )
        except Exception as e:
            print(f"[VectorStore] Could not save data: {e}")

    def _get_embedder(self):
        if self._embedder is None:
            from ingestion.embedder import LocalEmbedder
            self._embedder = LocalEmbedder()
        return self._embedder

    # ------------------------------------------------------------------
    # Public API (identical to the previous ChromaDB implementation)
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Takes a list of chunks: [{"text": str, "metadata": dict}]
        Computes embeddings and inserts into the store.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Sanitize metadata values to str/int/float/bool
        sanitized_metadatas = []
        for m in metadatas:
            clean_m = {
                k: v if isinstance(v, (str, int, float, bool)) else str(v)
                for k, v in m.items()
            }
            sanitized_metadatas.append(clean_m)

        ids = [m["chunk_id"] for m in metadatas]

        embedder = self._get_embedder()
        new_embs = np.array(embedder.embed_texts(texts), dtype=np.float32)

        # Upsert: remove existing entries with the same IDs
        if self._ids:
            id_set = set(ids)
            keep = [i for i, eid in enumerate(self._ids) if eid not in id_set]
            if keep:
                self._ids = [self._ids[i] for i in keep]
                self._texts = [self._texts[i] for i in keep]
                self._metadatas = [self._metadatas[i] for i in keep]
                self._embeddings = self._embeddings[keep]
            else:
                self._ids = []
                self._texts = []
                self._metadatas = []
                self._embeddings = np.empty((0,), dtype=np.float32)

        # Append new entries
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metadatas.extend(sanitized_metadatas)

        if self._embeddings.ndim == 1 and self._embeddings.shape[0] == 0:
            self._embeddings = new_embs
        else:
            self._embeddings = np.vstack([self._embeddings, new_embs])

        self._save()
        return len(texts)

    def query(
        self,
        query_text: str,
        n_results: int = 4,
        score_threshold: float = 0.70,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using cosine similarity.
        Returns a list of dicts: {"text", "metadata", "score", "distance", "id"}
        """
        if self._embeddings.ndim == 1 or self._embeddings.shape[0] == 0:
            return []

        embedder = self._get_embedder()
        q_emb = np.array(embedder.embed_query(query_text), dtype=np.float32)

        # Cosine similarity (embeddings are L2-normalised by fastembed)
        sims = self._embeddings @ q_emb  # shape: (n,)

        top_k = min(n_results, len(self._ids))
        top_indices = np.argsort(sims)[::-1][:top_k]

        hits = []
        for idx in top_indices:
            sim = float(sims[idx])
            if sim >= score_threshold:
                hits.append(
                    {
                        "id": self._ids[idx],
                        "text": self._texts[idx],
                        "metadata": self._metadatas[idx],
                        "score": round(sim, 4),
                        "distance": round(1.0 - sim, 4),
                    }
                )

        # Already sorted by descending similarity
        return hits

    def count(self) -> int:
        return len(self._ids)

    def list_sources(self) -> List[Dict[str, Any]]:
        """Returns stats about ingested documents."""
        sources: Dict[str, int] = {}
        for m in self._metadatas:
            src = m.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return [{"source": k, "chunk_count": v} for k, v in sources.items()]

    def clear(self):
        """Reset the store."""
        self._ids = []
        self._embeddings = np.empty((0,), dtype=np.float32)
        self._texts = []
        self._metadatas = []
        if self._store_path.exists():
            try:
                self._store_path.unlink()
            except Exception as e:
                print(f"[VectorStore] Could not delete store file: {e}")
