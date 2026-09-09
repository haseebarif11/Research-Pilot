import os
from typing import List, Dict, Any, Optional
from pathlib import Path

class ChromaVectorStore:
    def __init__(self, persist_dir: str = "./data/chroma", collection_name: str = "research_pilot_docs"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)
        self._client = None
        self._collection = None
        self._embedder = None
        self._init_db()

    def _init_db(self):
        try:
            import chromadb
            from ingestion.embedder import LocalEmbedder

            self._embedder = LocalEmbedder()
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            # Pass embedding_function=None because we supply our own embeddings;
            # this prevents chromadb 0.5+ from downloading its default ONNX model.
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"[ChromaVectorStore] Initializing ChromaDB error: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Takes a list of chunks: [{"text": str, "metadata": dict}]
        Computes embeddings and inserts into ChromaDB collection.
        """
        if not chunks or self._collection is None:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # Ensure metadata values are str/int/float/bool
        sanitized_metadatas = []
        for m in metadatas:
            clean_m = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_m[k] = v
                else:
                    clean_m[k] = str(v)
            sanitized_metadatas.append(clean_m)

        ids = [m["chunk_id"] for m in metadatas]
        embeddings = self._embedder.embed_texts(texts)

        # Batch upsert
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            end = i + batch_size
            self._collection.upsert(
                ids=ids[i:end],
                documents=texts[i:end],
                metadatas=sanitized_metadatas[i:end],
                embeddings=embeddings[i:end]
            )

        return len(texts)

    def query(self, query_text: str, n_results: int = 4, score_threshold: float = 0.70) -> List[Dict[str, Any]]:
        """
        Searches ChromaDB for relevant chunks.
        Cosine distance in Chroma: 0 is exact match, 1 is orthogonal, 2 is opposite.
        Cosine similarity = 1 - distance.
        Returns list of matching dicts: {"text": str, "metadata": dict, "score": float}
        """
        if self._collection is None:
            return []

        q_embedding = self._embedder.embed_query(query_text)
        try:
            results = self._collection.query(
                query_embeddings=[q_embedding],
                n_results=min(n_results, max(1, self.count()))
            )
        except Exception as e:
            print(f"[ChromaVectorStore] Query error: {e}")
            return []

        hits = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            ids = results["ids"][0] if "ids" in results else [""] * len(docs)

            for doc, meta, dist, cid in zip(docs, metas, distances, ids):
                # distance is cosine distance; similarity = 1 - dist
                sim = 1.0 - float(dist)
                if sim >= score_threshold:
                    hits.append({
                        "id": cid,
                        "text": doc,
                        "metadata": meta,
                        "score": round(sim, 4),
                        "distance": round(float(dist), 4)
                    })

        # Sort by highest similarity
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits

    def count(self) -> int:
        if self._collection:
            try:
                return self._collection.count()
            except Exception:
                return 0
        return 0

    def list_sources(self) -> List[Dict[str, Any]]:
        """
        Returns stats about ingested documents.
        """
        if self._collection is None:
            return []
        try:
            all_meta = self._collection.get(include=["metadatas"])["metadatas"]
            sources: Dict[str, int] = {}
            for m in all_meta:
                src = m.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
            return [{"source": k, "chunk_count": v} for k, v in sources.items()]
        except Exception:
            return []

    def clear(self):
        """Reset the collection."""
        if self._client and self.collection_name:
            try:
                self._client.delete_collection(name=self.collection_name)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"[ChromaVectorStore] Clear error: {e}")
