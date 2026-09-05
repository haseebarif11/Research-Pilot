from typing import List
import numpy as np

class LocalEmbedder:
    _instance = None
    _model = None

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super(LocalEmbedder, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        # Lazy load on first embed call
        pass

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[LocalEmbedder] Warning: sentence-transformers not initialized ({e}).")
                self._model = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings into normalized vector embeddings.
        """
        if not texts:
            return []

        self._ensure_model()
        if self._model is not None:
            embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return embeddings.tolist()
        else:
            # Deterministic fallback embedding for testing if model download is pending
            dim = 384
            res = []
            for t in texts:
                np.random.seed(abs(hash(t)) % (2**32))
                vec = np.random.randn(dim).astype(np.float32)
                vec /= np.linalg.norm(vec) + 1e-9
                res.append(vec.tolist())
            return res

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single search query string.
        """
        results = self.embed_texts([query])
        return results[0] if results else []
