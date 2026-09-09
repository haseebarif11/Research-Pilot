from typing import List


class EmbeddingModelError(RuntimeError):
    """Raised when the embedding model fails to load."""


class LocalEmbedder:
    """
    Lightweight embedder backed by fastembed (onnxruntime-based).

    Uses BAAI/bge-small-en-v1.5 — a 384-dim model similar to all-MiniLM-L6-v2.
    No torch or Rust-compiled tokenizers required; compatible with Python 3.14+.

    Requirements:
        pip install fastembed>=0.3.6
    """

    _instance = None
    _model = None

    def __new__(cls, model_name: str = "BAAI/bge-small-en-v1.5"):
        if cls._instance is None:
            cls._instance = super(LocalEmbedder, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance._model = None
        return cls._instance

    def _ensure_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as e:
                raise EmbeddingModelError(
                    f"Failed to load embedding model '{self.model_name}': {e}. "
                    "Ensure fastembed is installed: pip install fastembed>=0.3.6"
                ) from e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings into normalized vector embeddings.

        Raises:
            EmbeddingModelError: if the model cannot be loaded.
        """
        if not texts:
            return []

        self._ensure_model()
        # fastembed.TextEmbedding.embed() returns a generator of numpy arrays
        return [emb.tolist() for emb in self._model.embed(texts)]

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single search query string.

        Raises:
            EmbeddingModelError: if the model cannot be loaded.
        """
        results = self.embed_texts([query])
        return results[0] if results else []
