from typing import List


class EmbeddingModelError(RuntimeError):
    """Raised when the sentence-transformers model fails to load."""


class LocalEmbedder:
    _instance = None
    _model = None

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super(LocalEmbedder, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance._model = None
        return cls._instance

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise EmbeddingModelError(
                    f"Failed to load embedding model '{self.model_name}': {e}. "
                    "Ensure sentence-transformers is installed and the model can be downloaded."
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
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single search query string.

        Raises:
            EmbeddingModelError: if the model cannot be loaded.
        """
        results = self.embed_texts([query])
        return results[0] if results else []
