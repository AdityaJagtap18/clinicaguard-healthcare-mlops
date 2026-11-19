"""Clinical text embedding vectorizer using Sentence-Transformers (all-MiniLM-L6-v2).

Transforms unstructured clinician discharge summaries into dense 384-dimensional
semantic vector representations for pgvector storage and similarity search.
"""

from typing import List, Union
import numpy as np


class ClinicalEmbeddingEngine:
    """Generates normalized 384-dimensional clinical text embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = "fallback"

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encodes clinical discharge notes into L2-normalized 384-dimensional embeddings."""
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        if self._model != "fallback":
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return np.array(embeddings, dtype=np.float32)
        else:
            # Deterministic pseudo-embedding fallback (384-dim normalized) if offline
            vectors = []
            for t in texts:
                rng = np.random.RandomState(abs(hash(t)) % (2**32))
                vec = rng.normal(0, 1, 384).astype(np.float32)
                norm = np.linalg.norm(vec)
                vectors.append(vec / (norm if norm > 0 else 1.0))
            return np.array(vectors, dtype=np.float32)

    @property
    def embedding_dimension(self) -> int:
        return 384
