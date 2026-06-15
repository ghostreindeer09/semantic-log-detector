"""
Log Encoder Module — Sentence-BERT wrapper for log embedding.

Provides the LogEncoder class that was missing from the codebase,
unblocking src/detection/pipeline.py and demo.py.
"""

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)


class LogEncoder:
    """
    Encodes log text into dense vector embeddings using Sentence-BERT.

    Wraps the sentence-transformers library with lazy model loading
    to avoid paying the initialization cost if the encoder is never used.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Args:
            model_name: Name of the sentence-transformers model.
            device: Device to run the model on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded. Embedding dim: %d", self.embedding_dim)
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more texts into embeddings.

        Args:
            texts: Single string or list of strings to encode.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.

        Returns:
            np.ndarray of shape (n_texts, embedding_dim) or (embedding_dim,) for
            a single string input.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )

        if single:
            return embeddings[0]
        return embeddings
