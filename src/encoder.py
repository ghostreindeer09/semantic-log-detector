"""
Sentence-BERT Encoder Module

Generates semantic embeddings for log messages using pre-trained
transformer models.
"""

import numpy as np
from typing import List, Union, Optional
import torch
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: np.ndarray
    texts: List[str]
    model_name: str
    embedding_dim: int


class LogEncoder:
    """
    Sentence-BERT based encoder for log messages.
    
    Uses pre-trained transformer models to generate dense vector
    representations of log text.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the encoder.
        
        Args:
            model_name: Name of the Sentence-BERT model
            device: Device to use ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load the model
        print(f"Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_with_metadata(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False
    ) -> EmbeddingResult:
        """
        Encode texts and return with metadata.
        
        Args:
            texts: Single text or list of texts to encode
            show_progress: Whether to show progress bar
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.encode(texts, show_progress)
        
        return EmbeddingResult(
            embeddings=embeddings,
            texts=texts,
            model_name=self.model_name,
            embedding_dim=self.embedding_dim
        )
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        if self.normalize_embeddings:
            # Already normalized, just dot product
            return float(np.dot(embedding1, embedding2))
        else:
            # Compute cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def batch_similarity(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise similarities between query and corpus embeddings.
        
        Args:
            query_embeddings: Query embeddings (n_queries, embedding_dim)
            corpus_embeddings: Corpus embeddings (n_corpus, embedding_dim)
            
        Returns:
            Similarity matrix (n_queries, n_corpus)
        """
        if self.normalize_embeddings:
            return np.dot(query_embeddings, corpus_embeddings.T)
        else:
            # Normalize and compute
            query_norm = query_embeddings / np.linalg.norm(
                query_embeddings, axis=1, keepdims=True
            )
            corpus_norm = corpus_embeddings / np.linalg.norm(
                corpus_embeddings, axis=1, keepdims=True
            )
            return np.dot(query_norm, corpus_norm.T)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'normalize_embeddings': self.normalize_embeddings,
            'batch_size': self.batch_size
        }


if __name__ == "__main__":
    # Test the encoder
    encoder = LogEncoder()
    
    test_texts = [
        "failed to authenticate user invalid credentials",
        "user login successful authentication completed",
        "database connection timeout after 30 seconds",
        "memory usage critical threshold exceeded",
    ]
    
    print("\nEncoding test texts...")
    result = encoder.encode_with_metadata(test_texts, show_progress=True)
    
    print(f"\nEmbedding shape: {result.embeddings.shape}")
    print(f"Model: {result.model_name}")
    
    # Compute pairwise similarities
    print("\nPairwise similarities:")
    for i, text1 in enumerate(test_texts):
        for j, text2 in enumerate(test_texts):
            if i < j:
                sim = encoder.compute_similarity(
                    result.embeddings[i],
                    result.embeddings[j]
                )
                print(f"  '{text1[:30]}...' <-> '{text2[:30]}...': {sim:.4f}")
