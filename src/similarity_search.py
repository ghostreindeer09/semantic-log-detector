"""
Similarity Search Module - k-NN based similarity search for logs.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SimilarityResult:
    """Result of a similarity search."""
    query_id: str
    query_text: str
    neighbors: List[Dict]
    mean_distance: float
    min_distance: float
    max_distance: float


class SimilaritySearcher:
    """k-NN similarity searcher for log entries."""
    
    def __init__(self, vector_db, encoder, k_neighbors: int = 5, nprobe: int = 10):
        self.vector_db = vector_db
        self.encoder = encoder
        self.k_neighbors = k_neighbors
        self.nprobe = nprobe
    
    def search_by_embedding(self, embedding: np.ndarray, k: Optional[int] = None):
        k = k or self.k_neighbors
        return self.vector_db.search(embedding, k=k, nprobe=self.nprobe)
    
    def search_by_text(self, text: str, k: Optional[int] = None) -> SimilarityResult:
        k = k or self.k_neighbors
        embedding = self.encoder.encode(text)
        search_result = self.search_by_embedding(embedding, k)
        
        neighbors = []
        for i, (idx, id_, dist) in enumerate(zip(
            search_result.indices[0], search_result.ids[0], search_result.distances[0]
        )):
            if idx >= 0:
                metadata = self.vector_db.get_metadata(id_) or {}
                neighbors.append({'rank': i + 1, 'id': id_, 'distance': float(dist), 'metadata': metadata})
        
        distances = search_result.distances[0]
        valid_distances = distances[distances >= 0]
        
        return SimilarityResult(
            query_id="query", query_text=text, neighbors=neighbors,
            mean_distance=float(np.mean(valid_distances)) if len(valid_distances) > 0 else float('inf'),
            min_distance=float(np.min(valid_distances)) if len(valid_distances) > 0 else float('inf'),
            max_distance=float(np.max(valid_distances)) if len(valid_distances) > 0 else float('inf')
        )
    
    def batch_search(self, embeddings: np.ndarray, ids: List[str], k: Optional[int] = None):
        k = k or self.k_neighbors
        return self.vector_db.search(embeddings, k=k, nprobe=self.nprobe)
    
    def get_neighborhood_stats(self, embedding: np.ndarray, k: Optional[int] = None) -> Dict:
        k = k or self.k_neighbors
        result = self.search_by_embedding(embedding, k)
        distances = result.distances[0]
        valid_distances = distances[distances >= 0]
        
        if len(valid_distances) == 0:
            return {'num_neighbors': 0, 'mean_distance': float('inf')}
        
        return {
            'num_neighbors': len(valid_distances),
            'mean_distance': float(np.mean(valid_distances)),
            'std_distance': float(np.std(valid_distances)),
            'min_distance': float(np.min(valid_distances)),
            'max_distance': float(np.max(valid_distances))
        }
