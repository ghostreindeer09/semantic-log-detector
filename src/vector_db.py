"""
Vector Database Module

Implements vector storage and similarity search using FAISS.
This is a lightweight alternative to Milvus suitable for development
and smaller-scale deployments.
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import pickle
import os
from datetime import datetime


@dataclass
class SearchResult:
    """Result of a similarity search."""
    indices: np.ndarray  # Indices of nearest neighbors
    distances: np.ndarray  # Distances to nearest neighbors
    ids: List[str]  # Original IDs of the results
    

@dataclass
class VectorEntry:
    """A single entry in the vector database."""
    id: str
    embedding: np.ndarray
    metadata: Dict = field(default_factory=dict)
    timestamp: Optional[datetime] = None


class VectorDatabase:
    """
    Vector database using FAISS for efficient similarity search.
    
    Supports both flat (exact) and IVF (approximate) search indices
    for different performance/accuracy tradeoffs.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "Flat",
        nlist: int = 100,
        metric: str = "L2"
    ):
        """
        Initialize the vector database.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ("Flat" or "IVF")
            nlist: Number of clusters for IVF index
            metric: Distance metric ("L2" or "IP" for inner product)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.metric = metric
        
        # Storage for metadata
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.metadata: Dict[str, Dict] = {}
        self.timestamps: Dict[str, datetime] = {}
        
        # Initialize FAISS index
        self.index = self._create_index()
        self.is_trained = (index_type == "Flat")
        self.current_idx = 0
        
    def _create_index(self) -> faiss.Index:
        """Create the FAISS index based on configuration."""
        if self.metric == "L2":
            metric = faiss.METRIC_L2
        else:
            metric = faiss.METRIC_INNER_PRODUCT
        
        if self.index_type == "Flat":
            if self.metric == "L2":
                return faiss.IndexFlatL2(self.embedding_dim)
            else:
                return faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            return faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.nlist,
                metric
            )
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def train(self, embeddings: np.ndarray) -> None:
        """
        Train the index (required for IVF indices).
        
        Args:
            embeddings: Training embeddings
        """
        if self.index_type == "IVF" and not self.is_trained:
            embeddings = embeddings.astype(np.float32)
            self.index.train(embeddings)
            self.is_trained = True
            print(f"Index trained with {len(embeddings)} vectors")
    
    def add(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict]] = None,
        timestamps: Optional[List[datetime]] = None
    ) -> None:
        """
        Add embeddings to the database.
        
        Args:
            embeddings: Embeddings to add (n, embedding_dim)
            ids: Unique IDs for each embedding
            metadata: Optional metadata for each embedding
            timestamps: Optional timestamps for each embedding
        """
        if not self.is_trained and self.index_type == "IVF":
            raise RuntimeError("Index must be trained before adding vectors")
        
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        if len(embeddings) != len(ids):
            raise ValueError("Number of embeddings must match number of IDs")
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store mappings and metadata
        for i, id_ in enumerate(ids):
            idx = self.current_idx + i
            self.id_to_idx[id_] = idx
            self.idx_to_id[idx] = id_
            
            if metadata and i < len(metadata):
                self.metadata[id_] = metadata[i]
            
            if timestamps and i < len(timestamps):
                self.timestamps[id_] = timestamps[i]
        
        self.current_idx += len(embeddings)
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
        nprobe: int = 10
    ) -> SearchResult:
        """
        Search for nearest neighbors.
        
        Args:
            query_embeddings: Query embeddings (n_queries, embedding_dim)
            k: Number of neighbors to return
            nprobe: Number of clusters to search (for IVF)
            
        Returns:
            SearchResult with indices, distances, and IDs
        """
        query_embeddings = np.ascontiguousarray(query_embeddings.astype(np.float32))
        
        # Ensure 2D
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Set nprobe for IVF index
        if self.index_type == "IVF":
            self.index.nprobe = nprobe
        
        # Limit k to available vectors
        k = min(k, self.index.ntotal)
        if k == 0:
            return SearchResult(indices=np.array([[]]), distances=np.array([[]]), ids=[[]])
        
        # Perform search
        distances, indices = self.index.search(query_embeddings, k)
        
        # Map indices to IDs
        ids = []
        for query_indices in indices:
            query_ids = [
                self.idx_to_id.get(int(idx), "UNKNOWN")
                for idx in query_indices
                if idx >= 0  # FAISS returns -1 for not found
            ]
            ids.append(query_ids)
        
        return SearchResult(
            indices=indices,
            distances=distances,
            ids=ids
        )
    
    def get_embedding(self, id_: str) -> Optional[np.ndarray]:
        """Get embedding by ID."""
        if id_ not in self.id_to_idx:
            return None
        
        idx = self.id_to_idx[id_]
        return self.index.reconstruct(idx)
    
    def get_metadata(self, id_: str) -> Optional[Dict]:
        """Get metadata by ID."""
        return self.metadata.get(id_)
    
    def get_timestamp(self, id_: str) -> Optional[datetime]:
        """Get timestamp by ID."""
        return self.timestamps.get(id_)
    
    def size(self) -> int:
        """Return number of vectors in the database."""
        return self.index.ntotal
    
    def save(self, filepath: str) -> None:
        """
        Save the database to disk.
        
        Args:
            filepath: Path to save the database
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save metadata
        state = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'metric': self.metric,
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id,
            'metadata': self.metadata,
            'timestamps': self.timestamps,
            'is_trained': self.is_trained,
            'current_idx': self.current_idx
        }
        
        with open(f"{filepath}.meta", 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Database saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'VectorDatabase':
        """
        Load a database from disk.
        
        Args:
            filepath: Path to load the database from
            
        Returns:
            Loaded VectorDatabase instance
        """
        # Load metadata
        with open(f"{filepath}.meta", 'rb') as f:
            state = pickle.load(f)
        
        # Create instance
        db = cls(
            embedding_dim=state['embedding_dim'],
            index_type=state['index_type'],
            nlist=state['nlist'],
            metric=state['metric']
        )
        
        # Load FAISS index
        db.index = faiss.read_index(f"{filepath}.index")
        
        # Restore state
        db.id_to_idx = state['id_to_idx']
        db.idx_to_id = state['idx_to_id']
        db.metadata = state['metadata']
        db.timestamps = state['timestamps']
        db.is_trained = state['is_trained']
        db.current_idx = state['current_idx']
        
        print(f"Database loaded from {filepath} ({db.size()} vectors)")
        return db
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            'total_vectors': self.size(),
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'is_trained': self.is_trained,
            'has_metadata': len(self.metadata) > 0,
            'has_timestamps': len(self.timestamps) > 0
        }


if __name__ == "__main__":
    # Test the vector database
    import numpy as np
    
    # Create database
    db = VectorDatabase(embedding_dim=384, index_type="Flat")
    
    # Generate random embeddings for testing
    np.random.seed(42)
    embeddings = np.random.randn(100, 384).astype(np.float32)
    ids = [f"log_{i}" for i in range(100)]
    metadata = [{"source": "test", "index": i} for i in range(100)]
    
    # Add to database
    db.add(embeddings, ids, metadata)
    print(f"Database size: {db.size()}")
    
    # Search
    query = np.random.randn(1, 384).astype(np.float32)
    results = db.search(query, k=5)
    
    print(f"\nSearch results:")
    print(f"  IDs: {results.ids[0]}")
    print(f"  Distances: {results.distances[0]}")
    
    # Get statistics
    print(f"\nStats: {db.get_stats()}")
