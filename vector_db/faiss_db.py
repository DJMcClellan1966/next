"""
FAISS Vector Database Implementation
Fast similarity search with GPU support
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Install with: pip install faiss-cpu (or faiss-gpu for GPU)")

from .base import VectorDatabase


class FAISSVectorDB(VectorDatabase):
    """FAISS-based vector database with GPU support"""
    
    def __init__(self, dimension: int = 384, index_type: str = 'L2', use_gpu: bool = False):
        """
        Initialize FAISS vector database
        
        Args:
            dimension: Vector dimension
            index_type: 'L2' (Euclidean) or 'IP' (Inner Product/Cosine)
            use_gpu: Whether to use GPU acceleration
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        
        # Create FAISS index
        if index_type == 'L2':
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == 'IP':
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # GPU support
        self.gpu_index = None
        if use_gpu:
            try:
                ngpus = faiss.get_num_gpus()
                if ngpus > 0:
                    res = faiss.StandardGpuResources()
                    self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    logger.info(f"Using GPU for vector database (GPU 0 of {ngpus})")
                else:
                    logger.warning("GPU requested but not available. Using CPU.")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}. Using CPU.")
        
        # Metadata storage
        self.id_map = {}  # index -> original_id
        self.metadata_map = {}  # original_id -> metadata
        
        # Statistics
        self.stats = {
            'vectors_added': 0,
            'searches_performed': 0,
            'total_vectors': 0
        }
    
    def add_vectors(self, vectors: np.ndarray, ids: List[int], metadata: List[Dict] = None):
        """Add vectors with IDs and optional metadata"""
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} != {self.dimension}")
        
        # Normalize for cosine similarity if using IP
        if self.index_type == 'IP':
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / (norms + 1e-8)
        
        # Convert to float32 (FAISS requirement)
        vectors = vectors.astype('float32')
        
        # Add to index
        if self.gpu_index:
            self.gpu_index.add(vectors)
        else:
            self.index.add(vectors)
        
        # Map indices to IDs
        start_idx = self.stats['total_vectors']
        for i, vector_id in enumerate(ids):
            idx = start_idx + i
            self.id_map[idx] = vector_id
            if metadata and i < len(metadata):
                self.metadata_map[vector_id] = metadata[i]
        
        # Update stats
        self.stats['vectors_added'] += len(ids)
        self.stats['total_vectors'] = len(self.id_map)
    
    def search(self, query_vector: np.ndarray, k: int = 10, filter: Dict = None) -> List[Tuple]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector (1D or 2D)
            k: Number of results
            filter: Optional metadata filter (not implemented in basic FAISS)
        
        Returns:
            List of tuples: (id, distance, metadata)
        """
        # Ensure 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query_vector.shape[1]} != {self.dimension}")
        
        # Normalize for cosine similarity if using IP
        if self.index_type == 'IP':
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
        
        query_vector = query_vector.astype('float32')
        
        # Search
        if self.gpu_index:
            distances, indices = self.gpu_index.search(query_vector, min(k, self.stats['total_vectors']))
        else:
            distances, indices = self.index.search(query_vector, min(k, self.stats['total_vectors']))
        
        # Convert to results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            original_id = self.id_map.get(idx)
            if original_id is None:
                continue
            
            # Convert distance to similarity (for cosine/IP, higher is better)
            if self.index_type == 'IP':
                similarity = float(dist)  # Already similarity
            else:
                similarity = 1.0 / (1.0 + float(dist))  # Convert distance to similarity
            
            metadata = self.metadata_map.get(original_id, {})
            
            # Apply filter if provided
            if filter:
                if not self._matches_filter(metadata, filter):
                    continue
            
            results.append((original_id, similarity, metadata))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        self.stats['searches_performed'] += 1
        return results[:k]
    
    def _matches_filter(self, metadata: Dict, filter: Dict) -> bool:
        """Check if metadata matches filter (basic implementation)"""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def delete_vectors(self, ids: List[int]):
        """Delete vectors by IDs (note: FAISS doesn't support direct deletion)"""
        # FAISS doesn't support deletion, so we mark as deleted in metadata
        for vector_id in ids:
            if vector_id in self.metadata_map:
                self.metadata_map[vector_id]['_deleted'] = True
        logger.warning("FAISS doesn't support deletion. Vectors marked as deleted in metadata.")
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            **self.stats,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'using_gpu': self.gpu_index is not None,
            'memory_usage': self.index.ntotal * self.dimension * 4  # Approximate (float32)
        }
    
    def save(self, filepath: str):
        """Save index to file"""
        faiss.write_index(self.index, filepath)
        # Save metadata separately
        import json
        metadata_file = filepath + '.metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({
                'id_map': {str(k): v for k, v in self.id_map.items()},
                'metadata_map': {str(k): v for k, v in self.metadata_map.items()},
                'stats': self.stats
            }, f)
        logger.info(f"Saved FAISS index to {filepath}")
    
    def load(self, filepath: str):
        """Load index from file"""
        self.index = faiss.read_index(filepath)
        # Load metadata
        import json
        metadata_file = filepath + '.metadata.json'
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                self.id_map = {int(k): v for k, v in data['id_map'].items()}
                self.metadata_map = {int(k): v for k, v in data['metadata_map'].items()}
                self.stats = data['stats']
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {metadata_file}")
        logger.info(f"Loaded FAISS index from {filepath}")
