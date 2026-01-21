"""
Base Vector Database Interface
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np


class VectorDatabase(ABC):
    """Base interface for vector databases"""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, ids: List[int], metadata: List[Dict] = None):
        """Add vectors with IDs and optional metadata"""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10, filter: Dict = None) -> List[Tuple]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[int]):
        """Delete vectors by IDs"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        """Get database statistics"""
        pass
