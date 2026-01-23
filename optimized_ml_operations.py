"""
Optimized ML Operations
Applying speed optimization strategies to ML Toolbox
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import warnings
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).parent))

# Try to import optimization libraries
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. Install with: pip install numba")

try:
    from multiprocessing import Pool, cpu_count
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    warnings.warn("Multiprocessing not available")


class OptimizedMLOperations:
    """
    Optimized ML Operations using:
    1. NumPy vectorization
    2. Parallel processing
    3. Numba JIT compilation
    4. Efficient data structures
    """
    
    @staticmethod
    def vectorized_similarity_computation(embeddings: np.ndarray, metric: str = 'cosine') -> np.ndarray:
        """
        Vectorized similarity computation
        Replaces Python loops with NumPy operations
        """
        if metric == 'cosine':
            # Normalize embeddings (vectorized)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized = embeddings / norms
            
            # Compute similarity matrix (vectorized matrix multiplication)
            similarity = np.dot(normalized, normalized.T)
            return similarity
        
        elif metric == 'euclidean':
            # Vectorized Euclidean distance
            # Using broadcasting: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b
            dot_product = np.dot(embeddings, embeddings.T)
            squared_norms = np.sum(embeddings ** 2, axis=1)
            distances = np.sqrt(squared_norms[:, np.newaxis] + squared_norms[np.newaxis, :] - 2 * dot_product)
            # Convert distance to similarity (inverse)
            similarity = 1 / (1 + distances)
            return similarity
        
        else:
            # Fallback to cosine
            return OptimizedMLOperations.vectorized_similarity_computation(embeddings, 'cosine')
    
    @staticmethod
    def vectorized_deduplication(embeddings: np.ndarray, threshold: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized deduplication using similarity matrix
        """
        # Compute similarity matrix (vectorized)
        similarity_matrix = OptimizedMLOperations.vectorized_similarity_computation(embeddings)
        
        # Find duplicates (vectorized)
        # Upper triangle to avoid comparing twice
        upper_triangle = np.triu(similarity_matrix, k=1)
        duplicate_mask = upper_triangle > threshold
        
        # Get duplicate indices (vectorized)
        duplicate_indices = np.where(duplicate_mask)[1]  # Column indices (duplicates)
        unique_indices = np.setdiff1d(np.arange(len(embeddings)), duplicate_indices)
        
        return unique_indices, duplicate_indices
    
    @staticmethod
    def parallel_embedding_computation(texts: List[str], embed_func, n_jobs: Optional[int] = None, batch_size: int = 32):
        """
        Parallel embedding computation with batching
        """
        if not PARALLEL_AVAILABLE:
            # Fallback to sequential with batching
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = [embed_func(text) for text in batch]
                results.extend(batch_results)
            return results
        
        if n_jobs is None:
            n_jobs = min(cpu_count(), len(texts) // batch_size + 1)
        
        # Create batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        def process_batch(batch):
            return [embed_func(text) for text in batch]
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            batch_results = list(executor.map(process_batch, batches))
        
        # Flatten results
        results = [item for batch in batch_results for item in batch]
        return results
    
    @staticmethod
    @lru_cache(maxsize=128)
    def cached_embedding(text: str, model_name: str = 'default'):
        """
        Cached embedding computation
        Uses LRU cache to avoid recomputing embeddings
        """
        # This would call the actual embedding function
        # For now, return a placeholder
        return np.random.randn(256)  # Placeholder
    
    @staticmethod
    def vectorized_feature_selection(X: np.ndarray, y: np.ndarray, n_features: int = 10, method: str = 'variance') -> np.ndarray:
        """
        Vectorized feature selection
        """
        if method == 'variance':
            # Compute variance for each feature (vectorized)
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[-n_features:]
            return top_indices
        
        elif method == 'correlation':
            # Compute correlation with target (vectorized)
            if len(y.shape) == 1:
                correlations = np.abs(np.corrcoef(X.T, y)[:-1, -1])
            else:
                # Multi-output: use mean correlation
                correlations = np.mean([np.abs(np.corrcoef(X.T, y[:, i])[:-1, -1]) 
                                       for i in range(y.shape[1])], axis=0)
            top_indices = np.argsort(correlations)[-n_features:]
            return top_indices
        
        elif method == 'combined':
            # Combine variance and correlation (vectorized)
            variances = np.var(X, axis=0)
            if len(y.shape) == 1:
                correlations = np.abs(np.corrcoef(X.T, y)[:-1, -1])
            else:
                correlations = np.mean([np.abs(np.corrcoef(X.T, y[:, i])[:-1, -1]) 
                                       for i in range(y.shape[1])], axis=0)
            scores = variances * correlations
            top_indices = np.argsort(scores)[-n_features:]
            return top_indices
        
        else:
            # Default: variance
            return OptimizedMLOperations.vectorized_feature_selection(X, y, n_features, 'variance')
    
    @staticmethod
    def vectorized_preprocessing(X: np.ndarray, operations: List[str] = ['normalize', 'standardize']) -> np.ndarray:
        """
        Vectorized preprocessing pipeline
        """
        result = X.copy()
        
        for op in operations:
            if op == 'normalize':
                # Min-max normalization (vectorized)
                min_vals = np.min(result, axis=0, keepdims=True)
                max_vals = np.max(result, axis=0, keepdims=True)
                ranges = max_vals - min_vals
                ranges = np.where(ranges == 0, 1, ranges)  # Avoid division by zero
                result = (result - min_vals) / ranges
            
            elif op == 'standardize':
                # Z-score standardization (vectorized)
                mean_vals = np.mean(result, axis=0, keepdims=True)
                std_vals = np.std(result, axis=0, keepdims=True)
                std_vals = np.where(std_vals == 0, 1, std_vals)  # Avoid division by zero
                result = (result - mean_vals) / std_vals
            
            elif op == 'scale':
                # Scale to unit norm (vectorized)
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                result = result / norms
        
        return result


# Numba-optimized functions (if available)
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def numba_distance_matrix(X: np.ndarray) -> np.ndarray:
        """Numba-optimized distance matrix computation"""
        n = X.shape[0]
        distances = np.zeros((n, n))
        
        for i in prange(n):
            for j in prange(i + 1, n):
                dist = 0.0
                for k in range(X.shape[1]):
                    diff = X[i, k] - X[j, k]
                    dist += diff * diff
                distances[i, j] = np.sqrt(dist)
                distances[j, i] = distances[i, j]
        
        return distances
    
    @jit(nopython=True)
    def numba_softmax(x: np.ndarray) -> np.ndarray:
        """Numba-optimized softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
else:
    # Fallback implementations
    def numba_distance_matrix(X: np.ndarray) -> np.ndarray:
        """Fallback: NumPy implementation"""
        return OptimizedMLOperations.vectorized_similarity_computation(X, 'euclidean')
    
    def numba_softmax(x: np.ndarray) -> np.ndarray:
        """Fallback: NumPy implementation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


# Example usage
if __name__ == '__main__':
    print("Optimized ML Operations")
    print("="*80)
    
    # Test vectorized similarity
    print("\n1. Testing vectorized similarity computation...")
    embeddings = np.random.randn(100, 256)
    
    import time
    start = time.time()
    similarity = OptimizedMLOperations.vectorized_similarity_computation(embeddings)
    vectorized_time = time.time() - start
    print(f"Vectorized similarity: {vectorized_time:.4f}s for {len(embeddings)}x{len(embeddings)} matrix")
    
    # Test vectorized deduplication
    print("\n2. Testing vectorized deduplication...")
    start = time.time()
    unique_indices, duplicate_indices = OptimizedMLOperations.vectorized_deduplication(embeddings, threshold=0.9)
    dedup_time = time.time() - start
    print(f"Vectorized deduplication: {dedup_time:.4f}s")
    print(f"Unique: {len(unique_indices)}, Duplicates: {len(duplicate_indices)}")
    
    # Test vectorized feature selection
    print("\n3. Testing vectorized feature selection...")
    X = np.random.randn(1000, 100)
    y = np.random.randn(1000)
    
    start = time.time()
    selected_features = OptimizedMLOperations.vectorized_feature_selection(X, y, n_features=10)
    selection_time = time.time() - start
    print(f"Vectorized feature selection: {selection_time:.4f}s")
    print(f"Selected features: {selected_features}")
    
    print("\n✅ Optimized ML Operations ready!")
