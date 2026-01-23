"""
Speed Optimization Implementation Plan
Applying core strategies to improve ML Toolbox performance
"""
import sys
from pathlib import Path
import time
import cProfile
import pstats
from io import StringIO
import numpy as np
from typing import List, Dict, Any, Optional
import warnings

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
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False

try:
    import concurrent.futures
    CONCURRENT_AVAILABLE = True
except ImportError:
    CONCURRENT_AVAILABLE = False


class SpeedOptimizer:
    """
    Speed Optimization Framework
    Applies core strategies for performance improvement
    """
    
    def __init__(self):
        self.profiling_results = {}
        self.optimization_applied = []
    
    def profile_function(self, func, *args, **kwargs):
        """Profile a function to identify bottlenecks"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 slowest functions
        
        return result, s.getvalue()
    
    def vectorize_operation(self, data: np.ndarray, operation: str = 'mean'):
        """Use NumPy vectorization instead of Python loops"""
        if operation == 'mean':
            return np.mean(data, axis=0)
        elif operation == 'sum':
            return np.sum(data, axis=0)
        elif operation == 'std':
            return np.std(data, axis=0)
        elif operation == 'max':
            return np.max(data, axis=0)
        elif operation == 'min':
            return np.min(data, axis=0)
        else:
            return data
    
    def parallel_process(self, data: List, func, n_jobs: Optional[int] = None):
        """Parallel processing using multiprocessing"""
        if not MULTIPROCESSING_AVAILABLE:
            return [func(item) for item in data]
        
        if n_jobs is None:
            n_jobs = cpu_count()
        
        with Pool(n_jobs) as pool:
            results = pool.map(func, data)
        
        return results
    
    def optimize_with_numba(self, func):
        """Apply Numba JIT compilation to function"""
        if NUMBA_AVAILABLE:
            return jit(nopython=True, parallel=True)(func)
        else:
            return func


# Optimized vectorized operations
def vectorized_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Vectorized similarity matrix computation
    Replaces Python loops with NumPy operations
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    
    # Compute similarity matrix using matrix multiplication
    similarity = np.dot(normalized, normalized.T)
    
    return similarity


def vectorized_feature_selection(X: np.ndarray, y: np.ndarray, n_features: int = 10) -> np.ndarray:
    """
    Vectorized feature selection using variance and correlation
    """
    # Compute variance for each feature (vectorized)
    variances = np.var(X, axis=0)
    
    # Compute correlation with target (vectorized)
    if len(y.shape) == 1:
        correlations = np.abs(np.corrcoef(X.T, y)[:-1, -1])
    else:
        # Multi-output: use mean correlation
        correlations = np.mean([np.abs(np.corrcoef(X.T, y[:, i])[:-1, -1]) 
                               for i in range(y.shape[1])], axis=0)
    
    # Combine scores (vectorized)
    scores = variances * correlations
    
    # Select top features (vectorized)
    top_indices = np.argsort(scores)[-n_features:]
    
    return top_indices


@jit(nopython=True, parallel=True) if NUMBA_AVAILABLE else lambda x: x
def numba_optimized_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Numba-optimized distance matrix computation
    """
    n = X.shape[0]
    distances = np.zeros((n, n))
    
    for i in prange(n):
        for j in prange(i + 1, n):
            dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def optimized_batch_processing(data: List, batch_size: int = 32, func=None):
    """
    Optimized batch processing with vectorization
    """
    if func is None:
        func = lambda x: x
    
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        
        # Convert to numpy array if possible
        if isinstance(batch[0], (list, np.ndarray)):
            batch_array = np.array(batch)
            batch_result = func(batch_array)
        else:
            batch_result = [func(item) for item in batch]
        
        results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
    
    return results


# Example usage and testing
if __name__ == '__main__':
    optimizer = SpeedOptimizer()
    
    # Test vectorized operations
    print("Testing vectorized operations...")
    test_data = np.random.randn(1000, 100)
    
    start = time.time()
    mean_result = optimizer.vectorize_operation(test_data, 'mean')
    vectorized_time = time.time() - start
    print(f"Vectorized mean: {vectorized_time:.4f}s")
    
    # Test parallel processing
    if MULTIPROCESSING_AVAILABLE:
        print("\nTesting parallel processing...")
        test_list = list(range(1000))
        
        def square(x):
            return x ** 2
        
        start = time.time()
        parallel_result = optimizer.parallel_process(test_list, square, n_jobs=4)
        parallel_time = time.time() - start
        print(f"Parallel processing: {parallel_time:.4f}s")
        
        start = time.time()
        sequential_result = [square(x) for x in test_list]
        sequential_time = time.time() - start
        print(f"Sequential processing: {sequential_time:.4f}s")
        print(f"Speedup: {sequential_time / parallel_time:.2f}x")
    
    print("\nSpeed optimization framework ready!")
