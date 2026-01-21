"""
ML Toolbox Optimizations
Performance improvements, caching, and Big O optimizations
"""
import functools
import time
from typing import Dict, Any, Optional, Callable, List
from collections import OrderedDict
import hashlib
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np


class LRUCache:
    """
    Least Recently Used (LRU) Cache
    O(1) get and put operations
    
    Big O:
    - get(): O(1)
    - put(): O(1)
    - evict(): O(1)
    """
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache - O(1)"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache - O(1)"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                # Add new
                if len(self.cache) >= self.maxsize:
                    # Evict least recently used
                    self.cache.popitem(last=False)
                self.cache[key] = value
    
    def clear(self) -> None:
        """Clear cache - O(n)"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get cache size - O(1)"""
        return len(self.cache)


class PerformanceMonitor:
    """
    Performance monitoring and profiling
    
    Tracks:
    - Function execution times
    - Memory usage
    - Call counts
    - Big O analysis
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.call_counts: Dict[str, int] = {}
        self.lock = threading.Lock()
    
    def time_function(self, func_name: str):
        """Decorator to time function execution"""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                
                with self.lock:
                    if func_name not in self.metrics:
                        self.metrics[func_name] = []
                    self.metrics[func_name].append(execution_time)
                    self.call_counts[func_name] = self.call_counts.get(func_name, 0) + 1
                
                return result
            return wrapper
        return decorator
    
    def get_stats(self, func_name: str) -> Dict[str, Any]:
        """Get performance statistics for a function"""
        if func_name not in self.metrics:
            return {}
        
        times = self.metrics[func_name]
        return {
            'calls': self.call_counts[func_name],
            'total_time': sum(times),
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'median_time': np.median(times) if times else 0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all monitored functions"""
        return {name: self.get_stats(name) for name in self.metrics.keys()}


def cache_result(maxsize: int = 128, ttl: Optional[float] = None):
    """
    Cache function results with optional TTL
    
    Big O:
    - Cache hit: O(1)
    - Cache miss: O(f(n)) where f(n) is function complexity
    """
    def decorator(func: Callable):
        cache = LRUCache(maxsize=maxsize)
        cache_times: Dict[str, float] = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            cache_key = _create_cache_key(func.__name__, args, kwargs)
            
            # Check cache
            if ttl is None:
                result = cache.get(cache_key)
                if result is not None:
                    return result
            else:
                # Check TTL
                if cache_key in cache_times:
                    if time.time() - cache_times[cache_key] < ttl:
                        result = cache.get(cache_key)
                        if result is not None:
                            return result
                    else:
                        # Expired
                        cache_times.pop(cache_key, None)
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(cache_key, result)
            if ttl:
                cache_times[cache_key] = time.time()
            
            return result
        
        wrapper.cache_clear = cache.clear
        wrapper.cache_info = lambda: {'size': cache.size(), 'maxsize': cache.maxsize}
        
        return wrapper
    return decorator


def _create_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create cache key from function name and arguments"""
    # Use hash for efficiency
    key_data = (func_name, args, tuple(sorted(kwargs.items())))
    return hashlib.md5(pickle.dumps(key_data)).hexdigest()


class ParallelProcessor:
    """
    Parallel processing utilities
    
    Big O improvements:
    - Sequential: O(n * f(n))
    - Parallel: O(n/p * f(n)) where p is number of processes
    """
    
    def __init__(self, n_workers: Optional[int] = None, use_processes: bool = False):
        self.n_workers = n_workers
        self.use_processes = use_processes
        self.executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    def process_batch(self, items: List[Any], func: Callable, chunk_size: Optional[int] = None) -> List[Any]:
        """
        Process items in parallel
        
        Big O:
        - Sequential: O(n * f(n))
        - Parallel: O(n/p * f(n)) where p is workers
        """
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.n_workers or 4))
        
        with self.executor_class(max_workers=self.n_workers) as executor:
            # Process in chunks
            chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
            results = list(executor.map(func, chunks))
        
        # Flatten results
        return [item for sublist in results for item in sublist]
    
    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """
        Map function over items in parallel
        
        Big O: O(n/p * f(n))
        """
        with self.executor_class(max_workers=self.n_workers) as executor:
            return list(executor.map(func, items))


class MemoryOptimizer:
    """
    Memory optimization utilities
    """
    
    @staticmethod
    def optimize_array(array: np.ndarray) -> np.ndarray:
        """
        Optimize numpy array memory usage
        
        Big O: O(n) - linear scan
        """
        # Use smallest dtype that can hold values
        if array.dtype == np.float64:
            if np.all(array == array.astype(np.float32)):
                return array.astype(np.float32)
        elif array.dtype == np.int64:
            if np.all(array >= np.iinfo(np.int32).min) and np.all(array <= np.iinfo(np.int32).max):
                return array.astype(np.int32)
        
        return array
    
    @staticmethod
    def clear_memory():
        """Clear memory caches"""
        import gc
        gc.collect()


class BigOAnalyzer:
    """
    Big O notation analysis and documentation
    """
    
    @staticmethod
    def analyze_operation(operation: str, n: int, measured_time: float) -> Dict[str, Any]:
        """
        Analyze operation complexity
        
        Returns:
            Dictionary with complexity analysis
        """
        # Estimate complexity based on time
        if measured_time < 0.001:
            complexity = "O(1) or O(log n)"
        elif measured_time < n * 0.001:
            complexity = "O(n)"
        elif measured_time < n * np.log(n) * 0.001:
            complexity = "O(n log n)"
        elif measured_time < n * n * 0.001:
            complexity = "O(n²)"
        else:
            complexity = "O(n³) or worse"
        
        return {
            'operation': operation,
            'n': n,
            'time': measured_time,
            'estimated_complexity': complexity,
            'time_per_item': measured_time / n if n > 0 else 0
        }


# Global instances
_global_cache = LRUCache(maxsize=256)
_global_monitor = PerformanceMonitor()
_global_parallel = ParallelProcessor()


def get_global_cache() -> LRUCache:
    """Get global cache instance"""
    return _global_cache


def get_global_monitor() -> PerformanceMonitor:
    """Get global performance monitor"""
    return _global_monitor


def get_global_parallel() -> ParallelProcessor:
    """Get global parallel processor"""
    return _global_parallel
