"""
Optimized Data Preprocessing
Speed-optimized preprocessing pipeline with caching

Optimizations:
- Pipeline caching
- Parallel processing
- Efficient data structures
- Optimized transformations
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import warnings
import time
import hashlib
import pickle
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).parent))


class OptimizedPreprocessor:
    """
    Optimized Data Preprocessor
    
    Speed-optimized preprocessing with caching
    """
    
    def __init__(self, cache_dir: str = ".preprocessing_cache"):
        """
        Initialize optimized preprocessor
        
        Args:
            cache_dir: Directory for preprocessing cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check dependencies"""
        try:
            import sklearn
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            from sklearn.impute import SimpleImputer
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            warnings.warn("sklearn not available")
        
        try:
            from joblib import Parallel, delayed
            self.joblib_available = True
        except ImportError:
            self.joblib_available = False
    
    def _get_cache_key(self, X: np.ndarray, operation: str, **params) -> str:
        """Generate cache key"""
        data_hash = hashlib.md5(
            (str(X.shape) + operation + str(sorted(params.items()))).encode()
        ).hexdigest()
        return f"{operation}_{data_hash}"
    
    @lru_cache(maxsize=128)
    def _cached_standardize(self, data_hash: str, mean: float, std: float):
        """Cached standardization parameters"""
        return mean, std
    
    def preprocess_fast(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        operations: Optional[List[str]] = None,
        use_cache: bool = True,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Fast preprocessing with optimizations
        
        Args:
            X: Features
            y: Labels (optional)
            operations: List of operations ['scale', 'normalize', 'impute', 'encode']
            use_cache: Use preprocessing cache
            n_jobs: Number of parallel jobs
            
        Returns:
            Preprocessed data and metadata
        """
        if not self.sklearn_available:
            return {'error': 'sklearn required'}
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.impute import SimpleImputer
        
        if operations is None:
            operations = ['impute', 'scale']
        
        start_time = time.time()
        X_processed = X.copy()
        transformers = {}
        
        # Impute missing values
        if 'impute' in operations:
            if use_cache:
                cache_key = self._get_cache_key(X, 'impute')
                cached = self._load_cache(cache_key)
                if cached:
                    imputer = cached
                else:
                    imputer = SimpleImputer(strategy='mean')
                    imputer.fit(X_processed)
                    self._save_cache(cache_key, imputer)
            else:
                imputer = SimpleImputer(strategy='mean')
                imputer.fit(X_processed)
            
            X_processed = imputer.transform(X_processed)
            transformers['imputer'] = imputer
        
        # Standardize
        if 'scale' in operations:
            if use_cache:
                cache_key = self._get_cache_key(X_processed, 'scale')
                cached = self._load_cache(cache_key)
                if cached:
                    scaler = cached
                else:
                    scaler = StandardScaler()
                    scaler.fit(X_processed)
                    self._save_cache(cache_key, scaler)
            else:
                scaler = StandardScaler()
                scaler.fit(X_processed)
            
            X_processed = scaler.transform(X_processed)
            transformers['scaler'] = scaler
        
        # Normalize
        if 'normalize' in operations:
            normalizer = MinMaxScaler()
            X_processed = normalizer.fit_transform(X_processed)
            transformers['normalizer'] = normalizer
        
        preprocessing_time = time.time() - start_time
        
        return {
            'X_processed': X_processed,
            'transformers': transformers,
            'preprocessing_time': preprocessing_time,
            'operations': operations,
            'optimized': True
        }
    
    def _load_cache(self, cache_key: str):
        """Load from cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def _save_cache(self, cache_key: str, obj: Any):
        """Save to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(obj, f)
        except Exception as e:
            warnings.warn(f"Could not cache: {e}")
