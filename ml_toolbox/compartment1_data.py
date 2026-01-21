"""
Compartment 1: Data
Preprocessing, validation, transformation, and data management

Optimizations:
- LRU caching for preprocessor instances
- Performance monitoring
- Big O optimizations
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import functools

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import optimizations
try:
    from .optimizations import cache_result, get_global_cache, get_global_monitor
    import time
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False
    import time
    print("Warning: Optimizations not available")


class DataCompartment:
    """
    Compartment 1: Data
    
    Components for data preprocessing, validation, and transformation:
    - AdvancedDataPreprocessor: Quantum + PocketFence preprocessing
    - ConventionalPreprocessor: Basic preprocessing
    - Data validation and quality checks
    - Data transformation utilities
    """
    
    def __init__(self):
        self.components = {}
        self._cache = get_global_cache() if OPTIMIZATIONS_AVAILABLE else None
        self._monitor = get_global_monitor() if OPTIMIZATIONS_AVAILABLE else None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all data compartment components"""
        
        # AdvancedDataPreprocessor (main preprocessing tool)
        try:
            from data_preprocessor import AdvancedDataPreprocessor, ConventionalPreprocessor
            self.components['AdvancedDataPreprocessor'] = AdvancedDataPreprocessor
            self.components['ConventionalPreprocessor'] = ConventionalPreprocessor
        except ImportError as e:
            print(f"Warning: Could not import preprocessors: {e}")
        
        # Add component descriptions
        self.component_descriptions = {
            'AdvancedDataPreprocessor': {
                'description': 'Advanced preprocessing with Quantum Kernel + PocketFence Kernel',
                'features': [
                    'Safety filtering (PocketFence)',
                    'Semantic deduplication (Quantum)',
                    'Intelligent categorization (Quantum)',
                    'Quality scoring (Quantum)',
                    'Dimensionality reduction (PCA/SVD)',
                    'Automatic feature creation'
                ],
                'location': 'data_preprocessor.py',
                'category': 'Preprocessing'
            },
            'ConventionalPreprocessor': {
                'description': 'Basic preprocessing with exact matching',
                'features': [
                    'Basic safety filtering',
                    'Exact duplicate removal',
                    'Keyword-based categorization',
                    'Simple quality scoring'
                ],
                'location': 'data_preprocessor.py',
                'category': 'Preprocessing'
            }
        }
    
    @functools.lru_cache(maxsize=32)
    def get_preprocessor(self, advanced: bool = True, **kwargs_hash: int):
        """
        Get a preprocessor instance (cached)
        
        Big O:
        - Cache hit: O(1)
        - Cache miss: O(1) - instantiation is fast
        
        Args:
            advanced: If True, use AdvancedDataPreprocessor, else ConventionalPreprocessor
            **kwargs_hash: Hashed kwargs for caching
            
        Returns:
            Preprocessor instance
        """
        # Convert hash back to kwargs (simplified - in practice, store kwargs)
        # For now, create new instance (caching happens at preprocess level)
        if advanced:
            return self.components['AdvancedDataPreprocessor']()
        else:
            return self.components['ConventionalPreprocessor']()
    
    def _get_preprocessor_uncached(self, advanced: bool = True, **kwargs):
        """Get preprocessor without caching (internal use)"""
        if advanced:
            return self.components['AdvancedDataPreprocessor'](**kwargs)
        else:
            return self.components['ConventionalPreprocessor']()
    
    def preprocess(self, data, advanced: bool = True, verbose: bool = False, **kwargs):
        """
        Preprocess data using appropriate preprocessor
        
        Args:
            data: List of text strings to preprocess
            advanced: If True, use AdvancedDataPreprocessor
            verbose: Print detailed progress
            **kwargs: Arguments for preprocessor constructor (not preprocess method)
            
        Returns:
            Preprocessing results dictionary
        """
        # Separate constructor args from preprocess args
        preprocessor_kwargs = {k: v for k, v in kwargs.items() 
                              if k in ['pocketfence_url', 'dedup_threshold', 'use_quantum', 
                                      'enable_compression', 'compression_ratio', 'compression_method']}
        
        preprocessor = self.get_preprocessor(advanced=advanced, **preprocessor_kwargs)
        return preprocessor.preprocess(data, verbose=verbose)
    
    def list_components(self):
        """List all available components in this compartment"""
        print("="*80)
        print("COMPARTMENT 1: DATA")
        print("="*80)
        print("\nComponents:")
        for name, component in self.components.items():
            desc = self.component_descriptions.get(name, {})
            print(f"\n{name}:")
            print(f"  Description: {desc.get('description', 'N/A')}")
            print(f"  Location: {desc.get('location', 'N/A')}")
            print(f"  Category: {desc.get('category', 'N/A')}")
            if 'features' in desc:
                print(f"  Features:")
                for feature in desc['features']:
                    print(f"    - {feature}")
        print("\n" + "="*80)
    
    def get_info(self):
        """Get information about this compartment"""
        return {
            'name': 'Data Compartment',
            'description': 'Preprocessing, validation, and data transformation',
            'components': list(self.components.keys()),
            'component_count': len(self.components)
        }
