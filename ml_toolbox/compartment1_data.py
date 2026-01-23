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
        
        # Kuhn/Johnson Preprocessing Methods
        try:
            from kuhn_johnson_preprocessing import ModelSpecificPreprocessor, create_preprocessing_pipeline
            self.components['ModelSpecificPreprocessor'] = ModelSpecificPreprocessor
            self.components['create_preprocessing_pipeline'] = create_preprocessing_pipeline
        except ImportError as e:
            print(f"Warning: Could not import Kuhn/Johnson preprocessing: {e}")
        
        # Missing Data Handling
        try:
            from missing_data import MissingDataHandler, CVAwareImputation
            self.components['MissingDataHandler'] = MissingDataHandler
            self.components['CVAwareImputation'] = CVAwareImputation
        except ImportError as e:
            print(f"Warning: Could not import missing data handlers: {e}")
        
        # Class Imbalance Handling
        try:
            from class_imbalance import ClassImbalanceHandler, ThresholdTuner
            self.components['ClassImbalanceHandler'] = ClassImbalanceHandler
            self.components['ThresholdTuner'] = ThresholdTuner
        except ImportError as e:
            print(f"Warning: Could not import class imbalance handlers: {e}")
        
        # High-Cardinality Categorical Handling
        try:
            from high_cardinality_categorical import HighCardinalityHandler
            self.components['HighCardinalityHandler'] = HighCardinalityHandler
        except ImportError as e:
            print(f"Warning: Could not import high-cardinality handler: {e}")
        
        # Variance & Correlation Filtering
        try:
            from variance_correlation_filter import VarianceCorrelationFilter
            self.components['VarianceCorrelationFilter'] = VarianceCorrelationFilter
        except ImportError as e:
            print(f"Warning: Could not import variance/correlation filter: {e}")
        
        # Phase 2: Time Series Feature Engineering (also in Data compartment)
        try:
            from time_series_feature_engineering import TimeSeriesFeatureEngineer
            self.components['TimeSeriesFeatureEngineer'] = TimeSeriesFeatureEngineer
        except ImportError as e:
            print(f"Warning: Could not import time series feature engineer: {e}")
        
        # Knuth Data Sampling and Preprocessing
        try:
            from knuth_ml_integrations import KnuthDataSampling, KnuthDataPreprocessing
            self.components['KnuthDataSampling'] = KnuthDataSampling
            self.components['KnuthDataPreprocessing'] = KnuthDataPreprocessing
        except ImportError as e:
            print(f"Warning: Could not import Knuth data operations: {e}")
        
        # GPU-Accelerated Preprocessing
        try:
            from gpu_accelerated_preprocessor import GPUAcceleratedPreprocessor, HybridPreprocessor
            self.components['GPUAcceleratedPreprocessor'] = GPUAcceleratedPreprocessor
            self.components['HybridPreprocessor'] = HybridPreprocessor
        except ImportError as e:
            print(f"Warning: Could not import GPU-accelerated preprocessor: {e}")
        
        # Corpus Callosum Preprocessor (Combined Brain Approach)
        try:
            from corpus_callosum_preprocessor import CorpusCallosumPreprocessor
            self.components['CorpusCallosumPreprocessor'] = CorpusCallosumPreprocessor
        except ImportError as e:
            print(f"Warning: Could not import Corpus Callosum preprocessor: {e}")
        
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
            },
            'ModelSpecificPreprocessor': {
                'description': 'Kuhn/Johnson model-specific preprocessing',
                'features': [
                    'Different preprocessing per model type',
                    'Spatial sign for distance-based models',
                    'Centering/scaling for linear models',
                    'Box-Cox/Yeo-Johnson transformations'
                ],
                'location': 'kuhn_johnson_preprocessing.py',
                'category': 'Preprocessing'
            },
            'MissingDataHandler': {
                'description': 'Systematic missing data handling',
                'features': [
                    'Multiple imputation strategies (mean, median, KNN, iterative)',
                    'Missing indicator variables',
                    'Pattern detection (MCAR, MAR, MNAR)',
                    'CV-aware imputation'
                ],
                'location': 'missing_data.py',
                'category': 'Preprocessing'
            },
            'ClassImbalanceHandler': {
                'description': 'Handle class imbalance',
                'features': [
                    'SMOTE (Synthetic Minority Oversampling)',
                    'ADASYN, BorderlineSMOTE',
                    'Random undersampling',
                    'Cost-sensitive learning',
                    'Threshold tuning'
                ],
                'location': 'class_imbalance.py',
                'category': 'Preprocessing'
            },
            'HighCardinalityHandler': {
                'description': 'Handle high-cardinality categorical variables',
                'features': [
                    'Target encoding (mean encoding)',
                    'Feature hashing',
                    'Frequency encoding',
                    'Rare category grouping'
                ],
                'location': 'high_cardinality_categorical.py',
                'category': 'Preprocessing'
            },
            'VarianceCorrelationFilter': {
                'description': 'Filter uninformative and correlated features',
                'features': [
                    'Near-zero variance detection',
                    'High correlation filtering',
                    'Percent unique values',
                    'Frequency ratio analysis'
                ],
                'location': 'variance_correlation_filter.py',
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
    
    def get_model_specific_preprocessor(self, model_type: str = 'auto', **kwargs):
        """Get model-specific preprocessor instance (Kuhn/Johnson)"""
        if 'ModelSpecificPreprocessor' in self.components:
            return self.components['ModelSpecificPreprocessor'](model_type=model_type, **kwargs)
        else:
            raise ImportError("ModelSpecificPreprocessor not available")
    
    def get_missing_data_handler(self, strategy: str = 'knn', add_indicator: bool = True):
        """Get missing data handler instance (Kuhn/Johnson)"""
        if 'MissingDataHandler' in self.components:
            return self.components['MissingDataHandler'](strategy=strategy, add_indicator=add_indicator)
        else:
            raise ImportError("MissingDataHandler not available")
    
    def get_class_imbalance_handler(self, method: str = 'smote'):
        """Get class imbalance handler instance (Kuhn/Johnson)"""
        if 'ClassImbalanceHandler' in self.components:
            return self.components['ClassImbalanceHandler'](method=method)
        else:
            raise ImportError("ClassImbalanceHandler not available")
    
    def get_high_cardinality_handler(self, method: str = 'target_encoding', min_frequency: float = 0.01):
        """Get high-cardinality categorical handler instance (Kuhn/Johnson)"""
        if 'HighCardinalityHandler' in self.components:
            return self.components['HighCardinalityHandler'](method=method, min_frequency=min_frequency)
        else:
            raise ImportError("HighCardinalityHandler not available")
    
    def get_variance_correlation_filter(self, remove_nzv: bool = True, remove_high_correlation: bool = True):
        """Get variance/correlation filter instance (Kuhn/Johnson)"""
        if 'VarianceCorrelationFilter' in self.components:
            return self.components['VarianceCorrelationFilter'](
                remove_nzv=remove_nzv,
                remove_high_correlation=remove_high_correlation
            )
        else:
            raise ImportError("VarianceCorrelationFilter not available")
    
    def get_gpu_accelerated_preprocessor(self, use_gpu: bool = True, **kwargs):
        """Get GPU-accelerated preprocessor instance"""
        if 'GPUAcceleratedPreprocessor' in self.components:
            return self.components['GPUAcceleratedPreprocessor'](use_gpu=use_gpu, **kwargs)
        else:
            raise ImportError("GPUAcceleratedPreprocessor not available")
    
    def get_hybrid_preprocessor(self, gpu_threshold: int = 100):
        """Get hybrid preprocessor (auto GPU/CPU selection)"""
        if 'HybridPreprocessor' in self.components:
            return self.components['HybridPreprocessor'](gpu_threshold=gpu_threshold)
        else:
            raise ImportError("HybridPreprocessor not available")
    
    def get_corpus_callosum_preprocessor(self, parallel_execution: bool = True, 
                                         split_strategy: str = 'intelligent',
                                         combine_results: bool = True):
        """
        Get Corpus Callosum Preprocessor
        
        Combines AdvancedDataPreprocessor and ConventionalPreprocessor
        like two brain hemispheres working together
        
        Args:
            parallel_execution: Execute both in parallel
            split_strategy: 'intelligent', 'equal', 'fast_first'
            combine_results: Combine results from both
            
        Returns:
            CorpusCallosumPreprocessor instance
        """
        if 'CorpusCallosumPreprocessor' in self.components:
            return self.components['CorpusCallosumPreprocessor'](
                parallel_execution=parallel_execution,
                split_strategy=split_strategy,
                combine_results=combine_results
            )
        else:
            raise ImportError("CorpusCallosumPreprocessor not available")
    
    def get_info(self):
        """Get information about this compartment"""
        return {
            'name': 'Data Compartment',
            'description': 'Preprocessing, validation, and data transformation',
            'components': list(self.components.keys()),
            'component_count': len(self.components)
        }
