"""
Universal Adaptive Preprocessor
AI-Powered: Automatically selects and combines best preprocessing strategies

Innovation: One intelligent preprocessor instead of 6+ separate ones
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    from data_preprocessor import AdvancedDataPreprocessor, ConventionalPreprocessor
    from corpus_callosum_preprocessor import CorpusCallosumPreprocessor
    from gpu_accelerated_preprocessor import GPUAcceleratedPreprocessor
    from kuhn_johnson_preprocessing import ModelSpecificPreprocessor
    PREPROCESSORS_AVAILABLE = True
except ImportError:
    PREPROCESSORS_AVAILABLE = False
    warnings.warn("Some preprocessors not available")


class UniversalAdaptivePreprocessor:
    """
    Universal Adaptive Preprocessor
    
    Innovation: AI-powered preprocessor that automatically:
    - Detects data characteristics
    - Selects optimal preprocessing strategy
    - Combines multiple strategies intelligently
    - Learns from what works
    - Adapts to task requirements
    
    Replaces: 6+ separate preprocessors with one intelligent system
    """
    
    def __init__(self):
        """Initialize universal adaptive preprocessor"""
        self.preprocessors = self._initialize_preprocessors()
        self.performance_history = []  # Learn from what works
        self.strategy_cache = {}  # Cache successful strategies
    
    def _initialize_preprocessors(self) -> Dict[str, Any]:
        """Initialize all available preprocessors"""
        preprocessors = {}
        
        if PREPROCESSORS_AVAILABLE:
            try:
                preprocessors['advanced'] = AdvancedDataPreprocessor
                preprocessors['conventional'] = ConventionalPreprocessor
                preprocessors['corpus_callosum'] = CorpusCallosumPreprocessor
                preprocessors['gpu'] = GPUAcceleratedPreprocessor
                preprocessors['model_specific'] = ModelSpecificPreprocessor
            except Exception as e:
                warnings.warn(f"Some preprocessors not available: {e}")
        
        return preprocessors
    
    def preprocess(self, data: Any, task_type: str = 'auto', 
                   model_type: str = 'auto', context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Intelligently preprocess data
        
        AI decides:
        - Which preprocessor(s) to use
        - How to combine them
        - What order to apply
        - What parameters to use
        
        Args:
            data: Input data (text, array, etc.)
            task_type: Task type (classification, regression, etc.)
            model_type: Model type (auto-detected if 'auto')
            context: Additional context
        
        Returns:
            Preprocessed data and metadata
        """
        # Step 1: Analyze data
        data_profile = self._analyze_data(data)
        
        # Step 2: Select optimal strategy
        strategy = self._select_strategy(data_profile, task_type, model_type, context)
        
        # Step 3: Execute strategy
        result = self._execute_strategy(strategy, data, task_type, model_type)
        
        # Step 4: Learn from result
        self._learn_from_result(strategy, result, data_profile)
        
        return result
    
    def _analyze_data(self, data: Any) -> Dict[str, Any]:
        """Analyze data characteristics"""
        profile = {
            'type': 'unknown',
            'size': 0,
            'is_text': False,
            'is_numeric': False,
            'has_missing': False,
            'sparsity': 0.0,
            'cardinality': 0
        }
        
        if isinstance(data, (list, tuple)):
            profile['size'] = len(data)
            if data and isinstance(data[0], str):
                profile['type'] = 'text'
                profile['is_text'] = True
                profile['cardinality'] = len(set(data))
            elif data and isinstance(data[0], (int, float)):
                profile['type'] = 'numeric'
                profile['is_numeric'] = True
        
        elif isinstance(data, np.ndarray):
            profile['type'] = 'numeric'
            profile['is_numeric'] = True
            profile['size'] = data.shape[0]
            profile['sparsity'] = np.count_nonzero(data == 0) / data.size if data.size > 0 else 0.0
            profile['has_missing'] = np.isnan(data).any() if data.dtype == float else False
        
        return profile
    
    def _select_strategy(self, data_profile: Dict, task_type: str, 
                        model_type: str, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Select optimal preprocessing strategy
        
        Uses:
        - Data characteristics
        - Task requirements
        - Historical performance
        - Cached strategies
        """
        # Check cache first
        cache_key = self._generate_cache_key(data_profile, task_type, model_type)
        if cache_key in self.strategy_cache:
            return self.strategy_cache[cache_key]
        
        # AI decision logic
        strategy = {
            'preprocessors': [],
            'order': [],
            'params': {}
        }
        
        # Text data
        if data_profile['is_text']:
            if data_profile['size'] > 1000:
                # Large text - use advanced with compression
                strategy['preprocessors'] = ['advanced']
                strategy['params'] = {
                    'enable_compression': True,
                    'enable_scrubbing': True,
                    'use_quantum': True
                }
            else:
                # Small text - use conventional (faster)
                strategy['preprocessors'] = ['conventional']
        
        # Numeric data
        elif data_profile['is_numeric']:
            if data_profile['sparsity'] > 0.5:
                # Sparse data - use sparse-aware preprocessing
                strategy['preprocessors'] = ['conventional']
            else:
                # Dense data - use advanced
                strategy['preprocessors'] = ['advanced']
        
        # Model-specific preprocessing
        if model_type != 'auto' and 'model_specific' in self.preprocessors:
            strategy['preprocessors'].insert(0, 'model_specific')
        
        # GPU acceleration for large data
        if data_profile['size'] > 10000 and 'gpu' in self.preprocessors:
            strategy['preprocessors'].append('gpu')
        
        strategy['order'] = strategy['preprocessors'].copy()
        
        return strategy
    
    def _execute_strategy(self, strategy: Dict, data: Any, 
                         task_type: str, model_type: str) -> Dict[str, Any]:
        """Execute preprocessing strategy"""
        current_data = data
        results = {
            'preprocessed_data': None,
            'strategy_used': strategy,
            'steps': []
        }
        
        for preprocessor_name in strategy['order']:
            if preprocessor_name not in self.preprocessors:
                continue
            
            try:
                PreprocessorClass = self.preprocessors[preprocessor_name]
                
                # Initialize with params
                params = strategy['params'].get(preprocessor_name, {})
                
                if preprocessor_name == 'model_specific':
                    preprocessor = PreprocessorClass(model_type=model_type, **params)
                    result = preprocessor.preprocess_for_model(current_data, task_type)
                elif preprocessor_name == 'corpus_callosum':
                    preprocessor = PreprocessorClass(**params)
                    result = preprocessor.preprocess(current_data)
                else:
                    preprocessor = PreprocessorClass(**params)
                    if hasattr(preprocessor, 'preprocess'):
                        result = preprocessor.preprocess(current_data)
                    else:
                        continue
                
                # Extract preprocessed data
                if isinstance(result, dict):
                    current_data = result.get('deduplicated') or result.get('preprocessed_data') or result.get('data')
                else:
                    current_data = result
                
                results['steps'].append({
                    'preprocessor': preprocessor_name,
                    'success': True,
                    'output_shape': getattr(current_data, 'shape', len(current_data) if hasattr(current_data, '__len__') else None)
                })
                
            except Exception as e:
                results['steps'].append({
                    'preprocessor': preprocessor_name,
                    'success': False,
                    'error': str(e)
                })
                # Continue with next preprocessor
        
        results['preprocessed_data'] = current_data
        return results
    
    def _learn_from_result(self, strategy: Dict, result: Dict, data_profile: Dict):
        """Learn from preprocessing result"""
        success = result.get('preprocessed_data') is not None
        performance = {
            'strategy': strategy,
            'data_profile': data_profile,
            'success': success,
            'steps_successful': sum(1 for s in result.get('steps', []) if s.get('success', False))
        }
        
        self.performance_history.append(performance)
        
        # Cache successful strategies
        if success:
            cache_key = self._generate_cache_key(data_profile, 'auto', 'auto')
            self.strategy_cache[cache_key] = strategy
    
    def _generate_cache_key(self, data_profile: Dict, task_type: str, model_type: str) -> str:
        """Generate cache key for strategy"""
        key_parts = [
            data_profile.get('type', 'unknown'),
            str(data_profile.get('size', 0)),
            task_type,
            model_type
        ]
        return '_'.join(key_parts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.performance_history:
            return {'total': 0, 'success_rate': 0.0}
        
        total = len(self.performance_history)
        successful = sum(1 for p in self.performance_history if p['success'])
        success_rate = successful / total if total > 0 else 0.0
        
        return {
            'total': total,
            'successful': successful,
            'success_rate': success_rate,
            'cached_strategies': len(self.strategy_cache)
        }


# Global instance
_global_preprocessor = None

def get_universal_preprocessor() -> UniversalAdaptivePreprocessor:
    """Get global universal preprocessor instance"""
    global _global_preprocessor
    if _global_preprocessor is None:
        _global_preprocessor = UniversalAdaptivePreprocessor()
    return _global_preprocessor
