"""
Advanced Compartment 1: Big Data
Large-scale data processing, AdvancedDataPreprocessor, and big data management
"""
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AdvancedBigDataCompartment:
    """
    Advanced Compartment 1: Big Data
    
    Components for large-scale data processing:
    - AdvancedDataPreprocessor ⭐ (Quantum + PocketFence for big data)
    - Big data detection and handling
    - Distributed processing capabilities
    - Memory-efficient operations
    - Streaming data support
    """
    
    def __init__(self):
        self.components = {}
        self.big_data_threshold = 10000  # Items considered "big data"
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all big data compartment components"""
        
        # AdvancedDataPreprocessor (main component for big data preprocessing)
        try:
            from data_preprocessor import AdvancedDataPreprocessor, ConventionalPreprocessor
            self.components['AdvancedDataPreprocessor'] = AdvancedDataPreprocessor
            self.components['ConventionalPreprocessor'] = ConventionalPreprocessor
        except ImportError as e:
            print(f"Warning: Could not import preprocessors: {e}")
        
        # Add component descriptions
        self.component_descriptions = {
            'AdvancedDataPreprocessor': {
                'description': 'Advanced preprocessing with Quantum Kernel + PocketFence Kernel for big data',
                'features': [
                    'Safety filtering (PocketFence)',
                    'Semantic deduplication (Quantum)',
                    'Intelligent categorization (Quantum)',
                    'Quality scoring (Quantum)',
                    'Dimensionality reduction (PCA/SVD)',
                    'Automatic feature creation',
                    'Memory-efficient processing',
                    'Batch processing support'
                ],
                'location': 'data_preprocessor.py',
                'category': 'Big Data Preprocessing',
                'placement': 'Advanced Compartment 1: Big Data'
            },
            'ConventionalPreprocessor': {
                'description': 'Basic preprocessing for smaller datasets',
                'features': [
                    'Basic safety filtering',
                    'Exact duplicate removal',
                    'Keyword-based categorization',
                    'Simple quality scoring'
                ],
                'location': 'data_preprocessor.py',
                'category': 'Basic Preprocessing',
                'placement': 'Advanced Compartment 1: Big Data'
            }
        }
    
    def is_big_data(self, data: List[str]) -> bool:
        """
        Detect if data is considered "big data"
        
        Args:
            data: List of text strings
            
        Returns:
            True if data size exceeds threshold
        """
        return len(data) >= self.big_data_threshold
    
    def get_preprocessor(self, advanced: bool = True, **kwargs):
        """
        Get a preprocessor instance optimized for data size
        
        Args:
            advanced: If True, use AdvancedDataPreprocessor
            **kwargs: Arguments to pass to preprocessor constructor
            
        Returns:
            Preprocessor instance
        """
        if advanced:
            return self.components['AdvancedDataPreprocessor'](**kwargs)
        else:
            return self.components['ConventionalPreprocessor']()
    
    def preprocess(self, data: List[str], advanced: bool = True, 
                   detect_big_data: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Preprocess data with big data detection and optimization
        
        Args:
            data: List of text strings to preprocess
            advanced: If True, use AdvancedDataPreprocessor
            detect_big_data: If True, detect and optimize for big data
            **kwargs: Arguments for preprocessor
            
        Returns:
            Preprocessing results dictionary with big data info
        """
        # Detect big data
        is_big = self.is_big_data(data) if detect_big_data else False
        
        # Optimize parameters for big data
        if is_big:
            # Optimize for big data: more aggressive compression, batch processing
            kwargs.setdefault('compression_ratio', 0.3)  # More compression for big data
            kwargs.setdefault('dedup_threshold', 0.90)  # More aggressive deduplication
            if 'verbose' not in kwargs:
                kwargs['verbose'] = False  # Less verbose for big data
        
        # Get preprocessor
        preprocessor_kwargs = {k: v for k, v in kwargs.items() 
                              if k in ['pocketfence_url', 'dedup_threshold', 'use_quantum', 
                                      'enable_compression', 'compression_ratio', 'compression_method']}
        
        preprocessor = self.get_preprocessor(advanced=advanced, **preprocessor_kwargs)
        
        # Preprocess
        verbose = kwargs.get('verbose', False)
        results = preprocessor.preprocess(data, verbose=verbose)
        
        # Add big data information
        results['big_data_info'] = {
            'is_big_data': is_big,
            'data_size': len(data),
            'threshold': self.big_data_threshold,
            'optimized_for_big_data': is_big
        }
        
        return results
    
    def process_in_batches(self, data: List[str], batch_size: int = 1000,
                          advanced: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Process large datasets in batches
        
        Args:
            data: List of text strings
            batch_size: Number of items per batch
            advanced: If True, use AdvancedDataPreprocessor
            **kwargs: Arguments for preprocessor
            
        Returns:
            Combined preprocessing results
        """
        all_results = {
            'deduplicated': [],
            'duplicates': [],
            'categorized': {},
            'quality_scores': [],
            'compressed_embeddings': None,
            'big_data_info': {
                'processed_in_batches': True,
                'batch_size': batch_size,
                'num_batches': (len(data) + batch_size - 1) // batch_size
            }
        }
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_results = self.preprocess(batch, advanced=advanced, 
                                          detect_big_data=False, **kwargs)
            
            # Combine results
            all_results['deduplicated'].extend(batch_results['deduplicated'])
            all_results['duplicates'].extend(batch_results['duplicates'])
            
            # Merge categories
            for cat, items in batch_results.get('categorized', {}).items():
                if cat not in all_results['categorized']:
                    all_results['categorized'][cat] = []
                all_results['categorized'][cat].extend(items)
            
            all_results['quality_scores'].extend(batch_results.get('quality_scores', []))
        
        # Final deduplication across batches
        if advanced and len(all_results['deduplicated']) > 0:
            final_results = self.preprocess(
                all_results['deduplicated'],
                advanced=advanced,
                detect_big_data=True,
                **kwargs
            )
            all_results['deduplicated'] = final_results['deduplicated']
            all_results['compressed_embeddings'] = final_results.get('compressed_embeddings')
        
        all_results['final_count'] = len(all_results['deduplicated'])
        all_results['total_processed'] = len(data)
        
        return all_results
    
    def list_components(self):
        """List all available components in this compartment"""
        print("="*80)
        print("ADVANCED COMPARTMENT 1: BIG DATA")
        print("="*80)
        print("\nComponents:")
        for name, component in self.components.items():
            desc = self.component_descriptions.get(name, {})
            print(f"\n{name}:")
            print(f"  Description: {desc.get('description', 'N/A')}")
            print(f"  Location: {desc.get('location', 'N/A')}")
            print(f"  Category: {desc.get('category', 'N/A')}")
            print(f"  Placement: {desc.get('placement', 'N/A')}")
            if 'features' in desc:
                print(f"  Features:")
                for feature in desc['features']:
                    print(f"    - {feature}")
        print("\n" + "="*80)
    
    def get_info(self):
        """Get information about this compartment"""
        return {
            'name': 'Advanced Compartment 1: Big Data',
            'description': 'Large-scale data processing, AdvancedDataPreprocessor',
            'components': list(self.components.keys()),
            'component_count': len(self.components),
            'big_data_threshold': self.big_data_threshold,
            'main_component': 'AdvancedDataPreprocessor'
        }
