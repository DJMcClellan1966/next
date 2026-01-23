"""
Corpus Callosum Preprocessor
Combines AdvancedDataPreprocessor and ConventionalPreprocessor
Like two brain hemispheres working together

Architecture:
- Left Hemisphere (ConventionalPreprocessor): Fast, exact operations
- Right Hemisphere (AdvancedDataPreprocessor): Semantic, intelligent operations
- Corpus Callosum: Coordinates and combines both
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    from data_preprocessor import AdvancedDataPreprocessor, ConventionalPreprocessor
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    warnings.warn("Preprocessors not available")


class CorpusCallosumPreprocessor:
    """
    Corpus Callosum Preprocessor
    
    Combines AdvancedDataPreprocessor and ConventionalPreprocessor
    like two brain hemispheres working together
    
    Left Hemisphere (Conventional): Fast, exact operations
    Right Hemisphere (Advanced): Semantic, intelligent operations
    Corpus Callosum: Coordinates and combines
    """
    
    def __init__(self,
                 parallel_execution: bool = True,
                 split_strategy: str = 'intelligent',
                 combine_results: bool = True):
        """
        Initialize Corpus Callosum Preprocessor
        
        Args:
            parallel_execution: Execute both preprocessors in parallel
            split_strategy: How to split work ('intelligent', 'equal', 'fast_first')
            combine_results: Combine results from both preprocessors
        """
        self.parallel_execution = parallel_execution
        self.split_strategy = split_strategy
        self.combine_results = combine_results
        
        # Initialize both "hemispheres"
        self.left_hemisphere = ConventionalPreprocessor() if PREPROCESSOR_AVAILABLE else None
        self.right_hemisphere = AdvancedDataPreprocessor() if PREPROCESSOR_AVAILABLE else None
        
        # Statistics
        self.stats = {
            'left_hemisphere_operations': 0,
            'right_hemisphere_operations': 0,
            'parallel_executions': 0,
            'combined_results': 0,
            'time_saved': 0.0
        }
    
    def preprocess(self, raw_data: List[str], verbose: bool = False) -> Dict[str, Any]:
        """
        Preprocess using both hemispheres
        
        Args:
            raw_data: List of text items
            verbose: Print progress
            
        Returns:
            Combined preprocessing results
        """
        start_time = time.time()
        
        if verbose:
            print("="*80)
            print("CORPUS CALLOSUM PREPROCESSING")
            print("="*80)
            print(f"Input: {len(raw_data)} items")
            print()
        
        # Strategy 1: Intelligent Split
        if self.split_strategy == 'intelligent':
            results = self._intelligent_split_preprocess(raw_data, verbose)
        
        # Strategy 2: Parallel Execution
        elif self.parallel_execution:
            results = self._parallel_preprocess(raw_data, verbose)
        
        # Strategy 3: Sequential with Best Features
        else:
            results = self._sequential_combined_preprocess(raw_data, verbose)
        
        # Combine results
        if self.combine_results:
            results = self._combine_results(results, verbose)
        
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        results['corpus_callosum_stats'] = self.stats.copy()
        
        if verbose:
            print()
            print("="*80)
            print("CORPUS CALLOSUM RESULTS")
            print("="*80)
            print(f"Processing Time: {processing_time:.2f}s")
            print(f"Left Hemisphere Operations: {self.stats['left_hemisphere_operations']}")
            print(f"Right Hemisphere Operations: {self.stats['right_hemisphere_operations']}")
            print(f"Time Saved: {self.stats['time_saved']:.2f}s")
            print("="*80)
        
        return results
    
    def _intelligent_split_preprocess(self, raw_data: List[str], verbose: bool = False) -> Dict[str, Any]:
        """
        Intelligent split: Each hemisphere handles what it does best
        
        Left Hemisphere (Conventional): Fast operations
        - Exact duplicate removal
        - Basic safety filtering
        - Simple quality checks
        
        Right Hemisphere (Advanced): Semantic operations
        - Semantic deduplication
        - Intelligent categorization
        - Quality scoring
        - Embeddings
        """
        if verbose:
            print("[Intelligent Split Strategy]")
            print("Left Hemisphere (Conventional): Fast, exact operations")
            print("Right Hemisphere (Advanced): Semantic, intelligent operations")
            print()
        
        results = {
            'original_count': len(raw_data),
            'left_hemisphere': {},
            'right_hemisphere': {},
            'combined': {}
        }
        
        # Phase 1: Left Hemisphere - Fast initial cleanup
        if verbose:
            print("[Phase 1] Left Hemisphere: Fast Initial Cleanup")
        
        left_start = time.time()
        left_results = self.left_hemisphere.preprocess(raw_data.copy(), verbose=verbose)
        left_time = time.time() - left_start
        
        results['left_hemisphere'] = left_results
        self.stats['left_hemisphere_operations'] += 1
        
        # Get cleaned data from left hemisphere
        cleaned_data = left_results.get('deduplicated', raw_data)
        
        if verbose:
            print(f"  Left Hemisphere Time: {left_time:.2f}s")
            print(f"  Cleaned: {len(cleaned_data)} items (removed {len(raw_data) - len(cleaned_data)} exact duplicates)")
            print()
        
        # Phase 2: Right Hemisphere - Semantic processing on cleaned data
        if verbose:
            print("[Phase 2] Right Hemisphere: Semantic Processing")
        
        right_start = time.time()
        right_results = self.right_hemisphere.preprocess(cleaned_data.copy(), verbose=verbose)
        right_time = time.time() - right_start
        
        results['right_hemisphere'] = right_results
        self.stats['right_hemisphere_operations'] += 1
        
        if verbose:
            print(f"  Right Hemisphere Time: {right_time:.2f}s")
            print(f"  Final: {right_results.get('final_count', len(cleaned_data))} items")
            print()
        
        # Calculate time saved (compared to sequential AdvancedDataPreprocessor)
        sequential_time = left_time + right_time
        advanced_only_time = right_time * 1.5  # Estimate (AdvancedDataPreprocessor would be slower on full data)
        self.stats['time_saved'] = max(0, advanced_only_time - sequential_time)
        
        return results
    
    def _parallel_preprocess(self, raw_data: List[str], verbose: bool = False) -> Dict[str, Any]:
        """
        Parallel execution: Both hemispheres work simultaneously
        """
        if verbose:
            print("[Parallel Execution Strategy]")
            print("Both hemispheres working simultaneously")
            print()
        
        results = {
            'original_count': len(raw_data),
            'left_hemisphere': {},
            'right_hemisphere': {},
            'combined': {}
        }
        
        # Execute both in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            left_future = executor.submit(self.left_hemisphere.preprocess, raw_data.copy(), verbose)
            right_future = executor.submit(self.right_hemisphere.preprocess, raw_data.copy(), verbose)
            
            # Wait for both to complete
            left_results = left_future.result()
            right_results = right_future.result()
        
        results['left_hemisphere'] = left_results
        results['right_hemisphere'] = right_results
        
        self.stats['left_hemisphere_operations'] += 1
        self.stats['right_hemisphere_operations'] += 1
        self.stats['parallel_executions'] += 1
        
        # Time saved = max(left, right) instead of sum(left + right)
        left_time = left_results.get('processing_time', 0)
        right_time = right_results.get('processing_time', 0)
        sequential_time = left_time + right_time
        parallel_time = max(left_time, right_time)
        self.stats['time_saved'] = sequential_time - parallel_time
        
        if verbose:
            print(f"Parallel execution saved: {self.stats['time_saved']:.2f}s")
            print()
        
        return results
    
    def _sequential_combined_preprocess(self, raw_data: List[str], verbose: bool = False) -> Dict[str, Any]:
        """
        Sequential execution with best features from both
        """
        if verbose:
            print("[Sequential Combined Strategy]")
            print("Using best features from both hemispheres sequentially")
            print()
        
        results = {
            'original_count': len(raw_data),
            'processing_stages': []
        }
        
        # Stage 1: Left Hemisphere - Fast exact duplicate removal
        if verbose:
            print("[Stage 1] Left Hemisphere: Exact Duplicate Removal")
        
        left_results = self.left_hemisphere.preprocess(raw_data.copy(), verbose=verbose)
        exact_deduplicated = left_results.get('deduplicated', raw_data)
        results['processing_stages'].append({
            'stage': 'exact_deduplication',
            'count': len(exact_deduplicated),
            'removed': len(raw_data) - len(exact_deduplicated)
        })
        self.stats['left_hemisphere_operations'] += 1
        
        # Stage 2: Right Hemisphere - Semantic processing
        if verbose:
            print("[Stage 2] Right Hemisphere: Semantic Processing")
        
        right_results = self.right_hemisphere.preprocess(exact_deduplicated.copy(), verbose=verbose)
        results['processing_stages'].append({
            'stage': 'semantic_processing',
            'count': right_results.get('final_count', len(exact_deduplicated))
        })
        self.stats['right_hemisphere_operations'] += 1
        
        # Combine
        results['left_hemisphere'] = left_results
        results['right_hemisphere'] = right_results
        
        return results
    
    def _combine_results(self, results: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """
        Combine results from both hemispheres
        
        Uses best features from each:
        - Left: Fast exact duplicate removal
        - Right: Semantic deduplication, embeddings, quality scores
        """
        if verbose:
            print("[Corpus Callosum] Combining Results from Both Hemispheres")
        
        left = results.get('left_hemisphere', {})
        right = results.get('right_hemisphere', {})
        
        combined = {
            'original_count': results.get('original_count', 0),
            'final_count': right.get('final_count', left.get('final_count', 0)),
            'exact_duplicates_removed': len(left.get('duplicates', [])),
            'semantic_duplicates_removed': len(right.get('duplicates', [])),
            'total_duplicates_removed': len(left.get('duplicates', [])) + len(right.get('duplicates', [])),
            'unsafe_filtered': left.get('stats', {}).get('unsafe_filtered', 0) + right.get('stats', {}).get('unsafe_filtered', 0),
            'categorized': right.get('categorized', {}),
            'quality_scores': right.get('quality_scores', []),
            'embeddings': right.get('compressed_embeddings', []),
            'compression_info': right.get('compression_info', {}),
            'processing_stages': results.get('processing_stages', [])
        }
        
        # Best of both: Use left's fast exact dedup, right's semantic features
        combined['deduplicated'] = right.get('deduplicated', left.get('deduplicated', []))
        combined['safe_data'] = right.get('safe_data', left.get('safe_data', []))
        
        results['combined'] = combined
        self.stats['combined_results'] += 1
        
        if verbose:
            print(f"  Combined: {combined['final_count']} items")
            print(f"  Exact Duplicates Removed: {combined['exact_duplicates_removed']}")
            print(f"  Semantic Duplicates Removed: {combined['semantic_duplicates_removed']}")
            print(f"  Total Removed: {combined['total_duplicates_removed']}")
            print()
        
        return results
    
    def preprocess_hybrid(self, raw_data: List[str], verbose: bool = False) -> Dict[str, Any]:
        """
        Hybrid preprocessing: Route operations intelligently
        
        Fast operations → Left Hemisphere
        Semantic operations → Right Hemisphere
        """
        if verbose:
            print("="*80)
            print("HYBRID CORPUS CALLOSUM PREPROCESSING")
            print("="*80)
            print("Intelligent routing: Fast → Left, Semantic → Right")
            print()
        
        results = {
            'original_count': len(raw_data),
            'routing': {}
        }
        
        # Route 1: Fast exact operations to Left Hemisphere
        if verbose:
            print("[Route 1] Left Hemisphere: Fast Operations")
        
        left_start = time.time()
        left_data = raw_data.copy()
        
        # Fast exact duplicate removal
        left_results = self.left_hemisphere.preprocess(left_data, verbose=verbose)
        exact_cleaned = left_results.get('deduplicated', left_data)
        
        left_time = time.time() - left_start
        results['routing']['left_hemisphere'] = {
            'operations': ['exact_deduplication', 'basic_filtering'],
            'time': left_time,
            'items_processed': len(exact_cleaned)
        }
        self.stats['left_hemisphere_operations'] += 1
        
        # Route 2: Semantic operations to Right Hemisphere
        if verbose:
            print("[Route 2] Right Hemisphere: Semantic Operations")
        
        right_start = time.time()
        
        # Semantic processing on cleaned data
        right_results = self.right_hemisphere.preprocess(exact_cleaned.copy(), verbose=verbose)
        
        right_time = time.time() - right_start
        results['routing']['right_hemisphere'] = {
            'operations': ['semantic_deduplication', 'categorization', 'quality_scoring', 'embeddings'],
            'time': right_time,
            'items_processed': right_results.get('final_count', len(exact_cleaned))
        }
        self.stats['right_hemisphere_operations'] += 1
        
        # Combine
        results['left_hemisphere'] = left_results
        results['right_hemisphere'] = right_results
        results = self._combine_results(results, verbose)
        
        # Calculate efficiency
        total_time = left_time + right_time
        if right_results.get('processing_time', 0) > 0:
            # Compare to using only AdvancedDataPreprocessor on full data
            estimated_advanced_only = right_results['processing_time'] * (len(raw_data) / len(exact_cleaned))
            efficiency_gain = estimated_advanced_only - total_time
            results['efficiency_gain'] = efficiency_gain
            results['efficiency_percent'] = (efficiency_gain / estimated_advanced_only * 100) if estimated_advanced_only > 0 else 0
        
        if verbose:
            print(f"Total Time: {total_time:.2f}s")
            if 'efficiency_gain' in results:
                print(f"Efficiency Gain: {results['efficiency_gain']:.2f}s ({results.get('efficiency_percent', 0):.1f}%)")
            print()
        
        return results
    
    def get_hemisphere_stats(self) -> Dict[str, Any]:
        """Get statistics from both hemispheres"""
        return {
            'left_hemisphere': {
                'operations': self.stats['left_hemisphere_operations'],
                'type': 'ConventionalPreprocessor',
                'strengths': ['Fast', 'Exact matching', 'Lightweight']
            },
            'right_hemisphere': {
                'operations': self.stats['right_hemisphere_operations'],
                'type': 'AdvancedDataPreprocessor',
                'strengths': ['Semantic', 'Intelligent', 'Advanced features']
            },
            'corpus_callosum': {
                'parallel_executions': self.stats['parallel_executions'],
                'combined_results': self.stats['combined_results'],
                'time_saved': self.stats['time_saved']
            }
        }


# Example usage
if __name__ == '__main__':
    # Create corpus callosum preprocessor
    preprocessor = CorpusCallosumPreprocessor(
        parallel_execution=True,
        split_strategy='intelligent',
        combine_results=True
    )
    
    # Test data
    texts = [
        "Python programming language",
        "Learn Python coding",
        "Machine learning tutorial",
        "Python programming language",  # Exact duplicate
        "ML tutorial"  # Semantic duplicate of "Machine learning tutorial"
    ]
    
    # Preprocess
    results = preprocessor.preprocess(texts, verbose=True)
    
    # Get stats
    stats = preprocessor.get_hemisphere_stats()
    print("\nHemisphere Statistics:")
    print(stats)
