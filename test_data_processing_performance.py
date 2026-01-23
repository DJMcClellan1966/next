"""
Comprehensive Data Processing Performance Test
Tests all preprocessors and identifies bottlenecks
"""
import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Any
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    warnings.warn("ML Toolbox not available")

try:
    from data_preprocessor import AdvancedDataPreprocessor, ConventionalPreprocessor
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    warnings.warn("Preprocessors not available")

try:
    from corpus_callosum_preprocessor import CorpusCallosumPreprocessor
    CORPUS_CALLOSUM_AVAILABLE = True
except ImportError:
    CORPUS_CALLOSUM_AVAILABLE = False
    warnings.warn("Corpus Callosum Preprocessor not available")

try:
    from gpu_accelerated_preprocessor import GPUAcceleratedPreprocessor, HybridPreprocessor
    GPU_PREPROCESSOR_AVAILABLE = True
except ImportError:
    GPU_PREPROCESSOR_AVAILABLE = False
    warnings.warn("GPU Preprocessor not available")


def generate_test_data(size: int, duplicate_ratio: float = 0.2, semantic_duplicate_ratio: float = 0.1) -> List[str]:
    """Generate test data with duplicates"""
    base_texts = [
        "Python programming language tutorial",
        "Machine learning algorithms explained",
        "Deep learning neural networks guide",
        "Data science with Python",
        "Natural language processing techniques",
        "Computer vision applications",
        "Reinforcement learning basics",
        "Statistical analysis methods",
        "Big data processing frameworks",
        "Cloud computing architectures"
    ]
    
    # Generate base dataset
    texts = []
    for i in range(size):
        base = base_texts[i % len(base_texts)]
        texts.append(base)
    
    # Add exact duplicates
    num_exact_dupes = int(size * duplicate_ratio)
    for i in range(num_exact_dupes):
        texts.append(texts[i % len(base_texts)])
    
    # Add semantic duplicates (similar but not exact)
    num_semantic_dupes = int(size * semantic_duplicate_ratio)
    semantic_variations = [
        ("Python programming language tutorial", "Learn Python programming"),
        ("Machine learning algorithms explained", "ML algorithms tutorial"),
        ("Deep learning neural networks guide", "Neural networks deep learning"),
        ("Data science with Python", "Python data science"),
        ("Natural language processing techniques", "NLP methods and techniques")
    ]
    for i in range(num_semantic_dupes):
        original, variation = semantic_variations[i % len(semantic_variations)]
        texts.append(variation)
    
    return texts


def test_preprocessor_performance(preprocessor, name: str, data: List[str], verbose: bool = False) -> Dict[str, Any]:
    """Test a single preprocessor's performance"""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        results = preprocessor.preprocess(data.copy(), verbose=verbose)
        processing_time = time.time() - start_time
        
        # Extract metrics
        # Handle different result structures
        output_size = results.get('final_count', 0)
        if output_size == 0:
            output_size = len(results.get('deduplicated', []))
        if output_size == 0:
            output_size = len(results.get('combined', {}).get('deduplicated', []))
        
        duplicates_removed = len(results.get('duplicates', []))
        if duplicates_removed == 0:
            # Check combined results
            combined = results.get('combined', {})
            if combined:
                duplicates_removed = combined.get('total_duplicates_removed', 0)
        
        metrics = {
            'name': name,
            'processing_time': processing_time,
            'input_size': len(data),
            'output_size': output_size,
            'duplicates_removed': duplicates_removed,
            'unsafe_filtered': results.get('stats', {}).get('unsafe_filtered', 0),
            'throughput': len(data) / processing_time if processing_time > 0 else float('inf'),
            'success': True,
            'error': None
        }
        
        # Additional metrics if available
        if 'processing_stages' in results:
            metrics['stages'] = len(results['processing_stages'])
        
        if 'gpu_stats' in results:
            metrics['gpu_used'] = results['gpu_stats'].get('gpu_used', False)
            metrics['gpu_operations'] = results['gpu_stats'].get('gpu_operations', 0)
        
        if 'corpus_callosum_stats' in results:
            metrics['left_hemisphere_ops'] = results['corpus_callosum_stats'].get('left_hemisphere_operations', 0)
            metrics['right_hemisphere_ops'] = results['corpus_callosum_stats'].get('right_hemisphere_operations', 0)
            metrics['time_saved'] = results['corpus_callosum_stats'].get('time_saved', 0)
        
        if verbose:
            print(f"  Processing Time: {processing_time:.3f}s")
            print(f"  Throughput: {metrics['throughput']:.1f} items/sec")
            print(f"  Duplicates Removed: {metrics['duplicates_removed']}")
            print(f"  Output Size: {metrics['output_size']}")
        
        return metrics
        
    except Exception as e:
        processing_time = time.time() - start_time
        if verbose:
            print(f"  ERROR: {str(e)}")
        
        return {
            'name': name,
            'processing_time': processing_time,
            'input_size': len(data),
            'output_size': 0,
            'duplicates_removed': 0,
            'unsafe_filtered': 0,
            'throughput': 0,
            'success': False,
            'error': str(e)
        }


def identify_bottlenecks(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identify bottlenecks from test results"""
    bottlenecks = {
        'slowest_preprocessor': None,
        'fastest_preprocessor': None,
        'speedup_opportunities': [],
        'bottleneck_analysis': []
    }
    
    if not results:
        return bottlenecks
    
    # Find slowest and fastest
    successful_results = [r for r in results if r['success']]
    if successful_results:
        slowest = max(successful_results, key=lambda x: x['processing_time'])
        fastest = min(successful_results, key=lambda x: x['processing_time'])
        
        bottlenecks['slowest_preprocessor'] = {
            'name': slowest['name'],
            'time': slowest['processing_time'],
            'throughput': slowest['throughput']
        }
        
        bottlenecks['fastest_preprocessor'] = {
            'name': fastest['name'],
            'time': fastest['processing_time'],
            'throughput': fastest['throughput']
        }
        
        # Calculate speedup opportunities
        if slowest['processing_time'] > 0 and fastest['processing_time'] > 0:
            speedup = slowest['processing_time'] / fastest['processing_time']
            bottlenecks['max_speedup'] = speedup
        elif fastest['processing_time'] > 0:
            bottlenecks['max_speedup'] = float('inf')  # Fastest is instant
        else:
            bottlenecks['max_speedup'] = 1.0  # Both are instant
        
        # Identify bottlenecks
        avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        
        for result in successful_results:
            if result['processing_time'] > avg_time * 1.5:  # 50% slower than average
                bottlenecks['bottleneck_analysis'].append({
                    'preprocessor': result['name'],
                    'time': result['processing_time'],
                    'avg_time': avg_time,
                    'slower_by': (result['processing_time'] / avg_time - 1) * 100
                })
    
    return bottlenecks


def run_comprehensive_performance_test(sizes: List[int] = [100, 500, 1000, 5000], verbose: bool = True) -> Dict[str, Any]:
    """Run comprehensive performance test across all preprocessors"""
    print("="*80)
    print("COMPREHENSIVE DATA PROCESSING PERFORMANCE TEST")
    print("="*80)
    print()
    
    all_results = {}
    
    for size in sizes:
        print(f"\n{'#'*80}")
        print(f"TESTING WITH DATASET SIZE: {size} items")
        print(f"{'#'*80}")
        
        # Generate test data
        test_data = generate_test_data(size, duplicate_ratio=0.2, semantic_duplicate_ratio=0.1)
        
        results = []
        
        # Test 1: ConventionalPreprocessor
        if PREPROCESSOR_AVAILABLE:
            conventional = ConventionalPreprocessor()
            metrics = test_preprocessor_performance(
                conventional, 
                'ConventionalPreprocessor', 
                test_data, 
                verbose=verbose
            )
            results.append(metrics)
        
        # Test 2: AdvancedDataPreprocessor
        if PREPROCESSOR_AVAILABLE:
            advanced = AdvancedDataPreprocessor()
            metrics = test_preprocessor_performance(
                advanced, 
                'AdvancedDataPreprocessor', 
                test_data, 
                verbose=verbose
            )
            results.append(metrics)
        
        # Test 3: Corpus Callosum Preprocessor (Intelligent Split)
        if CORPUS_CALLOSUM_AVAILABLE:
            corpus_intelligent = CorpusCallosumPreprocessor(
                split_strategy='intelligent',
                parallel_execution=False,
                combine_results=True
            )
            metrics = test_preprocessor_performance(
                corpus_intelligent, 
                'CorpusCallosumPreprocessor (Intelligent)', 
                test_data, 
                verbose=verbose
            )
            results.append(metrics)
        
        # Test 4: Corpus Callosum Preprocessor (Parallel)
        if CORPUS_CALLOSUM_AVAILABLE:
            corpus_parallel = CorpusCallosumPreprocessor(
                split_strategy='intelligent',
                parallel_execution=True,
                combine_results=True
            )
            metrics = test_preprocessor_performance(
                corpus_parallel, 
                'CorpusCallosumPreprocessor (Parallel)', 
                test_data, 
                verbose=verbose
            )
            results.append(metrics)
        
        # Test 5: GPU-Accelerated Preprocessor
        if GPU_PREPROCESSOR_AVAILABLE:
            gpu_preprocessor = GPUAcceleratedPreprocessor(use_gpu=True, fallback_to_cpu=True)
            if gpu_preprocessor.gpu_available:
                metrics = test_preprocessor_performance(
                    gpu_preprocessor, 
                    'GPUAcceleratedPreprocessor', 
                    test_data, 
                    verbose=verbose
                )
                results.append(metrics)
            else:
                if verbose:
                    print("\nGPU Preprocessor: GPU not available, skipping")
        
        # Test 6: Hybrid Preprocessor
        if GPU_PREPROCESSOR_AVAILABLE:
            hybrid = HybridPreprocessor(gpu_threshold=100)
            metrics = test_preprocessor_performance(
                hybrid, 
                'HybridPreprocessor', 
                test_data, 
                verbose=verbose
            )
            results.append(metrics)
        
        # Store results for this size
        all_results[f'size_{size}'] = {
            'dataset_size': size,
            'results': results,
            'bottlenecks': identify_bottlenecks(results)
        }
        
        # Print summary for this size
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR SIZE {size}")
        print(f"{'='*80}")
        
        successful = [r for r in results if r['success']]
        if successful:
            print(f"{'Preprocessor':<40} {'Time (s)':<12} {'Throughput':<15} {'Speedup':<10}")
            print("-"*80)
            
            fastest_time = min(r['processing_time'] for r in successful)
            
            for result in successful:
                speedup = fastest_time / result['processing_time'] if result['processing_time'] > 0 else 0
                print(f"{result['name']:<40} {result['processing_time']:<12.3f} {result['throughput']:<15.1f} {speedup:<10.2f}x")
        
        print()
    
    # Overall analysis
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Aggregate results
    all_metrics = []
    for size_key, size_data in all_results.items():
        for result in size_data['results']:
            if result['success']:
                all_metrics.append({
                    'name': result['name'],
                    'size': size_data['dataset_size'],
                    'time': result['processing_time'],
                    'throughput': result['throughput']
                })
    
    # Calculate averages
    preprocessor_avg = {}
    for metric in all_metrics:
        name = metric['name']
        if name not in preprocessor_avg:
            preprocessor_avg[name] = {'times': [], 'throughputs': []}
        preprocessor_avg[name]['times'].append(metric['time'])
        preprocessor_avg[name]['throughputs'].append(metric['throughput'])
    
    print(f"\n{'Preprocessor':<40} {'Avg Time (s)':<15} {'Avg Throughput':<15}")
    print("-"*80)
    
    for name, data in sorted(preprocessor_avg.items()):
        avg_time = sum(data['times']) / len(data['times'])
        avg_throughput = sum(data['throughputs']) / len(data['throughputs'])
        print(f"{name:<40} {avg_time:<15.3f} {avg_throughput:<15.1f}")
    
    # Bottleneck improvements
    print("\n" + "="*80)
    print("BOTTLENECK IMPROVEMENTS")
    print("="*80)
    
    # Compare AdvancedDataPreprocessor vs Corpus Callosum
    advanced_results = [m for m in all_metrics if m['name'] == 'AdvancedDataPreprocessor']
    corpus_results = [m for m in all_metrics if 'CorpusCallosum' in m['name']]
    
    if advanced_results and corpus_results:
        print("\nAdvancedDataPreprocessor vs Corpus Callosum:")
        for size in sorted(set(m['size'] for m in advanced_results + corpus_results)):
            adv = next((m for m in advanced_results if m['size'] == size), None)
            corp = next((m for m in corpus_results if m['size'] == size), None)
            
            if adv and corp:
                if adv['time'] > 0:
                    improvement = (adv['time'] - corp['time']) / adv['time'] * 100
                else:
                    improvement = 0
                speedup = adv['time'] / corp['time'] if corp['time'] > 0 and adv['time'] > 0 else (1.0 if adv['time'] == corp['time'] else float('inf'))
                if speedup == float('inf'):
                    print(f"  Size {size}: Corpus Callosum is instant (infinite speedup)")
                else:
                    print(f"  Size {size}: {improvement:.1f}% faster ({speedup:.2f}x speedup)")
    
    return {
        'all_results': all_results,
        'summary': {
            'preprocessor_averages': preprocessor_avg,
            'total_tests': len(all_metrics),
            'successful_tests': len([m for m in all_metrics if m])
        }
    }


def generate_performance_report(results: Dict[str, Any], output_file: str = 'data_processing_performance_report.json'):
    """Generate performance report"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nPerformance report saved to: {output_file}")


if __name__ == '__main__':
    # Run comprehensive test
    results = run_comprehensive_performance_test(
        sizes=[100, 500, 1000, 5000],
        verbose=True
    )
    
    # Generate report
    generate_performance_report(results)
    
    print("\n" + "="*80)
    print("PERFORMANCE TEST COMPLETE")
    print("="*80)
