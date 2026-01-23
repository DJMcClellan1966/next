"""
ML Toolbox Profiler
Comprehensive profiling system to identify bottlenecks and optimization opportunities

Features:
- Function-level profiling
- Pipeline profiling
- Memory profiling
- Performance reports
- Bottleneck identification
- Optimization recommendations
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
import time
import cProfile
import pstats
import io
from functools import wraps
from collections import defaultdict
import json
import warnings
from datetime import datetime
import traceback

sys.path.insert(0, str(Path(__file__).parent))

try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy/Pandas not available for profiling")


class MLProfiler:
    """
    ML Toolbox Profiler
    
    Comprehensive profiling system for performance analysis
    """
    
    def __init__(self, enable_memory_profiling: bool = False):
        """
        Initialize profiler
        
        Args:
            enable_memory_profiling: Enable memory profiling (requires memory_profiler)
        """
        self.enable_memory_profiling = enable_memory_profiling
        self.profile_data = defaultdict(list)
        self.function_times = defaultdict(list)
        self.pipeline_times = {}
        self.memory_usage = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.active_profiles = {}
        
        # Try to import memory profiler
        try:
            from memory_profiler import profile as mem_profile
            self.memory_profiler_available = True
        except ImportError:
            self.memory_profiler_available = False
            if enable_memory_profiling:
                warnings.warn("memory_profiler not available. Install with: pip install memory-profiler")
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator to profile a function
        
        Args:
            func: Function to profile
            
        Returns:
            Profiled function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Time execution
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage() if self.enable_memory_profiling else 0
            
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                self.profile_data[func_name].append({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                raise
            
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage() if self.enable_memory_profiling else 0
            
            # Record metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory if self.enable_memory_profiling else 0
            
            self.function_times[func_name].append(execution_time)
            self.call_counts[func_name] += 1
            
            if self.enable_memory_profiling:
                self.memory_usage[func_name].append(memory_delta)
            
            self.profile_data[func_name].append({
                'execution_time': execution_time,
                'memory_delta': memory_delta,
                'timestamp': datetime.now().isoformat(),
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            })
            
            return result
        
        return wrapper
    
    def profile_pipeline(self, pipeline_name: str):
        """
        Context manager for profiling entire pipelines
        
        Args:
            pipeline_name: Name of the pipeline
            
        Usage:
            with profiler.profile_pipeline('data_preprocessing'):
                # Your code here
                pass
        """
        return PipelineProfiler(self, pipeline_name)
    
    def profile_with_cprofile(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile function using cProfile
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Profile results
        """
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        stats_output = s.getvalue()
        
        return {
            'result': result,
            'stats': stats_output,
            'profile_stats': ps
        }
    
    def get_function_statistics(self, func_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for function(s)
        
        Args:
            func_name: Specific function name, or None for all
            
        Returns:
            Statistics dictionary
        """
        if func_name:
            functions = [func_name]
        else:
            functions = list(self.function_times.keys())
        
        stats = {}
        
        for func in functions:
            if func not in self.function_times:
                continue
            
            times = self.function_times[func]
            if not times:
                continue
            
            stats[func] = {
                'call_count': self.call_counts[func],
                'total_time': sum(times),
                'mean_time': np.mean(times) if NUMPY_AVAILABLE else sum(times) / len(times),
                'median_time': np.median(times) if NUMPY_AVAILABLE else sorted(times)[len(times)//2],
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times) if NUMPY_AVAILABLE else 0,
                'p95_time': np.percentile(times, 95) if NUMPY_AVAILABLE else sorted(times)[int(len(times)*0.95)]
            }
            
            if self.enable_memory_profiling and func in self.memory_usage:
                memory = self.memory_usage[func]
                stats[func]['memory'] = {
                    'mean_delta': np.mean(memory) if NUMPY_AVAILABLE else sum(memory) / len(memory),
                    'max_delta': max(memory),
                    'total_delta': sum(memory)
                }
        
        return stats
    
    def identify_bottlenecks(self, threshold_percentile: float = 95) -> List[Dict[str, Any]]:
        """
        Identify bottlenecks in the pipeline
        
        Args:
            threshold_percentile: Percentile threshold for bottleneck identification
            
        Returns:
            List of bottlenecks with recommendations
        """
        bottlenecks = []
        stats = self.get_function_statistics()
        
        if not stats:
            return bottlenecks
        
        # Calculate total time
        total_time = sum(s['total_time'] for s in stats.values())
        
        # Find functions above threshold
        threshold_time = total_time * (threshold_percentile / 100)
        
        for func_name, func_stats in stats.items():
            if func_stats['total_time'] > threshold_time:
                # Calculate percentage of total time
                percentage = (func_stats['total_time'] / total_time * 100) if total_time > 0 else 0
                
                # Generate recommendations
                recommendations = self._generate_recommendations(func_name, func_stats)
                
                bottlenecks.append({
                    'function': func_name,
                    'total_time': func_stats['total_time'],
                    'percentage': percentage,
                    'call_count': func_stats['call_count'],
                    'mean_time': func_stats['mean_time'],
                    'recommendations': recommendations,
                    'priority': 'high' if percentage > 10 else 'medium' if percentage > 5 else 'low'
                })
        
        # Sort by total time
        bottlenecks.sort(key=lambda x: x['total_time'], reverse=True)
        
        return bottlenecks
    
    def _generate_recommendations(self, func_name: str, stats: Dict[str, Any]) -> List[str]:
        """
        Generate optimization recommendations for a function
        
        Args:
            func_name: Function name
            stats: Function statistics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # High call count
        if stats['call_count'] > 1000:
            recommendations.append(f"High call count ({stats['call_count']}): Consider caching or memoization")
        
        # High variance
        if stats['std_time'] > stats['mean_time'] * 0.5:
            recommendations.append("High execution time variance: Check for conditional logic or data-dependent operations")
        
        # Slow mean time
        if stats['mean_time'] > 1.0:
            recommendations.append(f"Slow execution ({stats['mean_time']:.3f}s): Consider optimization or parallelization")
        
        # Memory usage
        if self.enable_memory_profiling and 'memory' in stats:
            if stats['memory']['mean_delta'] > 100 * 1024 * 1024:  # 100MB
                recommendations.append("High memory usage: Consider memory-efficient algorithms or data structures")
        
        # Function-specific recommendations
        func_lower = func_name.lower()
        
        if 'fit' in func_lower or 'train' in func_lower:
            recommendations.append("Training function: Consider early stopping, batch processing, or model simplification")
        
        if 'predict' in func_lower:
            recommendations.append("Prediction function: Consider batch prediction or model optimization")
        
        if 'preprocess' in func_lower or 'transform' in func_lower:
            recommendations.append("Preprocessing function: Consider caching transformed data or using more efficient transformers")
        
        if 'loop' in func_lower or 'iter' in func_lower:
            recommendations.append("Loop detected: Consider vectorization with NumPy/Pandas or parallel processing")
        
        if not recommendations:
            recommendations.append("Function performance appears acceptable")
        
        return recommendations
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive profiling report
        
        Args:
            output_file: Optional file path to save report
            
        Returns:
            Report string
        """
        report = f"""
{'='*80}
ML TOOLBOX PROFILING REPORT
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # Overall statistics
        stats = self.get_function_statistics()
        if stats:
            total_time = sum(s['total_time'] for s in stats.values())
            total_calls = sum(s['call_count'] for s in stats.values())
            
            report += f"""
OVERALL STATISTICS
{'-'*80}
Total Functions Profiled: {len(stats)}
Total Execution Time: {total_time:.4f} seconds
Total Function Calls: {total_calls:,}
Average Time per Call: {total_time / total_calls:.6f} seconds

"""
        
        # Top slowest functions
        if stats:
            report += """
TOP 10 SLOWEST FUNCTIONS (by total time)
{'-'*80}
"""
            sorted_funcs = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)[:10]
            
            for i, (func_name, func_stats) in enumerate(sorted_funcs, 1):
                percentage = (func_stats['total_time'] / total_time * 100) if total_time > 0 else 0
                report += f"""
{i}. {func_name}
   Total Time: {func_stats['total_time']:.4f}s ({percentage:.2f}%)
   Calls: {func_stats['call_count']:,}
   Mean Time: {func_stats['mean_time']:.6f}s
   Min/Max: {func_stats['min_time']:.6f}s / {func_stats['max_time']:.6f}s
   P95 Time: {func_stats['p95_time']:.6f}s
"""
        
        # Bottlenecks
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            report += f"""
BOTTLENECKS IDENTIFIED
{'-'*80}
Found {len(bottlenecks)} potential bottlenecks:

"""
            for i, bottleneck in enumerate(bottlenecks[:10], 1):
                report += f"""
{i}. {bottleneck['function']} [{bottleneck['priority'].upper()} PRIORITY]
   Total Time: {bottleneck['total_time']:.4f}s ({bottleneck['percentage']:.2f}% of total)
   Calls: {bottleneck['call_count']:,}
   Mean Time: {bottleneck['mean_time']:.6f}s
   
   Recommendations:
"""
                for rec in bottleneck['recommendations']:
                    report += f"   • {rec}\n"
        
        # Pipeline times
        if self.pipeline_times:
            report += f"""
PIPELINE EXECUTION TIMES
{'-'*80}
"""
            for pipeline_name, pipeline_data in self.pipeline_times.items():
                report += f"""
{pipeline_name}:
   Total Time: {pipeline_data.get('total_time', 0):.4f}s
   Steps: {len(pipeline_data.get('steps', []))}
"""
                for step in pipeline_data.get('steps', []):
                    report += f"   - {step['name']}: {step['time']:.4f}s\n"
        
        # Memory usage (if enabled)
        if self.enable_memory_profiling and self.memory_usage:
            report += f"""
MEMORY USAGE ANALYSIS
{'-'*80}
"""
            memory_stats = {}
            for func_name, memory_deltas in self.memory_usage.items():
                if memory_deltas:
                    memory_stats[func_name] = {
                        'mean': np.mean(memory_deltas) if NUMPY_AVAILABLE else sum(memory_deltas) / len(memory_deltas),
                        'max': max(memory_deltas),
                        'total': sum(memory_deltas)
                    }
            
            if memory_stats:
                sorted_memory = sorted(memory_stats.items(), key=lambda x: x[1]['max'], reverse=True)[:10]
                for func_name, mem_stats in sorted_memory:
                    report += f"""
{func_name}:
   Mean Memory Delta: {mem_stats['mean'] / (1024*1024):.2f} MB
   Max Memory Delta: {mem_stats['max'] / (1024*1024):.2f} MB
"""
        
        # Optimization recommendations
        report += f"""
OPTIMIZATION RECOMMENDATIONS
{'-'*80}
"""
        
        if bottlenecks:
            high_priority = [b for b in bottlenecks if b['priority'] == 'high']
            if high_priority:
                report += f"""
HIGH PRIORITY ({len(high_priority)} functions):
   Focus optimization efforts on these functions first as they consume the most time.
"""
        
        report += """
GENERAL RECOMMENDATIONS:
   • Use caching for frequently called functions with same inputs
   • Consider parallelization for independent operations
   • Profile memory usage to identify memory bottlenecks
   • Use vectorized operations (NumPy/Pandas) instead of loops
   • Consider early stopping for training functions
   • Batch operations when possible
   • Use efficient data structures (e.g., sets for lookups)
"""
        
        report += f"""
{'='*80}
End of Report
{'='*80}
"""
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def export_data(self, output_file: str):
        """
        Export profiling data to JSON
        
        Args:
            output_file: Output file path
        """
        data = {
            'function_statistics': self.get_function_statistics(),
            'pipeline_times': self.pipeline_times,
            'bottlenecks': self.identify_bottlenecks(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        data = convert_types(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            return 0
    
    def reset(self):
        """Reset all profiling data"""
        self.profile_data.clear()
        self.function_times.clear()
        self.pipeline_times.clear()
        self.memory_usage.clear()
        self.call_counts.clear()
        self.active_profiles.clear()


class PipelineProfiler:
    """Context manager for profiling pipelines"""
    
    def __init__(self, profiler: MLProfiler, pipeline_name: str):
        """
        Args:
            profiler: MLProfiler instance
            pipeline_name: Name of the pipeline
        """
        self.profiler = profiler
        self.pipeline_name = pipeline_name
        self.start_time = None
        self.steps = []
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        total_time = end_time - self.start_time
        
        self.profiler.pipeline_times[self.pipeline_name] = {
            'total_time': total_time,
            'steps': self.steps,
            'timestamp': datetime.now().isoformat()
        }
    
    def add_step(self, step_name: str, step_time: float):
        """Add a step to the pipeline"""
        self.steps.append({
            'name': step_name,
            'time': step_time
        })


class ProfiledMLToolbox:
    """
    Profiled wrapper for ML Toolbox
    
    Automatically profiles all operations
    """
    
    def __init__(self, toolbox=None, enable_profiling: bool = True):
        """
        Args:
            toolbox: MLToolbox instance
            enable_profiling: Enable profiling
        """
        if toolbox is None:
            try:
                from ml_toolbox import MLToolbox
                toolbox = MLToolbox()
            except ImportError:
                raise ImportError("ML Toolbox not available")
        
        self.toolbox = toolbox
        self.profiler = MLProfiler() if enable_profiling else None
    
    def profile_operation(self, operation_name: str, func: Callable, *args, **kwargs):
        """
        Profile a single operation
        
        Args:
            operation_name: Name of the operation
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if self.profiler:
            profiled_func = self.profiler.profile_function(func)
            return profiled_func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def get_profiling_report(self) -> str:
        """Get profiling report"""
        if self.profiler:
            return self.profiler.generate_report()
        else:
            return "Profiling not enabled"
    
    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """Get identified bottlenecks"""
        if self.profiler:
            return self.profiler.identify_bottlenecks()
        else:
            return []


def profile_ml_pipeline(pipeline_func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to profile an ML pipeline
    
    Args:
        pipeline_func: Pipeline function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Results with profiling data
    """
    profiler = MLProfiler()
    
    # Profile with cProfile for detailed analysis
    profile_result = profiler.profile_with_cprofile(pipeline_func, *args, **kwargs)
    
    # Get statistics
    stats = profiler.get_function_statistics()
    bottlenecks = profiler.identify_bottlenecks()
    
    # Generate report
    report = profiler.generate_report()
    
    return {
        'result': profile_result['result'],
        'stats': stats,
        'bottlenecks': bottlenecks,
        'report': report,
        'cprofile_stats': profile_result['stats']
    }


# Example usage
if __name__ == '__main__':
    # Create profiler
    profiler = MLProfiler(enable_memory_profiling=False)
    
    # Example: Profile a function
    @profiler.profile_function
    def example_function(n: int):
        """Example function to profile"""
        result = 0
        for i in range(n):
            result += i ** 2
        return result
    
    # Run function multiple times
    for _ in range(10):
        example_function(10000)
    
    # Profile a pipeline
    with profiler.profile_pipeline('example_pipeline'):
        example_function(1000)
        example_function(2000)
        example_function(3000)
    
    # Generate report
    report = profiler.generate_report('profiling_report.txt')
    print(report)
    
    # Get bottlenecks
    bottlenecks = profiler.identify_bottlenecks()
    print("\nBottlenecks:")
    for bottleneck in bottlenecks:
        print(f"  {bottleneck['function']}: {bottleneck['total_time']:.4f}s")
