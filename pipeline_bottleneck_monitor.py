"""
Pipeline Bottleneck Monitor
Tracks CPU/memory usage and identifies slowest parts of ML pipeline
"""
import sys
from pathlib import Path
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
from contextlib import contextmanager
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    import cProfile
    import pstats
    from io import StringIO
    CPROFILE_AVAILABLE = True
except ImportError:
    CPROFILE_AVAILABLE = False
    warnings.warn("cProfile not available")

try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    warnings.warn("memory_profiler not available. Install with: pip install memory-profiler")


class PipelineBottleneckMonitor:
    """
    Comprehensive pipeline bottleneck monitor
    
    Tracks:
    - CPU usage (per function, per pipeline stage)
    - Memory usage (per function, per pipeline stage)
    - Execution time (per function, per pipeline stage)
    - Bottleneck identification
    - Resource usage trends
    """
    
    def __init__(self, sample_interval: float = 0.1):
        """
        Initialize monitor
        
        Args:
            sample_interval: How often to sample CPU/memory (seconds)
        """
        self.sample_interval = sample_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Metrics storage
        self.function_metrics = defaultdict(lambda: {
            'calls': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'max_time': 0.0,
            'min_time': float('inf'),
            'cpu_usage': [],
            'memory_usage': [],
            'peak_memory': 0.0
        })
        
        self.pipeline_stages = defaultdict(lambda: {
            'start_time': None,
            'end_time': None,
            'duration': 0.0,
            'cpu_samples': [],
            'memory_samples': [],
            'peak_cpu': 0.0,
            'peak_memory': 0.0,
            'avg_cpu': 0.0,
            'avg_memory': 0.0
        })
        
        self.bottlenecks = []
        self.profiling_data = {}
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                cpu_percent = process.cpu_percent(interval=None)
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Store samples for current pipeline stage
                # (This would be set by pipeline context)
                
                time.sleep(self.sample_interval)
            except Exception as e:
                warnings.warn(f"Monitoring error: {e}")
                break
    
    @contextmanager
    def monitor_function(self, function_name: str):
        """Context manager to monitor a function"""
        start_time = time.time()
        start_cpu = psutil.Process().cpu_percent(interval=None)
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            yield
        finally:
            end_time = time.time()
            end_cpu = psutil.Process().cpu_percent(interval=None)
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            duration = end_time - start_time
            
            # Update metrics
            metrics = self.function_metrics[function_name]
            metrics['calls'] += 1
            metrics['total_time'] += duration
            metrics['avg_time'] = metrics['total_time'] / metrics['calls']
            metrics['max_time'] = max(metrics['max_time'], duration)
            metrics['min_time'] = min(metrics['min_time'], duration)
            metrics['cpu_usage'].append(end_cpu)
            metrics['memory_usage'].append(end_memory)
            metrics['peak_memory'] = max(metrics['peak_memory'], end_memory)
    
    @contextmanager
    def monitor_pipeline_stage(self, stage_name: str):
        """Context manager to monitor a pipeline stage"""
        process = psutil.Process()
        stage = self.pipeline_stages[stage_name]
        
        stage['start_time'] = time.time()
        start_cpu = process.cpu_percent(interval=None)
        start_memory = process.memory_info().rss / 1024 / 1024
        
        # Start sampling
        samples = []
        monitoring = True
        
        def sample_loop():
            while monitoring:
                cpu = process.cpu_percent(interval=None)
                memory = process.memory_info().rss / 1024 / 1024
                samples.append({
                    'time': time.time(),
                    'cpu': cpu,
                    'memory': memory
                })
                time.sleep(self.sample_interval)
        
        sampler_thread = threading.Thread(target=sample_loop, daemon=True)
        sampler_thread.start()
        
        try:
            yield
        finally:
            monitoring = False
            sampler_thread.join(timeout=1.0)
            
            stage['end_time'] = time.time()
            stage['duration'] = stage['end_time'] - stage['start_time']
            stage['cpu_samples'] = [s['cpu'] for s in samples]
            stage['memory_samples'] = [s['memory'] for s in samples]
            
            if stage['cpu_samples']:
                stage['peak_cpu'] = max(stage['cpu_samples'])
                stage['avg_cpu'] = sum(stage['cpu_samples']) / len(stage['cpu_samples'])
            
            if stage['memory_samples']:
                stage['peak_memory'] = max(stage['memory_samples'])
                stage['avg_memory'] = sum(stage['memory_samples']) / len(stage['memory_samples'])
    
    def profile_function(self, func: Callable, *args, **kwargs):
        """Profile a function with cProfile"""
        if not CPROFILE_AVAILABLE:
            return func(*args, **kwargs), None
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        # Get stats
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20
        
        profile_output = s.getvalue()
        self.profiling_data[func.__name__] = profile_output
        
        return result, profile_output
    
    def identify_bottlenecks(self, threshold_percent: float = 10.0) -> List[Dict[str, Any]]:
        """
        Identify bottlenecks
        
        Args:
            threshold_percent: Minimum percentage of total time to be considered bottleneck
            
        Returns:
            List of bottleneck information
        """
        bottlenecks = []
        
        # Calculate total time
        total_time = sum(m['total_time'] for m in self.function_metrics.values())
        
        if total_time == 0:
            return bottlenecks
        
        # Find functions taking significant time
        for func_name, metrics in self.function_metrics.items():
            percent_time = (metrics['total_time'] / total_time) * 100
            
            if percent_time >= threshold_percent:
                bottlenecks.append({
                    'function': func_name,
                    'percent_time': percent_time,
                    'total_time': metrics['total_time'],
                    'avg_time': metrics['avg_time'],
                    'calls': metrics['calls'],
                    'peak_memory_mb': metrics['peak_memory'],
                    'avg_cpu': sum(metrics['cpu_usage']) / len(metrics['cpu_usage']) if metrics['cpu_usage'] else 0
                })
        
        # Sort by percent time
        bottlenecks.sort(key=lambda x: x['percent_time'], reverse=True)
        
        self.bottlenecks = bottlenecks
        return bottlenecks
    
    def identify_slow_pipeline_stages(self) -> List[Dict[str, Any]]:
        """Identify slow pipeline stages"""
        stages = []
        
        for stage_name, stage_data in self.pipeline_stages.items():
            stages.append({
                'stage': stage_name,
                'duration': stage_data['duration'],
                'peak_cpu': stage_data['peak_cpu'],
                'peak_memory_mb': stage_data['peak_memory'],
                'avg_cpu': stage_data['avg_cpu'],
                'avg_memory_mb': stage_data['avg_memory']
            })
        
        # Sort by duration
        stages.sort(key=lambda x: x['duration'], reverse=True)
        
        return stages
    
    def get_resource_usage_summary(self) -> Dict[str, Any]:
        """Get overall resource usage summary"""
        process = psutil.Process()
        
        # Current usage
        current_cpu = process.cpu_percent(interval=0.1)
        current_memory = process.memory_info().rss / 1024 / 1024
        
        # Function metrics summary
        total_function_time = sum(m['total_time'] for m in self.function_metrics.values())
        total_function_calls = sum(m['calls'] for m in self.function_metrics.values())
        
        # Pipeline stages summary
        total_pipeline_time = sum(s['duration'] for s in self.pipeline_stages.values())
        
        return {
            'current_cpu_percent': current_cpu,
            'current_memory_mb': current_memory,
            'total_function_time': total_function_time,
            'total_function_calls': total_function_calls,
            'total_pipeline_time': total_pipeline_time,
            'num_functions_monitored': len(self.function_metrics),
            'num_pipeline_stages': len(self.pipeline_stages),
            'bottlenecks_found': len(self.bottlenecks)
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive monitoring report"""
        report = []
        report.append("="*80)
        report.append("PIPELINE BOTTLENECK MONITORING REPORT")
        report.append("="*80)
        report.append("")
        
        # Resource usage summary
        summary = self.get_resource_usage_summary()
        report.append("RESOURCE USAGE SUMMARY")
        report.append("-"*80)
        report.append(f"Current CPU: {summary['current_cpu_percent']:.1f}%")
        report.append(f"Current Memory: {summary['current_memory_mb']:.1f} MB")
        report.append(f"Total Function Time: {summary['total_function_time']:.3f}s")
        report.append(f"Total Function Calls: {summary['total_function_calls']}")
        report.append(f"Total Pipeline Time: {summary['total_pipeline_time']:.3f}s")
        report.append("")
        
        # Bottlenecks
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            report.append("BOTTLENECKS (Top Functions)")
            report.append("-"*80)
            report.append(f"{'Function':<40} {'% Time':<10} {'Total Time':<12} {'Calls':<8} {'Peak Memory':<12}")
            report.append("-"*80)
            
            for bottleneck in bottlenecks[:10]:
                report.append(
                    f"{bottleneck['function']:<40} "
                    f"{bottleneck['percent_time']:<10.1f} "
                    f"{bottleneck['total_time']:<12.3f} "
                    f"{bottleneck['calls']:<8} "
                    f"{bottleneck['peak_memory_mb']:<12.1f} MB"
                )
            report.append("")
        
        # Slow pipeline stages
        slow_stages = self.identify_slow_pipeline_stages()
        if slow_stages:
            report.append("SLOW PIPELINE STAGES")
            report.append("-"*80)
            report.append(f"{'Stage':<40} {'Duration':<12} {'Peak CPU':<12} {'Peak Memory':<12}")
            report.append("-"*80)
            
            for stage in slow_stages[:10]:
                report.append(
                    f"{stage['stage']:<40} "
                    f"{stage['duration']:<12.3f}s "
                    f"{stage['peak_cpu']:<12.1f}% "
                    f"{stage['peak_memory_mb']:<12.1f} MB"
                )
            report.append("")
        
        # Top functions by time
        report.append("TOP FUNCTIONS BY TIME")
        report.append("-"*80)
        report.append(f"{'Function':<40} {'Total Time':<12} {'Avg Time':<12} {'Calls':<8}")
        report.append("-"*80)
        
        sorted_functions = sorted(
            self.function_metrics.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        
        for func_name, metrics in sorted_functions[:10]:
            report.append(
                f"{func_name:<40} "
                f"{metrics['total_time']:<12.3f}s "
                f"{metrics['avg_time']:<12.3f}s "
                f"{metrics['calls']:<8}"
            )
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)


# Decorator for easy function monitoring
def monitor_function(monitor: PipelineBottleneckMonitor, function_name: Optional[str] = None):
    """Decorator to monitor a function"""
    def decorator(func):
        name = function_name or func.__name__
        
        def wrapper(*args, **kwargs):
            with monitor.monitor_function(name):
                return func(*args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


# Example usage
if __name__ == '__main__':
    monitor = PipelineBottleneckMonitor(sample_interval=0.1)
    
    # Monitor a function
    @monitor_function(monitor, 'test_function')
    def test_function():
        time.sleep(0.1)
        return "done"
    
    # Test
    result = test_function()
    print(result)
    
    # Monitor pipeline stage
    with monitor.monitor_pipeline_stage('data_preprocessing'):
        time.sleep(0.2)
    
    # Generate report
    report = monitor.generate_report()
    print(report)
