"""
Test Medulla Performance Impact
Compares system performance and comprehensive ML tests with/without Medulla
"""
import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Install with: pip install psutil")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available")

try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    warnings.warn("ML Toolbox not available")


class SystemPerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self):
        self.metrics = []
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.metrics = []
    
    def record(self):
        """Record current system metrics"""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            self.metrics.append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'memory_used_mb': memory.used / (1024 * 1024)
            })
        except Exception as e:
            warnings.warn(f"Error recording metrics: {e}")
    
    def stop(self):
        """Stop monitoring"""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics:
            return {
                'duration': 0.0,
                'avg_cpu': 0.0,
                'max_cpu': 0.0,
                'avg_memory': 0.0,
                'max_memory': 0.0,
                'min_memory_available_mb': 0.0
            }
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]
        memory_available = [m['memory_available_mb'] for m in self.metrics]
        
        duration = (self.end_time or time.time()) - (self.start_time or 0)
        
        return {
            'duration': duration,
            'avg_cpu': sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
            'max_cpu': max(cpu_values) if cpu_values else 0.0,
            'min_cpu': min(cpu_values) if cpu_values else 0.0,
            'avg_memory': sum(memory_values) / len(memory_values) if memory_values else 0.0,
            'max_memory': max(memory_values) if memory_values else 0.0,
            'min_memory': min(memory_values) if memory_values else 0.0,
            'min_memory_available_mb': min(memory_available) if memory_available else 0.0,
            'max_memory_available_mb': max(memory_available) if memory_available else 0.0,
            'samples': len(self.metrics)
        }


def run_comprehensive_tests_with_toolbox(toolbox: MLToolbox, monitor: SystemPerformanceMonitor) -> Dict[str, Any]:
    """Run comprehensive ML tests using toolbox"""
    results = {
        'tests': {},
        'total_time': 0.0,
        'errors': 0,
        'successes': 0
    }
    
    monitor.start()
    
    try:
        # Simple tests
        print("\n[1/4] Running Simple Tests...")
        
        # Binary Classification
        try:
            start = time.time()
            monitor.record()
            
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Use toolbox
            if 'RandomForestClassifier' in toolbox.algorithms.components:
                model_class = toolbox.algorithms.components['RandomForestClassifier']
                model = model_class(n_estimators=10, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                elapsed = time.time() - start
                monitor.record()
                
                results['tests']['binary_classification'] = {
                    'accuracy': accuracy,
                    'time': elapsed,
                    'status': 'success'
                }
                results['successes'] += 1
            else:
                results['tests']['binary_classification'] = {'status': 'skipped', 'reason': 'model_not_available'}
        except Exception as e:
            results['tests']['binary_classification'] = {'status': 'error', 'error': str(e)}
            results['errors'] += 1
        
        # Simple Regression
        try:
            start = time.time()
            monitor.record()
            
            from sklearn.datasets import make_regression
            from sklearn.metrics import r2_score
            
            X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if 'RandomForestRegressor' in toolbox.algorithms.components:
                model_class = toolbox.algorithms.components['RandomForestRegressor']
                model = model_class(n_estimators=10, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                
                elapsed = time.time() - start
                monitor.record()
                
                results['tests']['simple_regression'] = {
                    'r2_score': r2,
                    'time': elapsed,
                    'status': 'success'
                }
                results['successes'] += 1
            else:
                results['tests']['simple_regression'] = {'status': 'skipped', 'reason': 'model_not_available'}
        except Exception as e:
            results['tests']['simple_regression'] = {'status': 'error', 'error': str(e)}
            results['errors'] += 1
        
        # Medium tests
        print("\n[2/4] Running Medium Tests...")
        
        # High-dimensional Classification
        try:
            start = time.time()
            monitor.record()
            
            X, y = make_classification(n_samples=2000, n_features=100, n_classes=2, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if 'RandomForestClassifier' in toolbox.algorithms.components:
                model_class = toolbox.algorithms.components['RandomForestClassifier']
                model = model_class(n_estimators=20, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                elapsed = time.time() - start
                monitor.record()
                
                results['tests']['high_dim_classification'] = {
                    'accuracy': accuracy,
                    'time': elapsed,
                    'status': 'success'
                }
                results['successes'] += 1
            else:
                results['tests']['high_dim_classification'] = {'status': 'skipped', 'reason': 'model_not_available'}
        except Exception as e:
            results['tests']['high_dim_classification'] = {'status': 'error', 'error': str(e)}
            results['errors'] += 1
        
        # Hard tests
        print("\n[3/4] Running Hard Tests...")
        
        # Very High-dimensional
        try:
            start = time.time()
            monitor.record()
            
            X, y = make_classification(n_samples=3000, n_features=500, n_classes=2, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if 'RandomForestClassifier' in toolbox.algorithms.components:
                model_class = toolbox.algorithms.components['RandomForestClassifier']
                model = model_class(n_estimators=30, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                elapsed = time.time() - start
                monitor.record()
                
                results['tests']['very_high_dim'] = {
                    'accuracy': accuracy,
                    'time': elapsed,
                    'status': 'success'
                }
                results['successes'] += 1
            else:
                results['tests']['very_high_dim'] = {'status': 'skipped', 'reason': 'model_not_available'}
        except Exception as e:
            results['tests']['very_high_dim'] = {'status': 'error', 'error': str(e)}
            results['errors'] += 1
        
        # Quantum operations test
        print("\n[4/4] Testing Quantum Operations...")
        
        try:
            start = time.time()
            monitor.record()
            
            qc = toolbox.get_quantum_computer(num_qubits=6)
            if qc:
                # Perform operations
                for i in range(6):
                    qc.apply_gate('H', i)
                
                # Parallel operations
                operations = [('X', i) for i in range(3)]
                qc.parallel_quantum_operation(operations)
                
                elapsed = time.time() - start
                monitor.record()
                
                metrics = qc.get_metrics()
                
                results['tests']['quantum_operations'] = {
                    'operations': metrics['operations_performed'],
                    'time': elapsed,
                    'avg_time_per_op': metrics['avg_operation_time'],
                    'status': 'success'
                }
                results['successes'] += 1
            else:
                results['tests']['quantum_operations'] = {'status': 'skipped', 'reason': 'quantum_computer_not_available'}
        except Exception as e:
            results['tests']['quantum_operations'] = {'status': 'error', 'error': str(e)}
            results['errors'] += 1
        
        results['total_time'] = time.time() - (monitor.start_time or 0)
        
    except Exception as e:
        results['errors'] += 1
        results['error'] = str(e)
    finally:
        monitor.stop()
    
    return results


def run_performance_comparison():
    """Run performance comparison with/without Medulla"""
    print("="*80)
    print("MEDULLA PERFORMANCE IMPACT TEST")
    print("="*80)
    print()
    
    if not TOOLBOX_AVAILABLE:
        print("[ERROR] ML Toolbox not available")
        return
    
    # Monitor for baseline (system idle)
    print("[BASELINE] Measuring system baseline...")
    baseline_monitor = SystemPerformanceMonitor()
    baseline_monitor.start()
    time.sleep(2)
    baseline_monitor.stop()
    baseline_summary = baseline_monitor.get_summary()
    print(f"[OK] Baseline: CPU={baseline_summary['avg_cpu']:.1f}%, Memory={baseline_summary['avg_memory']:.1f}%")
    
    # Test WITHOUT Medulla
    print("\n" + "="*80)
    print("TEST 1: WITHOUT MEDULLA")
    print("="*80)
    
    monitor_without = SystemPerformanceMonitor()
    toolbox_without = MLToolbox(auto_start_medulla=False)
    
    print("[INFO] Toolbox created without Medulla")
    print("[INFO] Running comprehensive tests...")
    
    results_without = run_comprehensive_tests_with_toolbox(toolbox_without, monitor_without)
    summary_without = monitor_without.get_summary()
    
    # Cleanup
    if toolbox_without.medulla and toolbox_without.medulla.regulation_running:
        toolbox_without.medulla.stop_regulation()
    
    print(f"\n[OK] Tests completed without Medulla")
    print(f"  Total time: {results_without['total_time']:.2f}s")
    print(f"  Successes: {results_without['successes']}")
    print(f"  Errors: {results_without['errors']}")
    print(f"  Avg CPU: {summary_without['avg_cpu']:.1f}%")
    print(f"  Max CPU: {summary_without['max_cpu']:.1f}%")
    print(f"  Avg Memory: {summary_without['avg_memory']:.1f}%")
    print(f"  Max Memory: {summary_without['max_memory']:.1f}%")
    
    # Wait for system to stabilize
    print("\n[WAIT] Waiting for system to stabilize...")
    time.sleep(3)
    
    # Test WITH Medulla
    print("\n" + "="*80)
    print("TEST 2: WITH MEDULLA")
    print("="*80)
    
    monitor_with = SystemPerformanceMonitor()
    toolbox_with = MLToolbox(auto_start_medulla=True)
    
    print("[INFO] Toolbox created with Medulla")
    if toolbox_with.medulla:
        medulla_status = toolbox_with.medulla.get_system_status()
        print(f"[INFO] Medulla state: {medulla_status['state']}")
        print(f"[INFO] Quantum resources: {medulla_status['quantum_resources']}")
    
    print("[INFO] Running comprehensive tests...")
    
    results_with = run_comprehensive_tests_with_toolbox(toolbox_with, monitor_with)
    summary_with = monitor_with.get_summary()
    
    # Cleanup
    if toolbox_with.medulla and toolbox_with.medulla.regulation_running:
        toolbox_with.medulla.stop_regulation()
    
    print(f"\n[OK] Tests completed with Medulla")
    print(f"  Total time: {results_with['total_time']:.2f}s")
    print(f"  Successes: {results_with['successes']}")
    print(f"  Errors: {results_with['errors']}")
    print(f"  Avg CPU: {summary_with['avg_cpu']:.1f}%")
    print(f"  Max CPU: {summary_with['max_cpu']:.1f}%")
    print(f"  Avg Memory: {summary_with['avg_memory']:.1f}%")
    print(f"  Max Memory: {summary_with['max_memory']:.1f}%")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Performance metrics comparison
    print("\n[1] System Performance Metrics:")
    print("-"*80)
    print(f"{'Metric':<30} {'Without Medulla':<20} {'With Medulla':<20} {'Difference':<20}")
    print("-"*80)
    
    cpu_diff = summary_with['avg_cpu'] - summary_without['avg_cpu']
    cpu_diff_pct = (cpu_diff / summary_without['avg_cpu'] * 100) if summary_without['avg_cpu'] > 0 else 0
    print(f"{'Avg CPU %':<30} {summary_without['avg_cpu']:<20.1f} {summary_with['avg_cpu']:<20.1f} {cpu_diff:+.1f}% ({cpu_diff_pct:+.1f}%)")
    
    max_cpu_diff = summary_with['max_cpu'] - summary_without['max_cpu']
    max_cpu_diff_pct = (max_cpu_diff / summary_without['max_cpu'] * 100) if summary_without['max_cpu'] > 0 else 0
    print(f"{'Max CPU %':<30} {summary_without['max_cpu']:<20.1f} {summary_with['max_cpu']:<20.1f} {max_cpu_diff:+.1f}% ({max_cpu_diff_pct:+.1f}%)")
    
    mem_diff = summary_with['avg_memory'] - summary_without['avg_memory']
    mem_diff_pct = (mem_diff / summary_without['avg_memory'] * 100) if summary_without['avg_memory'] > 0 else 0
    print(f"{'Avg Memory %':<30} {summary_without['avg_memory']:<20.1f} {summary_with['avg_memory']:<20.1f} {mem_diff:+.1f}% ({mem_diff_pct:+.1f}%)")
    
    max_mem_diff = summary_with['max_memory'] - summary_without['max_memory']
    max_mem_diff_pct = (max_mem_diff / summary_without['max_memory'] * 100) if summary_without['max_memory'] > 0 else 0
    print(f"{'Max Memory %':<30} {summary_without['max_memory']:<20.1f} {summary_with['max_memory']:<20.1f} {max_mem_diff:+.1f}% ({max_mem_diff_pct:+.1f}%)")
    
    time_diff = results_with['total_time'] - results_without['total_time']
    time_diff_pct = (time_diff / results_without['total_time'] * 100) if results_without['total_time'] > 0 else 0
    print(f"{'Total Time (s)':<30} {results_without['total_time']:<20.2f} {results_with['total_time']:<20.2f} {time_diff:+.2f}s ({time_diff_pct:+.1f}%)")
    
    # Test results comparison
    print("\n[2] Test Results Comparison:")
    print("-"*80)
    
    common_tests = set(results_without['tests'].keys()) & set(results_with['tests'].keys())
    
    for test_name in sorted(common_tests):
        test_without = results_without['tests'][test_name]
        test_with = results_with['tests'][test_name]
        
        if test_without.get('status') == 'success' and test_with.get('status') == 'success':
            time_without = test_without.get('time', 0)
            time_with = test_with.get('time', 0)
            time_diff = time_with - time_without
            time_diff_pct = (time_diff / time_without * 100) if time_without > 0 else 0
            
            print(f"\n{test_name}:")
            print(f"  Without Medulla: {time_without:.4f}s")
            print(f"  With Medulla: {time_with:.4f}s")
            print(f"  Difference: {time_diff:+.4f}s ({time_diff_pct:+.1f}%)")
            
            # Compare accuracy/scores if available
            if 'accuracy' in test_without and 'accuracy' in test_with:
                acc_without = test_without['accuracy']
                acc_with = test_with['accuracy']
                acc_diff = acc_with - acc_without
                print(f"  Accuracy: {acc_without:.4f} -> {acc_with:.4f} ({acc_diff:+.4f})")
            elif 'r2_score' in test_without and 'r2_score' in test_with:
                r2_without = test_without['r2_score']
                r2_with = test_with['r2_score']
                r2_diff = r2_with - r2_without
                print(f"  R2 Score: {r2_without:.4f} -> {r2_with:.4f} ({r2_diff:+.4f})")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n[WITHOUT MEDULLA]")
    print(f"  Total time: {results_without['total_time']:.2f}s")
    print(f"  Successes: {results_without['successes']}")
    print(f"  Errors: {results_without['errors']}")
    print(f"  Avg CPU: {summary_without['avg_cpu']:.1f}%")
    print(f"  Max CPU: {summary_without['max_cpu']:.1f}%")
    print(f"  Avg Memory: {summary_without['avg_memory']:.1f}%")
    print(f"  Max Memory: {summary_without['max_memory']:.1f}%")
    
    print(f"\n[WITH MEDULLA]")
    print(f"  Total time: {results_with['total_time']:.2f}s")
    print(f"  Successes: {results_with['successes']}")
    print(f"  Errors: {results_with['errors']}")
    print(f"  Avg CPU: {summary_with['avg_cpu']:.1f}%")
    print(f"  Max CPU: {summary_with['max_cpu']:.1f}%")
    print(f"  Avg Memory: {summary_with['avg_memory']:.1f}%")
    print(f"  Max Memory: {summary_with['max_memory']:.1f}%")
    
    # Impact analysis
    print(f"\n[IMPACT ANALYSIS]")
    cpu_impact = cpu_diff_pct
    mem_impact = mem_diff_pct
    time_impact = time_diff_pct
    
    if abs(cpu_impact) < 5:
        print(f"  CPU Impact: Minimal ({cpu_impact:+.1f}%)")
    elif cpu_impact > 0:
        print(f"  CPU Impact: Higher usage with Medulla ({cpu_impact:+.1f}%)")
    else:
        print(f"  CPU Impact: Lower usage with Medulla ({cpu_impact:+.1f}%)")
    
    if abs(mem_impact) < 5:
        print(f"  Memory Impact: Minimal ({mem_impact:+.1f}%)")
    elif mem_impact > 0:
        print(f"  Memory Impact: Higher usage with Medulla ({mem_impact:+.1f}%)")
    else:
        print(f"  Memory Impact: Lower usage with Medulla ({mem_impact:+.1f}%)")
    
    if abs(time_impact) < 5:
        print(f"  Time Impact: Minimal ({time_impact:+.1f}%)")
    elif time_impact > 0:
        print(f"  Time Impact: Slower with Medulla ({time_impact:+.1f}%)")
    else:
        print(f"  Time Impact: Faster with Medulla ({time_impact:+.1f}%)")
    
    # Save results
    results_file = Path('medulla_performance_comparison.json')
    comparison_data = {
        'baseline': baseline_summary,
        'without_medulla': {
            'system_performance': summary_without,
            'test_results': results_without
        },
        'with_medulla': {
            'system_performance': summary_with,
            'test_results': results_with
        },
        'comparison': {
            'cpu_diff_percent': cpu_diff_pct,
            'memory_diff_percent': mem_diff_pct,
            'time_diff_percent': time_diff_pct
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n[OK] Results saved to {results_file}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    run_performance_comparison()
