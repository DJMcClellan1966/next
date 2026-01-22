"""
ML Benchmark Suite
Comprehensive benchmarking framework for ML Toolbox

Tests from simple to complex:
1. Simple classification (Iris)
2. Simple regression (Boston Housing)
3. Text classification
4. Image classification (MNIST)
5. Time series forecasting
6. Large-scale datasets
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import time
import json
import datetime
import warnings
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))


class MLBenchmarkSuite:
    """
    ML Benchmark Suite
    
    Comprehensive benchmarking from simple to complex
    """
    
    def __init__(self):
        """Initialize benchmark suite"""
        self.results = []
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check required dependencies"""
        try:
            import sklearn
            from sklearn import datasets
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            warnings.warn("sklearn required for benchmarks")
        
        try:
            import torch
            self.torch_available = True
        except ImportError:
            self.torch_available = False
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all benchmarks from simple to complex
        
        Returns:
            Comprehensive benchmark results
        """
        print("=" * 80)
        print("ML Toolbox Benchmark Suite")
        print("=" * 80)
        
        all_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'benchmarks': []
        }
        
        # 1. Simple Classification (Iris)
        print("\n[1/6] Simple Classification (Iris Dataset)")
        result = self.benchmark_iris_classification()
        all_results['benchmarks'].append(result)
        
        # 2. Simple Regression (Boston Housing)
        print("\n[2/6] Simple Regression (Boston Housing)")
        result = self.benchmark_boston_regression()
        all_results['benchmarks'].append(result)
        
        # 3. Text Classification
        print("\n[3/6] Text Classification")
        result = self.benchmark_text_classification()
        all_results['benchmarks'].append(result)
        
        # 4. Image Classification (MNIST)
        print("\n[4/6] Image Classification (MNIST)")
        result = self.benchmark_mnist_classification()
        all_results['benchmarks'].append(result)
        
        # 5. Time Series Forecasting
        print("\n[5/6] Time Series Forecasting")
        result = self.benchmark_time_series()
        all_results['benchmarks'].append(result)
        
        # 6. Large-scale Test
        print("\n[6/6] Large-scale Dataset")
        result = self.benchmark_large_scale()
        all_results['benchmarks'].append(result)
        
        # Generate summary
        summary = self._generate_summary(all_results)
        all_results['summary'] = summary
        
        return all_results
    
    def benchmark_iris_classification(self) -> Dict[str, Any]:
        """Benchmark: Iris classification (simple)"""
        if not self.sklearn_available:
            return {'error': 'sklearn required'}
        
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # Load data
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {
            'name': 'Iris Classification',
            'difficulty': 'Simple',
            'dataset_size': len(X),
            'features': X.shape[1],
            'classes': len(np.unique(y)),
            'models': []
        }
        
        # Test with ML Toolbox
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            
            # Test Simple ML Tasks
            start_time = time.time()
            simple = toolbox.algorithms.get_simple_ml_tasks()
            result = simple.train_classifier(X_train, y_train, model_type='random_forest')
            training_time = time.time() - start_time
            
            # Evaluate
            y_pred = result['model'].predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['models'].append({
                'name': 'ML Toolbox - Simple ML',
                'accuracy': accuracy,
                'training_time': training_time,
                'status': 'success'
            })
        except Exception as e:
            results['models'].append({
                'name': 'ML Toolbox - Simple ML',
                'error': str(e),
                'status': 'failed'
            })
        
        # Test with scikit-learn (baseline)
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            start_time = time.time()
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            sklearn_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            sklearn_accuracy = accuracy_score(y_test, y_pred)
            
            results['models'].append({
                'name': 'scikit-learn (baseline)',
                'accuracy': sklearn_accuracy,
                'training_time': sklearn_time,
                'status': 'success'
            })
        except Exception as e:
            results['models'].append({
                'name': 'scikit-learn (baseline)',
                'error': str(e),
                'status': 'failed'
            })
        
        return results
    
    def benchmark_boston_regression(self) -> Dict[str, Any]:
        """Benchmark: Boston Housing regression"""
        if not self.sklearn_available:
            return {'error': 'sklearn required'}
        
        try:
            from sklearn.datasets import fetch_california_housing
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_squared_error
            
            # Load data (using California Housing as Boston is deprecated)
            housing = fetch_california_housing()
            X, y = housing.data, housing.target
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        except:
            # Fallback to synthetic data
            X = np.random.rand(500, 8)
            y = np.random.rand(500)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        results = {
            'name': 'Housing Regression',
            'difficulty': 'Simple',
            'dataset_size': len(X),
            'features': X.shape[1],
            'models': []
        }
        
        # Test with ML Toolbox
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            
            start_time = time.time()
            simple = toolbox.algorithms.get_simple_ml_tasks()
            result = simple.train_regressor(X_train, y_train, model_type='random_forest')
            training_time = time.time() - start_time
            
            y_pred = result['model'].predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            results['models'].append({
                'name': 'ML Toolbox - Simple ML',
                'r2_score': r2,
                'mse': mse,
                'training_time': training_time,
                'status': 'success'
            })
        except Exception as e:
            results['models'].append({
                'name': 'ML Toolbox - Simple ML',
                'error': str(e),
                'status': 'failed'
            })
        
        # Baseline
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            start_time = time.time()
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            sklearn_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            sklearn_r2 = r2_score(y_test, y_pred)
            sklearn_mse = mean_squared_error(y_test, y_pred)
            
            results['models'].append({
                'name': 'scikit-learn (baseline)',
                'r2_score': sklearn_r2,
                'mse': sklearn_mse,
                'training_time': sklearn_time,
                'status': 'success'
            })
        except Exception as e:
            results['models'].append({
                'name': 'scikit-learn (baseline)',
                'error': str(e),
                'status': 'failed'
            })
        
        return results
    
    def benchmark_text_classification(self) -> Dict[str, Any]:
        """Benchmark: Text classification"""
        results = {
            'name': 'Text Classification',
            'difficulty': 'Medium',
            'models': []
        }
        
        # Create synthetic text data
        texts = [
            "This is a positive review",
            "I love this product",
            "Great quality and fast shipping",
            "This is a negative review",
            "Poor quality product",
            "Not worth the money",
            "Excellent service",
            "Terrible experience"
        ] * 50  # 400 samples
        
        labels = [1, 1, 1, 0, 0, 0, 1, 0] * 50
        
        # Simple feature extraction (bag of words)
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        vectorizer = CountVectorizer(max_features=100)
        X = vectorizer.fit_transform(texts).toarray()
        y = np.array(labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results['dataset_size'] = len(texts)
        results['features'] = X.shape[1]
        
        # Test with ML Toolbox
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            
            start_time = time.time()
            simple = toolbox.algorithms.get_simple_ml_tasks()
            result = simple.train_classifier(X_train, y_train)
            training_time = time.time() - start_time
            
            y_pred = result['model'].predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['models'].append({
                'name': 'ML Toolbox - Simple ML',
                'accuracy': accuracy,
                'training_time': training_time,
                'status': 'success'
            })
        except Exception as e:
            results['models'].append({
                'name': 'ML Toolbox - Simple ML',
                'error': str(e),
                'status': 'failed'
            })
        
        return results
    
    def benchmark_mnist_classification(self) -> Dict[str, Any]:
        """Benchmark: MNIST image classification"""
        results = {
            'name': 'MNIST Classification',
            'difficulty': 'Medium-Hard',
            'models': []
        }
        
        # Use smaller subset for speed
        try:
            from sklearn.datasets import fetch_openml
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            print("  Loading MNIST dataset...")
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            X, y = mnist.data[:5000], mnist.target[:5000].astype(int)  # Use subset
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            results['dataset_size'] = len(X)
            results['features'] = X.shape[1]
            results['classes'] = len(np.unique(y))
            
            # Test with ML Toolbox
            try:
                from ml_toolbox import MLToolbox
                toolbox = MLToolbox()
                
                print("  Testing ML Toolbox...")
                start_time = time.time()
                simple = toolbox.algorithms.get_simple_ml_tasks()
                result = simple.train_classifier(X_train, y_train, model_type='random_forest')
                training_time = time.time() - start_time
                
                y_pred = result['model'].predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results['models'].append({
                    'name': 'ML Toolbox - Simple ML',
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'status': 'success'
                })
            except Exception as e:
                results['models'].append({
                    'name': 'ML Toolbox - Simple ML',
                    'error': str(e),
                    'status': 'failed'
                })
            
            # Test Deep Learning if available
            if self.torch_available:
                try:
                    from ml_toolbox import MLToolbox
                    toolbox = MLToolbox()
                    
                    print("  Testing Deep Learning...")
                    # Note: This would require data loaders, simplified for now
                    results['models'].append({
                        'name': 'ML Toolbox - Deep Learning',
                        'note': 'Requires data loaders - not tested in this benchmark',
                        'status': 'skipped'
                    })
                except Exception as e:
                    pass
        
        except Exception as e:
            results['error'] = f"Could not load MNIST: {str(e)}"
            results['models'].append({
                'name': 'MNIST Test',
                'error': str(e),
                'status': 'failed'
            })
        
        return results
    
    def benchmark_time_series(self) -> Dict[str, Any]:
        """Benchmark: Time series forecasting"""
        results = {
            'name': 'Time Series Forecasting',
            'difficulty': 'Medium',
            'models': []
        }
        
        # Create synthetic time series
        np.random.seed(42)
        n_samples = 1000
        time_points = np.arange(n_samples)
        trend = 0.01 * time_points
        seasonal = 10 * np.sin(2 * np.pi * time_points / 100)
        noise = np.random.randn(n_samples) * 2
        y = trend + seasonal + noise
        
        # Create features (lagged values)
        X = np.column_stack([
            np.roll(y, 1),
            np.roll(y, 2),
            np.roll(y, 3),
            time_points
        ])
        
        # Remove first few samples (NaN from roll)
        X = X[3:]
        y = y[3:]
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results['dataset_size'] = len(X)
        results['features'] = X.shape[1]
        
        # Test with ML Toolbox
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            
            start_time = time.time()
            simple = toolbox.algorithms.get_simple_ml_tasks()
            result = simple.train_regressor(X_train, y_train)
            training_time = time.time() - start_time
            
            y_pred = result['model'].predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results['models'].append({
                'name': 'ML Toolbox - Simple ML',
                'mse': mse,
                'r2_score': r2,
                'training_time': training_time,
                'status': 'success'
            })
        except Exception as e:
            results['models'].append({
                'name': 'ML Toolbox - Simple ML',
                'error': str(e),
                'status': 'failed'
            })
        
        return results
    
    def benchmark_large_scale(self) -> Dict[str, Any]:
        """Benchmark: Large-scale dataset"""
        results = {
            'name': 'Large-scale Dataset',
            'difficulty': 'Hard',
            'models': []
        }
        
        # Create large synthetic dataset
        n_samples = 10000
        n_features = 100
        
        X = np.random.rand(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 1.0).astype(int)
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results['dataset_size'] = n_samples
        results['features'] = n_features
        
        # Test with ML Toolbox
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            
            print("  Testing ML Toolbox on large dataset...")
            start_time = time.time()
            simple = toolbox.algorithms.get_simple_ml_tasks()
            result = simple.train_classifier(X_train, y_train, model_type='random_forest')
            training_time = time.time() - start_time
            
            y_pred = result['model'].predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['models'].append({
                'name': 'ML Toolbox - Simple ML',
                'accuracy': accuracy,
                'training_time': training_time,
                'status': 'success'
            })
        except Exception as e:
            results['models'].append({
                'name': 'ML Toolbox - Simple ML',
                'error': str(e),
                'status': 'failed'
            })
        
        # Test AutoML
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            
            print("  Testing AutoML...")
            start_time = time.time()
            automl = toolbox.algorithms.get_automl_framework()
            result = automl.automl_pipeline(X_train, y_train, task_type='classification', time_budget=30)
            automl_time = time.time() - start_time
            
            if 'best_model' in result:
                y_pred = result['best_model'].predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results['models'].append({
                    'name': 'ML Toolbox - AutoML',
                    'accuracy': accuracy,
                    'training_time': automl_time,
                    'status': 'success'
                })
        except Exception as e:
            results['models'].append({
                'name': 'ML Toolbox - AutoML',
                'error': str(e),
                'status': 'failed'
            })
        
        return results
    
    def _generate_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            'total_benchmarks': len(all_results['benchmarks']),
            'successful_tests': 0,
            'failed_tests': 0,
            'performance_stats': {},
            'improvements': []
        }
        
        # Analyze results
        all_times = []
        all_accuracies = []
        
        for benchmark in all_results['benchmarks']:
            for model in benchmark.get('models', []):
                if model.get('status') == 'success':
                    summary['successful_tests'] += 1
                    
                    if 'training_time' in model:
                        all_times.append(model['training_time'])
                    
                    if 'accuracy' in model:
                        all_accuracies.append(model['accuracy'])
                elif model.get('status') == 'failed':
                    summary['failed_tests'] += 1
        
        # Performance stats
        if all_times:
            summary['performance_stats']['avg_training_time'] = np.mean(all_times)
            summary['performance_stats']['min_training_time'] = np.min(all_times)
            summary['performance_stats']['max_training_time'] = np.max(all_times)
        
        if all_accuracies:
            summary['performance_stats']['avg_accuracy'] = np.mean(all_accuracies)
            summary['performance_stats']['min_accuracy'] = np.min(all_accuracies)
            summary['performance_stats']['max_accuracy'] = np.max(all_accuracies)
        
        # Identify improvements
        improvements = []
        
        # Check for errors
        if summary['failed_tests'] > 0:
            improvements.append({
                'category': 'Reliability',
                'issue': f'{summary["failed_tests"]} tests failed',
                'recommendation': 'Fix error handling and improve robustness'
            })
        
        # Check performance
        if all_times and np.mean(all_times) > 10:
            improvements.append({
                'category': 'Performance',
                'issue': 'Training times are high',
                'recommendation': 'Optimize algorithms and add caching'
            })
        
        # Check accuracy
        if all_accuracies and np.mean(all_accuracies) < 0.8:
            improvements.append({
                'category': 'Accuracy',
                'issue': 'Model accuracy could be improved',
                'recommendation': 'Add hyperparameter tuning and better preprocessing'
            })
        
        summary['improvements'] = improvements
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str = "benchmark_results.json"):
        """Save benchmark results"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return output_path
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable report"""
        report = []
        report.append("=" * 80)
        report.append("ML Toolbox Benchmark Report")
        report.append("=" * 80)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append("")
        
        # Summary
        summary = results.get('summary', {})
        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"Total Benchmarks: {summary.get('total_benchmarks', 0)}")
        report.append(f"Successful Tests: {summary.get('successful_tests', 0)}")
        report.append(f"Failed Tests: {summary.get('failed_tests', 0)}")
        report.append("")
        
        # Performance Stats
        perf = summary.get('performance_stats', {})
        if perf:
            report.append("PERFORMANCE STATISTICS")
            report.append("-" * 80)
            if 'avg_training_time' in perf:
                report.append(f"Average Training Time: {perf['avg_training_time']:.4f}s")
                report.append(f"Min Training Time: {perf['min_training_time']:.4f}s")
                report.append(f"Max Training Time: {perf['max_training_time']:.4f}s")
            if 'avg_accuracy' in perf:
                report.append(f"Average Accuracy: {perf['avg_accuracy']:.4f}")
                report.append(f"Min Accuracy: {perf['min_accuracy']:.4f}")
                report.append(f"Max Accuracy: {perf['max_accuracy']:.4f}")
            report.append("")
        
        # Detailed Results
        report.append("DETAILED RESULTS")
        report.append("-" * 80)
        for benchmark in results.get('benchmarks', []):
            report.append(f"\n{benchmark.get('name', 'Unknown')} ({benchmark.get('difficulty', 'Unknown')})")
            report.append(f"  Dataset Size: {benchmark.get('dataset_size', 'N/A')}")
            report.append(f"  Features: {benchmark.get('features', 'N/A')}")
            
            for model in benchmark.get('models', []):
                report.append(f"  {model.get('name', 'Unknown')}:")
                if model.get('status') == 'success':
                    if 'accuracy' in model:
                        report.append(f"    Accuracy: {model['accuracy']:.4f}")
                    if 'r2_score' in model:
                        report.append(f"    R² Score: {model['r2_score']:.4f}")
                    if 'mse' in model:
                        report.append(f"    MSE: {model['mse']:.4f}")
                    if 'training_time' in model:
                        report.append(f"    Training Time: {model['training_time']:.4f}s")
                elif model.get('status') == 'failed':
                    report.append(f"    ERROR: {model.get('error', 'Unknown error')}")
        
        # Improvements
        improvements = summary.get('improvements', [])
        if improvements:
            report.append("")
            report.append("AREAS FOR IMPROVEMENT")
            report.append("-" * 80)
            for imp in improvements:
                report.append(f"\n{imp.get('category', 'General')}:")
                report.append(f"  Issue: {imp.get('issue', 'N/A')}")
                report.append(f"  Recommendation: {imp.get('recommendation', 'N/A')}")
        
        return "\n".join(report)


if __name__ == '__main__':
    suite = MLBenchmarkSuite()
    results = suite.run_all_benchmarks()
    
    # Save results
    suite.save_results(results, 'benchmark_results.json')
    
    # Generate and print report
    report = suite.generate_report(results)
    print("\n" + "=" * 80)
    print("BENCHMARK REPORT")
    print("=" * 80)
    print(report)
    
    # Save report
    with open('benchmark_report.txt', 'w') as f:
        f.write(report)
    
    print("\nResults saved to:")
    print("  - benchmark_results.json")
    print("  - benchmark_report.txt")
