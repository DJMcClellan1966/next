"""
Performance Benchmark Script
Compares ML Toolbox performance with scikit-learn
Run this to see performance gaps
"""
import sys
from pathlib import Path
import time
import numpy as np
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import scikit-learn
try:
    from sklearn.datasets import make_classification, make_regression, make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, r2_score, silhouette_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

# Try to import ML Toolbox
try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    print("Error: ML Toolbox not available")

def benchmark_classification():
    """Benchmark classification algorithms"""
    print("="*80)
    print("CLASSIFICATION BENCHMARKS")
    print("="*80)
    
    if not SKLEARN_AVAILABLE or not TOOLBOX_AVAILABLE:
        print("Cannot run benchmarks - missing dependencies")
        return {}
    
    results = {}
    
    # Generate data
    print("\nGenerating classification dataset...")
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Logistic Regression
    print("\n1. Logistic Regression")
    print("-" * 80)
    
    # Scikit-learn
    sk_model = LogisticRegression(random_state=42, max_iter=1000)
    start = time.time()
    sk_model.fit(X_train, y_train)
    sk_fit_time = time.time() - start
    
    start = time.time()
    sk_pred = sk_model.predict(X_test)
    sk_predict_time = time.time() - start
    
    sk_accuracy = accuracy_score(y_test, sk_pred)
    
    # ML Toolbox
    toolbox = MLToolbox()
    start = time.time()
    try:
        result = toolbox.fit(X_train, y_train, task_type='classification')
        tb_fit_time = time.time() - start
        
        start = time.time()
        tb_pred = toolbox.predict(result['model'], X_test)
        tb_predict_time = time.time() - start
        
        tb_accuracy = accuracy_score(y_test, tb_pred)
        
        results['logistic_regression'] = {
            'sklearn': {
                'fit_time': sk_fit_time,
                'predict_time': sk_predict_time,
                'accuracy': sk_accuracy
            },
            'toolbox': {
                'fit_time': tb_fit_time,
                'predict_time': tb_predict_time,
                'accuracy': tb_accuracy
            },
            'speedup': {
                'fit': sk_fit_time / tb_fit_time if tb_fit_time > 0 else 0,
                'predict': sk_predict_time / tb_predict_time if tb_predict_time > 0 else 0
            },
            'accuracy_diff': tb_accuracy - sk_accuracy
        }
        
        print(f"  Scikit-learn: Fit={sk_fit_time:.4f}s, Predict={sk_predict_time:.4f}s, Accuracy={sk_accuracy:.4f}")
        print(f"  ML Toolbox:   Fit={tb_fit_time:.4f}s, Predict={tb_predict_time:.4f}s, Accuracy={tb_accuracy:.4f}")
        print(f"  Speedup:      Fit={results['logistic_regression']['speedup']['fit']:.2f}x, Predict={results['logistic_regression']['speedup']['predict']:.2f}x")
        print(f"  Accuracy Diff: {results['logistic_regression']['accuracy_diff']:.4f}")
    except Exception as e:
        print(f"  ML Toolbox Error: {e}")
        results['logistic_regression'] = {'error': str(e)}
    
    # Random Forest
    print("\n2. Random Forest")
    print("-" * 80)
    
    # Scikit-learn
    sk_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    start = time.time()
    sk_model.fit(X_train, y_train)
    sk_fit_time = time.time() - start
    
    start = time.time()
    sk_pred = sk_model.predict(X_test)
    sk_predict_time = time.time() - start
    
    sk_accuracy = accuracy_score(y_test, sk_pred)
    
    # ML Toolbox
    start = time.time()
    try:
        result = toolbox.fit(X_train, y_train, task_type='classification', model_type='random_forest')
        tb_fit_time = time.time() - start
        
        start = time.time()
        tb_pred = toolbox.predict(result['model'], X_test)
        tb_predict_time = time.time() - start
        
        tb_accuracy = accuracy_score(y_test, tb_pred)
        
        results['random_forest'] = {
            'sklearn': {
                'fit_time': sk_fit_time,
                'predict_time': sk_predict_time,
                'accuracy': sk_accuracy
            },
            'toolbox': {
                'fit_time': tb_fit_time,
                'predict_time': tb_predict_time,
                'accuracy': tb_accuracy
            },
            'speedup': {
                'fit': sk_fit_time / tb_fit_time if tb_fit_time > 0 else 0,
                'predict': sk_predict_time / tb_predict_time if tb_predict_time > 0 else 0
            },
            'accuracy_diff': tb_accuracy - sk_accuracy
        }
        
        print(f"  Scikit-learn: Fit={sk_fit_time:.4f}s, Predict={sk_predict_time:.4f}s, Accuracy={sk_accuracy:.4f}")
        print(f"  ML Toolbox:   Fit={tb_fit_time:.4f}s, Predict={tb_predict_time:.4f}s, Accuracy={tb_accuracy:.4f}")
        print(f"  Speedup:      Fit={results['random_forest']['speedup']['fit']:.2f}x, Predict={results['random_forest']['speedup']['predict']:.2f}x")
        print(f"  Accuracy Diff: {results['random_forest']['accuracy_diff']:.4f}")
    except Exception as e:
        print(f"  ML Toolbox Error: {e}")
        results['random_forest'] = {'error': str(e)}
    
    return results

def benchmark_regression():
    """Benchmark regression algorithms"""
    print("\n" + "="*80)
    print("REGRESSION BENCHMARKS")
    print("="*80)
    
    if not SKLEARN_AVAILABLE or not TOOLBOX_AVAILABLE:
        print("Cannot run benchmarks - missing dependencies")
        return {}
    
    results = {}
    
    # Generate data
    print("\nGenerating regression dataset...")
    X, y = make_regression(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    print("\n1. Linear Regression")
    print("-" * 80)
    
    # Scikit-learn
    sk_model = LinearRegression()
    start = time.time()
    sk_model.fit(X_train, y_train)
    sk_fit_time = time.time() - start
    
    start = time.time()
    sk_pred = sk_model.predict(X_test)
    sk_predict_time = time.time() - start
    
    sk_r2 = r2_score(y_test, sk_pred)
    
    # ML Toolbox
    toolbox = MLToolbox()
    start = time.time()
    try:
        result = toolbox.fit(X_train, y_train, task_type='regression')
        tb_fit_time = time.time() - start
        
        start = time.time()
        tb_pred = toolbox.predict(result['model'], X_test)
        tb_predict_time = time.time() - start
        
        tb_r2 = r2_score(y_test, tb_pred)
        
        results['linear_regression'] = {
            'sklearn': {
                'fit_time': sk_fit_time,
                'predict_time': sk_predict_time,
                'r2': sk_r2
            },
            'toolbox': {
                'fit_time': tb_fit_time,
                'predict_time': tb_predict_time,
                'r2': tb_r2
            },
            'speedup': {
                'fit': sk_fit_time / tb_fit_time if tb_fit_time > 0 else 0,
                'predict': sk_predict_time / tb_predict_time if tb_predict_time > 0 else 0
            },
            'r2_diff': tb_r2 - sk_r2
        }
        
        print(f"  Scikit-learn: Fit={sk_fit_time:.4f}s, Predict={sk_predict_time:.4f}s, R²={sk_r2:.4f}")
        print(f"  ML Toolbox:   Fit={tb_fit_time:.4f}s, Predict={tb_predict_time:.4f}s, R²={tb_r2:.4f}")
        print(f"  Speedup:      Fit={results['linear_regression']['speedup']['fit']:.2f}x, Predict={results['linear_regression']['speedup']['predict']:.2f}x")
        print(f"  R² Diff:      {results['linear_regression']['r2_diff']:.4f}")
    except Exception as e:
        print(f"  ML Toolbox Error: {e}")
        results['linear_regression'] = {'error': str(e)}
    
    return results

def benchmark_clustering():
    """Benchmark clustering algorithms"""
    print("\n" + "="*80)
    print("CLUSTERING BENCHMARKS")
    print("="*80)
    
    if not SKLEARN_AVAILABLE or not TOOLBOX_AVAILABLE:
        print("Cannot run benchmarks - missing dependencies")
        return {}
    
    results = {}
    
    # Generate data
    print("\nGenerating clustering dataset...")
    X, y_true = make_blobs(
        n_samples=5000,
        n_features=10,
        centers=5,
        random_state=42
    )
    
    # K-Means
    print("\n1. K-Means")
    print("-" * 80)
    
    # Scikit-learn
    sk_model = KMeans(n_clusters=5, random_state=42, n_init=10)
    start = time.time()
    sk_labels = sk_model.fit_predict(X)
    sk_fit_time = time.time() - start
    
    sk_silhouette = silhouette_score(X, sk_labels)
    
    # ML Toolbox
    toolbox = MLToolbox()
    start = time.time()
    try:
        result = toolbox.fit(X, None, task_type='clustering', n_clusters=5)
        tb_fit_time = time.time() - start
        
        tb_labels = toolbox.predict(result['model'], X)
        tb_silhouette = silhouette_score(X, tb_labels)
        
        results['kmeans'] = {
            'sklearn': {
                'fit_time': sk_fit_time,
                'silhouette': sk_silhouette
            },
            'toolbox': {
                'fit_time': tb_fit_time,
                'silhouette': tb_silhouette
            },
            'speedup': {
                'fit': sk_fit_time / tb_fit_time if tb_fit_time > 0 else 0
            },
            'silhouette_diff': tb_silhouette - sk_silhouette
        }
        
        print(f"  Scikit-learn: Fit={sk_fit_time:.4f}s, Silhouette={sk_silhouette:.4f}")
        print(f"  ML Toolbox:   Fit={tb_fit_time:.4f}s, Silhouette={tb_silhouette:.4f}")
        print(f"  Speedup:      Fit={results['kmeans']['speedup']['fit']:.2f}x")
        print(f"  Silhouette Diff: {results['kmeans']['silhouette_diff']:.4f}")
    except Exception as e:
        print(f"  ML Toolbox Error: {e}")
        results['kmeans'] = {'error': str(e)}
    
    return results

def run_all_benchmarks():
    """Run all benchmarks"""
    print("="*80)
    print("ML TOOLBOX vs SCIKIT-LEARN PERFORMANCE BENCHMARKS")
    print("="*80)
    
    all_results = {
        'classification': benchmark_classification(),
        'regression': benchmark_regression(),
        'clustering': benchmark_clustering()
    }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_tests = 0
    faster_tests = 0
    slower_tests = 0
    equal_tests = 0
    
    for category, results in all_results.items():
        for algo_name, algo_results in results.items():
            if 'error' not in algo_results and 'speedup' in algo_results:
                total_tests += 1
                if 'fit' in algo_results['speedup']:
                    speedup = algo_results['speedup']['fit']
                    if speedup > 1.1:
                        faster_tests += 1
                    elif speedup < 0.9:
                        slower_tests += 1
                    else:
                        equal_tests += 1
    
    print(f"Total tests: {total_tests}")
    print(f"Faster than sklearn: {faster_tests} ({faster_tests/total_tests*100:.1f}%)")
    print(f"Slower than sklearn: {slower_tests} ({slower_tests/total_tests*100:.1f}%)")
    print(f"Equal performance: {equal_tests} ({equal_tests/total_tests*100:.1f}%)")
    
    # Save results
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\nResults saved to: benchmark_results.json")
    
    return all_results

if __name__ == '__main__':
    results = run_all_benchmarks()
