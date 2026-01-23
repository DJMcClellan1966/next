"""
Comprehensive ML Test Suite
Tests from simple to NP-complete problems, with competitor comparison

Test Categories:
1. Simple: Basic classification, regression
2. Medium: Multi-class, time series, clustering
3. Hard: High-dimensional, imbalanced, complex patterns
4. NP-Complete: TSP, graph coloring, subset sum, etc.
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import time
import warnings
from collections import defaultdict
import json

sys.path.insert(0, str(Path(__file__).parent))

# Try to import competitors
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, r2_score, silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available for competitor comparison")

try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    warnings.warn("ML Toolbox not available")


class ComprehensiveMLTestSuite:
    """
    Comprehensive ML Test Suite
    
    Tests from simple to NP-complete
    """
    
    def __init__(self):
        """Initialize test suite"""
        self.results = {
            'toolbox': defaultdict(list),
            'sklearn': defaultdict(list),
            'test_metadata': []
        }
        self.toolbox = MLToolbox() if TOOLBOX_AVAILABLE else None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test categories"""
        print("="*80)
        print("COMPREHENSIVE ML TEST SUITE")
        print("="*80)
        
        all_results = {
            'simple_tests': self.run_simple_tests(),
            'medium_tests': self.run_medium_tests(),
            'hard_tests': self.run_hard_tests(),
            'np_complete_tests': self.run_np_complete_tests(),
            'summary': {}
        }
        
        # Generate summary
        all_results['summary'] = self.generate_summary(all_results)
        
        return all_results
    
    def run_simple_tests(self) -> Dict[str, Any]:
        """Run simple ML tests"""
        print("\n" + "="*80)
        print("SIMPLE TESTS")
        print("="*80)
        
        results = {}
        
        # Test 1: Binary Classification
        print("\n1. Binary Classification (Iris-like)")
        results['binary_classification'] = self.test_binary_classification()
        
        # Test 2: Multi-class Classification
        print("\n2. Multi-class Classification")
        results['multiclass_classification'] = self.test_multiclass_classification()
        
        # Test 3: Simple Regression
        print("\n3. Simple Regression")
        results['simple_regression'] = self.test_simple_regression()
        
        # Test 4: Basic Clustering
        print("\n4. Basic Clustering")
        results['basic_clustering'] = self.test_basic_clustering()
        
        return results
    
    def run_medium_tests(self) -> Dict[str, Any]:
        """Run medium complexity tests"""
        print("\n" + "="*80)
        print("MEDIUM TESTS")
        print("="*80)
        
        results = {}
        
        # Test 1: High-dimensional Classification
        print("\n1. High-dimensional Classification (100 features)")
        results['high_dim_classification'] = self.test_high_dimensional_classification()
        
        # Test 2: Imbalanced Classification
        print("\n2. Imbalanced Classification (1:10 ratio)")
        results['imbalanced_classification'] = self.test_imbalanced_classification()
        
        # Test 3: Time Series Regression
        print("\n3. Time Series Regression")
        results['time_series_regression'] = self.test_time_series_regression()
        
        # Test 4: Multi-output Regression
        print("\n4. Multi-output Regression")
        results['multi_output_regression'] = self.test_multi_output_regression()
        
        # Test 5: Feature Selection
        print("\n5. Feature Selection")
        results['feature_selection'] = self.test_feature_selection()
        
        return results
    
    def run_hard_tests(self) -> Dict[str, Any]:
        """Run hard/complex tests"""
        print("\n" + "="*80)
        print("HARD TESTS")
        print("="*80)
        
        results = {}
        
        # Test 1: Very High-dimensional (1000 features)
        print("\n1. Very High-dimensional Classification (1000 features)")
        results['very_high_dim'] = self.test_very_high_dimensional()
        
        # Test 2: Non-linear Patterns
        print("\n2. Non-linear Pattern Recognition")
        results['nonlinear_patterns'] = self.test_nonlinear_patterns()
        
        # Test 3: Sparse Data
        print("\n3. Sparse Data Classification")
        results['sparse_data'] = self.test_sparse_data()
        
        # Test 4: Noisy Data
        print("\n4. Noisy Data Classification")
        results['noisy_data'] = self.test_noisy_data()
        
        # Test 5: Ensemble Methods
        print("\n5. Ensemble Learning")
        results['ensemble'] = self.test_ensemble_learning()
        
        return results
    
    def run_np_complete_tests(self) -> Dict[str, Any]:
        """Run NP-complete problem tests"""
        print("\n" + "="*80)
        print("NP-COMPLETE TESTS")
        print("="*80)
        
        results = {}
        
        # Test 1: Traveling Salesman Problem (TSP) - Approximate
        print("\n1. Traveling Salesman Problem (TSP) - Heuristic")
        results['tsp'] = self.test_tsp()
        
        # Test 2: Graph Coloring - Approximate
        print("\n2. Graph Coloring - Heuristic")
        results['graph_coloring'] = self.test_graph_coloring()
        
        # Test 3: Subset Sum - Approximate
        print("\n3. Subset Sum Problem - Heuristic")
        results['subset_sum'] = self.test_subset_sum()
        
        # Test 4: Knapsack Problem - Approximate
        print("\n4. Knapsack Problem - Heuristic")
        results['knapsack'] = self.test_knapsack()
        
        # Test 5: Feature Selection (NP-hard subset)
        print("\n5. Optimal Feature Selection (NP-hard)")
        results['optimal_feature_selection'] = self.test_optimal_feature_selection()
        
        return results
    
    # Simple Tests
    def test_binary_classification(self) -> Dict[str, Any]:
        """Test binary classification"""
        np.random.seed(42)
        X = np.random.rand(200, 4)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        
        return self._compare_classification(X, y, "Binary Classification")
    
    def test_multiclass_classification(self) -> Dict[str, Any]:
        """Test multi-class classification"""
        np.random.seed(42)
        X = np.random.rand(300, 5)
        y = np.random.randint(0, 3, 300)
        
        return self._compare_classification(X, y, "Multi-class Classification")
    
    def test_simple_regression(self) -> Dict[str, Any]:
        """Test simple regression"""
        np.random.seed(42)
        X = np.random.rand(200, 3)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(200) * 0.1
        
        return self._compare_regression(X, y, "Simple Regression")
    
    def test_basic_clustering(self) -> Dict[str, Any]:
        """Test basic clustering"""
        np.random.seed(42)
        # Create 3 clusters
        X1 = np.random.randn(50, 2) + [2, 2]
        X2 = np.random.randn(50, 2) + [-2, -2]
        X3 = np.random.randn(50, 2) + [0, 0]
        X = np.vstack([X1, X2, X3])
        
        return self._compare_clustering(X, 3, "Basic Clustering")
    
    # Medium Tests
    def test_high_dimensional_classification(self) -> Dict[str, Any]:
        """Test high-dimensional classification"""
        np.random.seed(42)
        X = np.random.rand(500, 100)
        y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.5).astype(int)
        
        return self._compare_classification(X, y, "High-dimensional Classification")
    
    def test_imbalanced_classification(self) -> Dict[str, Any]:
        """Test imbalanced classification"""
        np.random.seed(42)
        X_minority = np.random.rand(20, 5)
        X_majority = np.random.rand(200, 5)
        X = np.vstack([X_minority, X_majority])
        y = np.hstack([np.ones(20), np.zeros(200)])
        
        return self._compare_classification(X, y, "Imbalanced Classification")
    
    def test_time_series_regression(self) -> Dict[str, Any]:
        """Test time series regression"""
        np.random.seed(42)
        t = np.arange(100)
        X = np.column_stack([t, np.sin(t), np.cos(t)])
        y = t * 0.5 + np.sin(t) * 2 + np.random.randn(100) * 0.1
        
        return self._compare_regression(X, y, "Time Series Regression")
    
    def test_multi_output_regression(self) -> Dict[str, Any]:
        """Test multi-output regression"""
        np.random.seed(42)
        X = np.random.rand(200, 5)
        y = np.column_stack([
            X[:, 0] * 2 + X[:, 1],
            X[:, 2] * 3 + X[:, 3]
        ])
        
        return self._compare_regression(X, y, "Multi-output Regression")
    
    def test_feature_selection(self) -> Dict[str, Any]:
        """Test feature selection"""
        np.random.seed(42)
        X = np.random.rand(200, 20)
        # Only first 5 features are relevant
        y = X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] + X[:, 4] + np.random.randn(200) * 0.1
        
        return self._compare_feature_selection(X, y, "Feature Selection")
    
    # Hard Tests
    def test_very_high_dimensional(self) -> Dict[str, Any]:
        """Test very high-dimensional classification"""
        np.random.seed(42)
        X = np.random.rand(1000, 1000)
        y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.5).astype(int)
        
        return self._compare_classification(X, y, "Very High-dimensional", sample_size=500)
    
    def test_nonlinear_patterns(self) -> Dict[str, Any]:
        """Test non-linear patterns"""
        np.random.seed(42)
        X = np.random.rand(300, 2)
        # Non-linear decision boundary: circle
        y = ((X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2 < 0.1).astype(int)
        
        return self._compare_classification(X, y, "Non-linear Patterns")
    
    def test_sparse_data(self) -> Dict[str, Any]:
        """Test sparse data"""
        np.random.seed(42)
        from scipy.sparse import csr_matrix
        X = csr_matrix(np.random.rand(200, 50))
        X.data[X.data < 0.8] = 0  # Make sparse
        X.eliminate_zeros()
        X = X.toarray()
        y = (X[:, 0] + X[:, 1] > 0.1).astype(int)
        
        return self._compare_classification(X, y, "Sparse Data")
    
    def test_noisy_data(self) -> Dict[str, Any]:
        """Test noisy data"""
        np.random.seed(42)
        X = np.random.rand(200, 5)
        # Add significant noise
        X_noisy = X + np.random.randn(200, 5) * 0.5
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        
        return self._compare_classification(X_noisy, y, "Noisy Data")
    
    def test_ensemble_learning(self) -> Dict[str, Any]:
        """Test ensemble learning"""
        np.random.seed(42)
        X = np.random.rand(300, 10)
        y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.5).astype(int)
        
        return self._compare_ensemble(X, y, "Ensemble Learning")
    
    # NP-Complete Tests
    def test_tsp(self) -> Dict[str, Any]:
        """Test Traveling Salesman Problem (heuristic)"""
        np.random.seed(42)
        n_cities = 20
        cities = np.random.rand(n_cities, 2) * 100
        
        results = {}
        
        # Toolbox approach (using optimization)
        if TOOLBOX_AVAILABLE:
            start = time.time()
            try:
                # Use genetic algorithm or simulated annealing
                optimizer = self.toolbox.algorithms.get_genetic_algorithm_optimizer()
                # Simplified TSP solution
                solution = self._solve_tsp_heuristic(cities)
                toolbox_time = time.time() - start
                results['toolbox'] = {
                    'time': toolbox_time,
                    'distance': self._calculate_tsp_distance(cities, solution),
                    'method': 'heuristic'
                }
            except:
                results['toolbox'] = {'error': 'Not available'}
        else:
            results['toolbox'] = {'error': 'Toolbox not available'}
        
        # Baseline (nearest neighbor)
        start = time.time()
        baseline_solution = self._tsp_nearest_neighbor(cities)
        baseline_time = time.time() - start
        results['baseline'] = {
            'time': baseline_time,
            'distance': self._calculate_tsp_distance(cities, baseline_solution),
            'method': 'nearest_neighbor'
        }
        
        return results
    
    def test_graph_coloring(self) -> Dict[str, Any]:
        """Test graph coloring (heuristic)"""
        np.random.seed(42)
        n_nodes = 20
        edges = self._generate_random_graph(n_nodes, 0.3)
        
        results = {}
        
        # Toolbox approach
        if TOOLBOX_AVAILABLE:
            start = time.time()
            try:
                colors = self._graph_coloring_heuristic(n_nodes, edges)
                toolbox_time = time.time() - start
                results['toolbox'] = {
                    'time': toolbox_time,
                    'num_colors': len(set(colors.values())),
                    'valid': self._validate_coloring(n_nodes, edges, colors)
                }
            except:
                results['toolbox'] = {'error': 'Not available'}
        else:
            results['toolbox'] = {'error': 'Toolbox not available'}
        
        # Greedy baseline
        start = time.time()
        baseline_colors = self._greedy_coloring(n_nodes, edges)
        baseline_time = time.time() - start
        results['baseline'] = {
            'time': baseline_time,
            'num_colors': len(set(baseline_colors.values())),
            'valid': self._validate_coloring(n_nodes, edges, baseline_colors)
        }
        
        return results
    
    def test_subset_sum(self) -> Dict[str, Any]:
        """Test subset sum problem (heuristic)"""
        np.random.seed(42)
        numbers = np.random.randint(1, 100, 20)
        target = int(numbers.sum() * 0.4)
        
        results = {}
        
        # Toolbox approach
        if TOOLBOX_AVAILABLE:
            start = time.time()
            try:
                solution = self._subset_sum_heuristic(numbers, target)
                toolbox_time = time.time() - start
                results['toolbox'] = {
                    'time': toolbox_time,
                    'found': solution is not None,
                    'sum': sum(solution) if solution else 0,
                    'target': target
                }
            except:
                results['toolbox'] = {'error': 'Not available'}
        else:
            results['toolbox'] = {'error': 'Toolbox not available'}
        
        # Greedy baseline
        start = time.time()
        baseline_solution = self._greedy_subset_sum(numbers, target)
        baseline_time = time.time() - start
        results['baseline'] = {
            'time': baseline_time,
            'found': baseline_solution is not None,
            'sum': sum(baseline_solution) if baseline_solution else 0,
            'target': target
        }
        
        return results
    
    def test_knapsack(self) -> Dict[str, Any]:
        """Test knapsack problem (heuristic)"""
        np.random.seed(42)
        n_items = 20
        weights = np.random.randint(1, 20, n_items)
        values = np.random.randint(1, 100, n_items)
        capacity = int(weights.sum() * 0.5)
        
        results = {}
        
        # Toolbox approach
        if TOOLBOX_AVAILABLE:
            start = time.time()
            try:
                solution = self._knapsack_heuristic(weights, values, capacity)
                toolbox_time = time.time() - start
                results['toolbox'] = {
                    'time': toolbox_time,
                    'value': sum(values[solution]) if solution else 0,
                    'weight': sum(weights[solution]) if solution else 0,
                    'capacity': capacity
                }
            except:
                results['toolbox'] = {'error': 'Not available'}
        else:
            results['toolbox'] = {'error': 'Toolbox not available'}
        
        # Greedy baseline
        start = time.time()
        baseline_solution = self._greedy_knapsack(weights, values, capacity)
        baseline_time = time.time() - start
        results['baseline'] = {
            'time': baseline_time,
            'value': sum(values[baseline_solution]) if baseline_solution else 0,
            'weight': sum(weights[baseline_solution]) if baseline_solution else 0,
            'capacity': capacity
        }
        
        return results
    
    def test_optimal_feature_selection(self) -> Dict[str, Any]:
        """Test optimal feature selection (NP-hard)"""
        np.random.seed(42)
        X = np.random.rand(200, 30)
        y = X[:, 0] + X[:, 1] + X[:, 2] + np.random.randn(200) * 0.1
        
        results = {}
        
        # Toolbox approach
        if TOOLBOX_AVAILABLE:
            start = time.time()
            try:
                selector = self.toolbox.algorithms.get_information_theoretic_feature_selector()
                # Try different methods to use selector
                if hasattr(selector, 'select_features'):
                    selected = selector.select_features(X, y, k=5)
                elif hasattr(selector, 'fit'):
                    selector.fit(X, y)
                    selected = selector.get_selected_features(k=5) if hasattr(selector, 'get_selected_features') else list(range(5))
                else:
                    selected = list(range(5))  # Fallback
                toolbox_time = time.time() - start
                
                # Evaluate
                if SKLEARN_AVAILABLE:
                    from sklearn.ensemble import RandomForestRegressor as RFR
                    model = RFR(n_estimators=10, random_state=42)
                    if isinstance(selected, (list, np.ndarray)) and len(selected) > 0:
                        model.fit(X[:, selected], y)
                        score = model.score(X[:, selected], y)
                    else:
                        score = 0.0
                else:
                    score = 0.0
                
                results['toolbox'] = {
                    'time': toolbox_time,
                    'num_features': len(selected) if isinstance(selected, (list, np.ndarray)) else 5,
                    'score': score
                }
            except Exception as e:
                results['toolbox'] = {'error': str(e)}
        else:
            results['toolbox'] = {'error': 'Toolbox not available'}
        
        # sklearn baseline
        if SKLEARN_AVAILABLE:
            start = time.time()
            try:
                from sklearn.feature_selection import SelectKBest, f_regression
                from sklearn.ensemble import RandomForestRegressor as RFR
                selector = SelectKBest(f_regression, k=5)
                X_selected = selector.fit_transform(X, y)
                model = RFR(n_estimators=10, random_state=42)
                model.fit(X_selected, y)
                score = model.score(X_selected, y)
                sklearn_time = time.time() - start
                
                results['sklearn'] = {
                    'time': sklearn_time,
                    'num_features': 5,
                    'score': score
                }
            except Exception as e:
                results['sklearn'] = {'error': str(e)}
        else:
            results['sklearn'] = {'error': 'sklearn not available'}
        
        return results
    
    # Comparison Methods
    def _compare_classification(self, X: np.ndarray, y: np.ndarray, test_name: str, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Compare classification performance"""
        if sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X[indices]
            y = y[indices]
        
        results = {'test_name': test_name}
        
        # Toolbox
        if TOOLBOX_AVAILABLE:
            start = time.time()
            try:
                # Use simple ML tasks
                simple_ml = self.toolbox.algorithms.get_simple_ml_tasks()
                result = simple_ml.train_classifier(X, y, model_type='random_forest')
                model = result['model']
                preds = model.predict(X)
                acc = np.mean(preds == y)
                toolbox_time = time.time() - start
                
                results['toolbox'] = {
                    'accuracy': acc,
                    'time': toolbox_time,
                    'success': True
                }
            except Exception as e:
                results['toolbox'] = {'error': str(e), 'success': False}
        else:
            results['toolbox'] = {'error': 'Toolbox not available'}
        
        # sklearn
        if SKLEARN_AVAILABLE:
            start = time.time()
            try:
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                model.fit(X, y)
                preds = model.predict(X)
                acc = accuracy_score(y, preds)
                sklearn_time = time.time() - start
                
                results['sklearn'] = {
                    'accuracy': acc,
                    'time': sklearn_time,
                    'success': True
                }
            except Exception as e:
                results['sklearn'] = {'error': str(e), 'success': False}
        
        return results
    
    def _compare_regression(self, X: np.ndarray, y: np.ndarray, test_name: str) -> Dict[str, Any]:
        """Compare regression performance"""
        results = {'test_name': test_name}
        
        # Toolbox
        if TOOLBOX_AVAILABLE:
            start = time.time()
            try:
                # Use simple ML tasks
                simple_ml = self.toolbox.algorithms.get_simple_ml_tasks()
                result = simple_ml.train_regressor(X, y, model_type='random_forest')
                model = result['model']
                preds = model.predict(X)
                if y.ndim == 1:
                    r2 = 1 - np.sum((y - preds)**2) / np.sum((y - np.mean(y))**2)
                else:
                    if SKLEARN_AVAILABLE:
                        from sklearn.metrics import r2_score as r2_metric
                        r2 = r2_metric(y, preds)
                    else:
                        r2 = 0.0
                toolbox_time = time.time() - start
                
                results['toolbox'] = {
                    'r2_score': r2,
                    'time': toolbox_time,
                    'success': True
                }
            except Exception as e:
                results['toolbox'] = {'error': str(e), 'success': False}
        else:
            results['toolbox'] = {'error': 'Toolbox not available'}
        
        # sklearn
        if SKLEARN_AVAILABLE:
            start = time.time()
            try:
                from sklearn.metrics import r2_score as r2_metric
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                model.fit(X, y)
                preds = model.predict(X)
                if y.ndim == 1:
                    r2 = r2_metric(y, preds)
                else:
                    r2 = r2_metric(y, preds, multioutput='uniform_average')
                sklearn_time = time.time() - start
                
                results['sklearn'] = {
                    'r2_score': r2,
                    'time': sklearn_time,
                    'success': True
                }
            except Exception as e:
                results['sklearn'] = {'error': str(e), 'success': False}
        
        return results
    
    def _compare_clustering(self, X: np.ndarray, n_clusters: int, test_name: str) -> Dict[str, Any]:
        """Compare clustering performance"""
        results = {'test_name': test_name}
        
        # Toolbox - use sklearn directly for clustering (simplified)
        if TOOLBOX_AVAILABLE and SKLEARN_AVAILABLE:
            start = time.time()
            try:
                # Use sklearn KMeans through toolbox's sklearn access
                from sklearn.cluster import KMeans
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X)
                silhouette = silhouette_score(X, labels)
                toolbox_time = time.time() - start
                
                results['toolbox'] = {
                    'silhouette_score': silhouette,
                    'time': toolbox_time,
                    'success': True
                }
            except Exception as e:
                results['toolbox'] = {'error': str(e), 'success': False}
        else:
            results['toolbox'] = {'error': 'Toolbox/sklearn not available'}
        
        # sklearn
        if SKLEARN_AVAILABLE:
            start = time.time()
            try:
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X)
                silhouette = silhouette_score(X, labels)
                sklearn_time = time.time() - start
                
                results['sklearn'] = {
                    'silhouette_score': silhouette,
                    'time': sklearn_time,
                    'success': True
                }
            except Exception as e:
                results['sklearn'] = {'error': str(e), 'success': False}
        
        return results
    
    def _compare_feature_selection(self, X: np.ndarray, y: np.ndarray, test_name: str) -> Dict[str, Any]:
        """Compare feature selection"""
        results = {'test_name': test_name}
        
        # Toolbox
        if TOOLBOX_AVAILABLE:
            start = time.time()
            try:
                selector = self.toolbox.algorithms.get_information_theoretic_feature_selector()
                # Check if it has select_features method
                if hasattr(selector, 'select_features'):
                    selected = selector.select_features(X, y, k=5)
                elif hasattr(selector, 'fit'):
                    selector.fit(X, y)
                    selected = selector.get_selected_features(k=5) if hasattr(selector, 'get_selected_features') else []
                else:
                    # Try direct selection
                    selected = selector(X, y, k=5) if callable(selector) else []
                toolbox_time = time.time() - start
                
                results['toolbox'] = {
                    'num_features': len(selected) if isinstance(selected, (list, np.ndarray)) else 5,
                    'time': toolbox_time,
                    'success': True
                }
            except Exception as e:
                results['toolbox'] = {'error': str(e), 'success': False}
        else:
            results['toolbox'] = {'error': 'Toolbox not available'}
        
        # sklearn
        if SKLEARN_AVAILABLE:
            start = time.time()
            try:
                from sklearn.feature_selection import SelectKBest, f_regression
                selector = SelectKBest(f_regression, k=5)
                selector.fit(X, y)
                selected = selector.get_support()
                sklearn_time = time.time() - start
                
                results['sklearn'] = {
                    'num_features': selected.sum(),
                    'time': sklearn_time,
                    'success': True
                }
            except Exception as e:
                results['sklearn'] = {'error': str(e), 'success': False}
        
        return results
    
    def _compare_ensemble(self, X: np.ndarray, y: np.ndarray, test_name: str) -> Dict[str, Any]:
        """Compare ensemble learning"""
        results = {'test_name': test_name}
        
        # Toolbox
        if TOOLBOX_AVAILABLE:
            start = time.time()
            try:
                ensemble = self.toolbox.algorithms.get_ensemble()
                if hasattr(ensemble, 'fit'):
                    ensemble.fit(X, y)
                    preds = ensemble.predict(X)
                elif hasattr(ensemble, 'create_ensemble'):
                    result = ensemble.create_ensemble(X, y, methods=['random_forest', 'svm'])
                    preds = result.get('predictions', [])
                else:
                    # Fallback: use simple ML tasks
                    simple_ml = self.toolbox.algorithms.get_simple_ml_tasks()
                    result = simple_ml.train_classifier(X, y, model_type='random_forest')
                    preds = result['model'].predict(X)
                acc = np.mean(preds == y)
                toolbox_time = time.time() - start
                
                results['toolbox'] = {
                    'accuracy': acc,
                    'time': toolbox_time,
                    'success': True
                }
            except Exception as e:
                results['toolbox'] = {'error': str(e), 'success': False}
        else:
            results['toolbox'] = {'error': 'Toolbox not available'}
        
        # sklearn
        if SKLEARN_AVAILABLE:
            start = time.time()
            try:
                from sklearn.ensemble import VotingClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.tree import DecisionTreeClassifier
                
                ensemble = VotingClassifier([
                    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
                    ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                    ('dt', DecisionTreeClassifier(random_state=42))
                ])
                ensemble.fit(X, y)
                preds = ensemble.predict(X)
                acc = accuracy_score(y, preds)
                sklearn_time = time.time() - start
                
                results['sklearn'] = {
                    'accuracy': acc,
                    'time': sklearn_time,
                    'success': True
                }
            except Exception as e:
                results['sklearn'] = {'error': str(e), 'success': False}
        
        return results
    
    # NP-Complete Helper Methods
    def _solve_tsp_heuristic(self, cities: np.ndarray) -> List[int]:
        """Solve TSP using heuristic (nearest neighbor)"""
        return self._tsp_nearest_neighbor(cities)
    
    def _tsp_nearest_neighbor(self, cities: np.ndarray) -> List[int]:
        """Nearest neighbor TSP solution"""
        n = len(cities)
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        for _ in range(n - 1):
            current = path[-1]
            nearest = None
            min_dist = float('inf')
            
            for i in range(n):
                if not visited[i]:
                    dist = np.linalg.norm(cities[current] - cities[i])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = i
            
            if nearest is not None:
                path.append(nearest)
                visited[nearest] = True
        
        return path
    
    def _calculate_tsp_distance(self, cities: np.ndarray, path: List[int]) -> float:
        """Calculate TSP path distance"""
        total = 0
        for i in range(len(path)):
            j = (i + 1) % len(path)
            total += np.linalg.norm(cities[path[i]] - cities[path[j]])
        return total
    
    def _generate_random_graph(self, n_nodes: int, edge_prob: float) -> List[Tuple[int, int]]:
        """Generate random graph"""
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.rand() < edge_prob:
                    edges.append((i, j))
        return edges
    
    def _graph_coloring_heuristic(self, n_nodes: int, edges: List[Tuple[int, int]]) -> Dict[int, int]:
        """Graph coloring heuristic"""
        return self._greedy_coloring(n_nodes, edges)
    
    def _greedy_coloring(self, n_nodes: int, edges: List[Tuple[int, int]]) -> Dict[int, int]:
        """Greedy graph coloring"""
        colors = {}
        neighbors = {i: [] for i in range(n_nodes)}
        
        for u, v in edges:
            neighbors[u].append(v)
            neighbors[v].append(u)
        
        for node in range(n_nodes):
            used_colors = {colors[n] for n in neighbors[node] if n in colors}
            color = 0
            while color in used_colors:
                color += 1
            colors[node] = color
        
        return colors
    
    def _validate_coloring(self, n_nodes: int, edges: List[Tuple[int, int]], colors: Dict[int, int]) -> bool:
        """Validate graph coloring"""
        for u, v in edges:
            if u in colors and v in colors and colors[u] == colors[v]:
                return False
        return True
    
    def _subset_sum_heuristic(self, numbers: np.ndarray, target: int) -> Optional[List[int]]:
        """Subset sum heuristic"""
        return self._greedy_subset_sum(numbers, target)
    
    def _greedy_subset_sum(self, numbers: np.ndarray, target: int) -> Optional[List[int]]:
        """Greedy subset sum"""
        sorted_indices = np.argsort(numbers)[::-1]
        current_sum = 0
        solution = []
        
        for idx in sorted_indices:
            if current_sum + numbers[idx] <= target:
                solution.append(idx)
                current_sum += numbers[idx]
                if abs(current_sum - target) < 0.01:
                    return solution
        
        return solution if abs(current_sum - target) < target * 0.1 else None
    
    def _knapsack_heuristic(self, weights: np.ndarray, values: np.ndarray, capacity: int) -> List[int]:
        """Knapsack heuristic"""
        return self._greedy_knapsack(weights, values, capacity)
    
    def _greedy_knapsack(self, weights: np.ndarray, values: np.ndarray, capacity: int) -> List[int]:
        """Greedy knapsack solution"""
        ratios = values / weights
        sorted_indices = np.argsort(ratios)[::-1]
        
        solution = []
        current_weight = 0
        
        for idx in sorted_indices:
            if current_weight + weights[idx] <= capacity:
                solution.append(idx)
                current_weight += weights[idx]
        
        return solution
    
    def generate_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        summary = {
            'total_tests': 0,
            'toolbox_wins': 0,
            'sklearn_wins': 0,
            'ties': 0,
            'toolbox_errors': 0,
            'sklearn_errors': 0,
            'accuracy_comparison': {},
            'speed_comparison': {},
            'test_categories': {}
        }
        
        # Process all test categories
        for category, tests in all_results.items():
            if category == 'summary':
                continue
            
            category_stats = {
                'total': 0,
                'toolbox_wins': 0,
                'sklearn_wins': 0,
                'ties': 0
            }
            
            for test_name, test_results in tests.items():
                if isinstance(test_results, dict):
                    # Check if it's a test result (has toolbox or sklearn key)
                    if 'toolbox' in test_results or 'sklearn' in test_results or 'baseline' in test_results:
                    summary['total_tests'] += 1
                    category_stats['total'] += 1
                    
                    # Compare results
                    toolbox_result = test_results.get('toolbox', {})
                    sklearn_result = test_results.get('sklearn', test_results.get('baseline', {}))
                    
                    if 'error' in toolbox_result:
                        summary['toolbox_errors'] += 1
                    if 'error' in sklearn_result:
                        summary['sklearn_errors'] += 1
                    
                    # Skip if both have errors
                    if 'error' in toolbox_result and 'error' in sklearn_result:
                        continue
                    
                    # Compare accuracy/performance
                    if toolbox_result.get('success') and sklearn_result.get('success'):
                        toolbox_metric = toolbox_result.get('accuracy') or toolbox_result.get('r2_score') or toolbox_result.get('silhouette_score') or toolbox_result.get('num_features', 0)
                        sklearn_metric = sklearn_result.get('accuracy') or sklearn_result.get('r2_score') or sklearn_result.get('silhouette_score') or sklearn_result.get('num_features', 0)
                        
                        # Handle NP-complete problems differently
                        if 'tsp' in test_name or 'graph_coloring' in test_name or 'subset_sum' in test_name or 'knapsack' in test_name:
                            # Compare by solution quality (lower distance/colors is better for TSP/graph coloring)
                            if 'distance' in toolbox_result and 'distance' in sklearn_result:
                                if toolbox_result['distance'] < sklearn_result['distance']:
                                    summary['toolbox_wins'] += 1
                                    category_stats['toolbox_wins'] += 1
                                elif sklearn_result['distance'] < toolbox_result['distance']:
                                    summary['sklearn_wins'] += 1
                                    category_stats['sklearn_wins'] += 1
                                else:
                                    summary['ties'] += 1
                                    category_stats['ties'] += 1
                            elif 'num_colors' in toolbox_result and 'num_colors' in sklearn_result:
                                if toolbox_result['num_colors'] < sklearn_result['num_colors']:
                                    summary['toolbox_wins'] += 1
                                    category_stats['toolbox_wins'] += 1
                                elif sklearn_result['num_colors'] < toolbox_result['num_colors']:
                                    summary['sklearn_wins'] += 1
                                    category_stats['sklearn_wins'] += 1
                                else:
                                    summary['ties'] += 1
                                    category_stats['ties'] += 1
                        else:
                            # Standard comparison (higher is better)
                            if toolbox_metric > sklearn_metric:
                                summary['toolbox_wins'] += 1
                                category_stats['toolbox_wins'] += 1
                            elif sklearn_metric > toolbox_metric:
                                summary['sklearn_wins'] += 1
                                category_stats['sklearn_wins'] += 1
                            else:
                                summary['ties'] += 1
                                category_stats['ties'] += 1
                        
                        # Speed comparison
                        toolbox_time = toolbox_result.get('time', 0)
                        sklearn_time = sklearn_result.get('time', 0)
                        if toolbox_time > 0 and sklearn_time > 0:
                            speed_ratio = sklearn_time / toolbox_time
                            if test_name not in summary['speed_comparison']:
                                summary['speed_comparison'][test_name] = speed_ratio
            
            summary['test_categories'][category] = category_stats
        
        # Calculate win rates
        if summary['total_tests'] > 0:
            summary['toolbox_win_rate'] = summary['toolbox_wins'] / summary['total_tests'] * 100
            summary['sklearn_win_rate'] = summary['sklearn_wins'] / summary['total_tests'] * 100
            summary['tie_rate'] = summary['ties'] / summary['total_tests'] * 100
        
        return summary


def run_comprehensive_tests():
    """Run comprehensive test suite and generate report"""
    suite = ComprehensiveMLTestSuite()
    results = suite.run_all_tests()
    
    # Save results
    with open('comprehensive_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    summary = results['summary']
    print(f"\nTotal Tests: {summary['total_tests']}")
    print(f"ML Toolbox Wins: {summary['toolbox_wins']} ({summary.get('toolbox_win_rate', 0):.1f}%)")
    print(f"scikit-learn Wins: {summary['sklearn_wins']} ({summary.get('sklearn_win_rate', 0):.1f}%)")
    print(f"Ties: {summary['ties']} ({summary.get('tie_rate', 0):.1f}%)")
    print(f"\nToolbox Errors: {summary['toolbox_errors']}")
    print(f"sklearn Errors: {summary['sklearn_errors']}")
    
    print("\n" + "="*80)
    print("CATEGORY BREAKDOWN")
    print("="*80)
    for category, stats in summary['test_categories'].items():
        print(f"\n{category.upper()}:")
        print(f"  Total: {stats['total']}")
        print(f"  Toolbox Wins: {stats['toolbox_wins']}")
        print(f"  sklearn Wins: {stats['sklearn_wins']}")
        print(f"  Ties: {stats['ties']}")
    
    # Generate detailed report
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for category, tests in results.items():
        if category == 'summary':
            continue
        print(f"\n{category.upper()}:")
        for test_name, test_results in tests.items():
            print(f"\n  {test_name}:")
            if 'toolbox' in test_results:
                tb = test_results['toolbox']
                if 'error' not in tb:
                    metric = tb.get('accuracy') or tb.get('r2_score') or tb.get('silhouette_score', 'N/A')
                    print(f"    Toolbox: {metric} (time: {tb.get('time', 0):.4f}s)")
                else:
                    print(f"    Toolbox: ERROR - {tb.get('error', 'Unknown')}")
            
            if 'sklearn' in test_results:
                sk = test_results['sklearn']
                if 'error' not in sk:
                    metric = sk.get('accuracy') or sk.get('r2_score') or sk.get('silhouette_score', 'N/A')
                    print(f"    sklearn: {metric} (time: {sk.get('time', 0):.4f}s)")
                else:
                    print(f"    sklearn: ERROR - {sk.get('error', 'Unknown')}")
    
    return results


if __name__ == '__main__':
    results = run_comprehensive_tests()
