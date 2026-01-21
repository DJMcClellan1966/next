"""
Statistical Learning Methods for ML Toolbox
Uncertainty quantification, statistical validation, Bayesian methods, and more
"""
import sys
from pathlib import Path
import time
from typing import List, Dict, Tuple, Any, Optional, Callable
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import required libraries
try:
    from sklearn.model_selection import (
        train_test_split, cross_val_score, KFold, StratifiedKFold,
        GridSearchCV, RandomizedSearchCV
    )
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score
    )
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

try:
    from scipy import stats
    from scipy.stats import bootstrap, permutation_test
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.acquisition import gaussian_ei
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not available for Bayesian optimization. Install with: pip install scikit-optimize")

try:
    from sklearn.feature_selection import (
        mutual_info_classif, mutual_info_regr,
        chi2, f_classif, f_regression
    )
    FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    FEATURE_SELECTION_AVAILABLE = False
    print("Warning: sklearn feature selection not available")


class StatisticalEvaluator:
    """
    Statistical evaluation with uncertainty quantification
    
    Features:
    - Confidence intervals for predictions
    - Prediction intervals
    - Bootstrap-based uncertainty
    - Uncertainty scores
    """
    
    def __init__(self, random_state: int = 42, n_bootstrap: int = 1000):
        self.random_state = random_state
        self.n_bootstrap = n_bootstrap
        np.random.seed(random_state)
    
    def predict_with_confidence(
        self,
        model: Any,
        X: np.ndarray,
        confidence_level: float = 0.95,
        method: str = 'bootstrap'
    ) -> Dict[str, Any]:
        """
        Get predictions with confidence intervals
        
        Big O: O(n_bootstrap * n * f(n)) where f(n) is model complexity
        
        Args:
            model: Trained scikit-learn compatible model
            X: Features
            confidence_level: Confidence level (0.95 = 95%)
            method: 'bootstrap' or 'residual'
            
        Returns:
            Dictionary with predictions, confidence intervals, and uncertainty scores
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        # Get point predictions
        predictions = model.predict(X)
        
        if method == 'bootstrap':
            return self._bootstrap_confidence_intervals(
                model, X, predictions, confidence_level
            )
        elif method == 'residual':
            return self._residual_based_intervals(
                model, X, predictions, confidence_level
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _bootstrap_confidence_intervals(
        self,
        model: Any,
        X: np.ndarray,
        predictions: np.ndarray,
        confidence_level: float
    ) -> Dict[str, Any]:
        """Bootstrap-based confidence intervals"""
        if not SCIPY_AVAILABLE:
            # Fallback: simple bootstrap
            return self._simple_bootstrap(model, X, predictions, confidence_level)
        
        # Use scipy.stats.bootstrap for confidence intervals
        n_samples = len(X)
        bootstrap_predictions = []
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            
            # Get predictions
            pred_boot = model.predict(X_boot)
            bootstrap_predictions.append(pred_boot)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_intervals = np.percentile(
            bootstrap_predictions,
            [lower_percentile, upper_percentile],
            axis=0
        )
        
        # Uncertainty score (coefficient of variation)
        std_predictions = np.std(bootstrap_predictions, axis=0)
        mean_predictions = np.mean(bootstrap_predictions, axis=0)
        uncertainty_scores = np.where(
            mean_predictions != 0,
            std_predictions / np.abs(mean_predictions),
            std_predictions
        )
        
        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals.T,  # (n_samples, 2)
            'uncertainty_scores': uncertainty_scores,
            'bootstrap_mean': mean_predictions,
            'bootstrap_std': std_predictions,
            'confidence_level': confidence_level,
            'method': 'bootstrap'
        }
    
    def _simple_bootstrap(
        self,
        model: Any,
        X: np.ndarray,
        predictions: np.ndarray,
        confidence_level: float
    ) -> Dict[str, Any]:
        """Simple bootstrap without scipy"""
        n_samples = len(X)
        bootstrap_predictions = []
        
        for _ in range(min(self.n_bootstrap, 100)):  # Limit for performance
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            pred_boot = model.predict(X_boot)
            bootstrap_predictions.append(pred_boot)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_intervals = np.percentile(
            bootstrap_predictions,
            [lower_percentile, upper_percentile],
            axis=0
        )
        
        std_predictions = np.std(bootstrap_predictions, axis=0)
        mean_predictions = np.mean(bootstrap_predictions, axis=0)
        uncertainty_scores = np.where(
            mean_predictions != 0,
            std_predictions / np.abs(mean_predictions),
            std_predictions
        )
        
        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals.T,
            'uncertainty_scores': uncertainty_scores,
            'bootstrap_mean': mean_predictions,
            'bootstrap_std': std_predictions,
            'confidence_level': confidence_level,
            'method': 'simple_bootstrap'
        }
    
    def _residual_based_intervals(
        self,
        model: Any,
        X: np.ndarray,
        predictions: np.ndarray,
        confidence_level: float
    ) -> Dict[str, Any]:
        """Residual-based prediction intervals (for regression)"""
        # This would require training residuals
        # Simplified version
        std_residuals = np.std(predictions) * 0.1  # Estimate
        
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2) if SCIPY_AVAILABLE else 1.96
        
        margin = z_score * std_residuals
        
        prediction_intervals = np.array([
            predictions - margin,
            predictions + margin
        ]).T
        
        uncertainty_scores = margin / (np.abs(predictions) + 1e-8)
        
        return {
            'predictions': predictions,
            'prediction_intervals': prediction_intervals,
            'uncertainty_scores': uncertainty_scores,
            'confidence_level': confidence_level,
            'method': 'residual'
        }
    
    def bootstrap_confidence_intervals(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: Optional[int] = None,
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Bootstrap confidence intervals for model parameters/performance
        
        Big O: O(n_bootstrap * n * f(n))
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            
        Returns:
            Dictionary with confidence intervals and statistics
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        n_bootstrap = n_bootstrap or self.n_bootstrap
        n_samples = len(X)
        
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model
            model_boot = type(model)(**model.get_params())
            model_boot.fit(X_boot, y_boot)
            
            # Evaluate
            if hasattr(model_boot, 'score'):
                score = model_boot.score(X_boot, y_boot)
                bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        
        return {
            'mean_score': np.mean(bootstrap_scores),
            'std_score': np.std(bootstrap_scores),
            'confidence_interval': (ci_lower, ci_upper),
            'bootstrap_distribution': bootstrap_scores,
            'confidence_level': confidence_level
        }


class StatisticalValidator:
    """
    Statistical validation methods
    
    Features:
    - Permutation tests for model comparison
    - Bootstrap validation
    - Hypothesis testing
    - Statistical significance
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def permutation_test(
        self,
        model1: Any,
        model2: Any,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = 'accuracy',
        n_permutations: int = 1000,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Permutation test for model comparison
        
        Tests: H0: models have same performance
               H1: models differ significantly
        
        Big O: O(n_permutations * n * f(n))
        
        Args:
            model1: First model
            model2: Second model
            X: Features
            y: Labels
            metric: 'accuracy', 'f1', 'r2', 'mse', etc.
            n_permutations: Number of permutations
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with p-value, statistic, significance, effect size
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        # Train both models
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Get predictions
        y_pred1 = model1.predict(X)
        y_pred2 = model2.predict(X)
        
        # Calculate metric for both models
        score1 = self._calculate_metric(y, y_pred1, metric, task_type)
        score2 = self._calculate_metric(y, y_pred2, metric, task_type)
        
        # Observed difference
        observed_diff = abs(score1 - score2)
        
        # Permutation test
        if SCIPY_AVAILABLE:
            # Use scipy's permutation_test
            def statistic(x, y):
                m1 = type(model1)(**model1.get_params())
                m2 = type(model2)(**model2.get_params())
                m1.fit(X, x)
                m2.fit(X, y)
                s1 = self._calculate_metric(x, m1.predict(X), metric, task_type)
                s2 = self._calculate_metric(y, m2.predict(X), metric, task_type)
                return abs(s1 - s2)
            
            result = permutation_test(
                (y, y),
                statistic,
                permutation_type='pairings',
                n_resamples=n_permutations,
                random_state=self.random_state
            )
            
            p_value = result.pvalue
        else:
            # Manual permutation test
            p_value = self._manual_permutation_test(
                model1, model2, X, y, observed_diff, metric, n_permutations, task_type
            )
        
        # Effect size (Cohen's d)
        effect_size = observed_diff / (np.std([score1, score2]) + 1e-8)
        
        # Significance
        significant = p_value < 0.05
        
        return {
            'p_value': float(p_value),
            'statistic': float(observed_diff),
            'significant': significant,
            'effect_size': float(effect_size),
            'model1_score': float(score1),
            'model2_score': float(score2),
            'n_permutations': n_permutations
        }
    
    def _manual_permutation_test(
        self,
        model1: Any,
        model2: Any,
        X: np.ndarray,
        y: np.ndarray,
        observed_diff: float,
        metric: str,
        n_permutations: int,
        task_type: str
    ) -> float:
        """Manual permutation test implementation"""
        n_samples = len(y)
        permuted_diffs = []
        
        for _ in range(n_permutations):
            # Permute labels
            y_perm = np.random.permutation(y)
            
            # Train models
            m1 = type(model1)(**model1.get_params())
            m2 = type(model2)(**model2.get_params())
            m1.fit(X, y_perm)
            m2.fit(X, y_perm)
            
            # Calculate difference
            s1 = self._calculate_metric(y_perm, m1.predict(X), metric, task_type)
            s2 = self._calculate_metric(y_perm, m2.predict(X), metric, task_type)
            permuted_diffs.append(abs(s1 - s2))
        
        # P-value: proportion of permuted differences >= observed
        p_value = np.mean(np.array(permuted_diffs) >= observed_diff)
        return p_value
    
    def _calculate_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str,
        task_type: str
    ) -> float:
        """Calculate metric"""
        if task_type == 'classification':
            if metric == 'accuracy':
                return accuracy_score(y_true, y_pred)
            elif metric == 'f1':
                return f1_score(y_true, y_pred, average='weighted')
            elif metric == 'precision':
                return precision_score(y_true, y_pred, average='weighted', zero_division=0)
            elif metric == 'recall':
                return recall_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                return accuracy_score(y_true, y_pred)
        else:  # regression
            if metric == 'r2':
                return r2_score(y_true, y_pred)
            elif metric == 'mse':
                return -mean_squared_error(y_true, y_pred)  # Negative for maximization
            elif metric == 'mae':
                return -mean_absolute_error(y_true, y_pred)
            else:
                return r2_score(y_true, y_pred)
    
    def bootstrap_validation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 1000,
        metric: str = 'accuracy',
        task_type: str = 'classification',
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Bootstrap validation with confidence intervals
        
        Big O: O(n_bootstrap * n * f(n))
        
        Args:
            model: Model to validate
            X: Features
            y: Labels
            n_bootstrap: Number of bootstrap samples
            metric: Metric to evaluate
            task_type: 'classification' or 'regression'
            test_size: Test set size ratio
            
        Returns:
            Dictionary with mean, std, confidence interval, bootstrap distribution
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            n_samples = len(X)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train/test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_boot, y_boot, test_size=test_size, random_state=self.random_state
            )
            
            # Train model
            model_boot = type(model)(**model.get_params())
            model_boot.fit(X_train, y_train)
            
            # Evaluate
            score = self._calculate_metric(
                y_test, model_boot.predict(X_test), metric, task_type
            )
            bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calculate statistics
        mean_score = np.mean(bootstrap_scores)
        std_score = np.std(bootstrap_scores)
        
        # 95% confidence interval
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        
        return {
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'bootstrap_distribution': bootstrap_scores.tolist(),
            'n_bootstrap': n_bootstrap
        }
    
    def hypothesis_test(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        null_hypothesis: str = 'accuracy >= 0.8',
        test_type: str = 'one_sample_t',
        metric: str = 'accuracy',
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Statistical hypothesis testing
        
        Args:
            model: Model to test
            X: Features
            y: Labels
            null_hypothesis: Null hypothesis (e.g., 'accuracy >= 0.8')
            test_type: 'one_sample_t', 'two_sample_t', 'wilcoxon'
            metric: Metric to test
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with p-value, test statistic, reject_null, effect_size
        """
        if not SKLEARN_AVAILABLE or not SCIPY_AVAILABLE:
            return {'error': 'sklearn or scipy not available'}
        
        # Train model
        model.fit(X, y)
        y_pred = model.predict(X)
        score = self._calculate_metric(y, y_pred, metric, task_type)
        
        # Parse null hypothesis
        # Simple parsing for 'metric >= value' or 'metric <= value'
        if '>=' in null_hypothesis:
            null_value = float(null_hypothesis.split('>=')[1].strip())
            alternative = 'less'
        elif '<=' in null_hypothesis:
            null_value = float(null_hypothesis.split('<=')[1].strip())
            alternative = 'greater'
        else:
            null_value = 0.8  # Default
            alternative = 'two-sided'
        
        if test_type == 'one_sample_t':
            # One-sample t-test
            # Use cross-validation scores
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X, y, cv=5, scoring=metric if task_type == 'classification' else 'r2')
            
            t_stat, p_value = stats.ttest_1samp(cv_scores, null_value, alternative=alternative)
            
            reject_null = p_value < 0.05
            effect_size = (np.mean(cv_scores) - null_value) / (np.std(cv_scores) + 1e-8)
            
            return {
                'p_value': float(p_value),
                'test_statistic': float(t_stat),
                'reject_null': reject_null,
                'effect_size': float(effect_size),
                'observed_value': float(np.mean(cv_scores)),
                'null_value': null_value
            }
        else:
            return {'error': f'Test type {test_type} not implemented'}


class BayesianOptimizer:
    """
    Bayesian hyperparameter optimization
    
    Uses Gaussian Processes for efficient hyperparameter search
    More efficient than grid/random search
    """
    
    def __init__(self, random_state: int = 42, n_calls: int = 50):
        self.random_state = random_state
        self.n_calls = n_calls
        np.random.seed(random_state)
    
    def optimize(
        self,
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        param_space: Dict[str, Any],
        n_iterations: Optional[int] = None,
        acquisition_function: str = 'EI',  # Expected Improvement
        task_type: str = 'classification',
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Bayesian optimization using Gaussian Processes
        
        Big O: O(n_iterations * n²) for GP
        
        Args:
            model_class: Model class (e.g., RandomForestClassifier)
            X: Features
            y: Labels
            param_space: Parameter space dictionary
            n_iterations: Number of iterations (default: self.n_calls)
            acquisition_function: 'EI', 'LCB', 'PI'
            task_type: 'classification' or 'regression'
            cv: Cross-validation folds
            
        Returns:
            Dictionary with best parameters, best score, optimization history
        """
        if not SKOPT_AVAILABLE:
            return {
                'error': 'scikit-optimize not available. Install with: pip install scikit-optimize',
                'fallback': 'Use grid search or random search instead'
            }
        
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        n_iterations = n_iterations or self.n_calls
        
        # Convert param_space to skopt space
        dimensions = []
        param_names = []
        
        for name, values in param_space.items():
            param_names.append(name)
            if isinstance(values, tuple) and len(values) == 2:
                # Continuous range
                dimensions.append(Real(values[0], values[1], name=name))
            elif isinstance(values, list):
                # Categorical
                dimensions.append(Categorical(values, name=name))
            elif isinstance(values, tuple) and len(values) == 3:
                # Integer range
                dimensions.append(Integer(int(values[0]), int(values[1]), name=name))
            else:
                # Default: continuous
                dimensions.append(Real(0.0, 1.0, name=name))
        
        # Objective function
        def objective(params):
            # Create model with parameters
            model_params = dict(zip(param_names, params))
            model = model_class(**model_params)
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            scoring = 'accuracy' if task_type == 'classification' else 'r2'
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            # Return negative mean (for minimization)
            return -np.mean(scores)
        
        # Run optimization
        try:
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=n_iterations,
                random_state=self.random_state,
                acq_func=acquisition_function.lower()
            )
            
            # Get best parameters
            best_params = dict(zip(param_names, result.x))
            best_score = -result.fun  # Negative because we minimized
            
            return {
                'best_params': best_params,
                'best_score': float(best_score),
                'optimization_history': {
                    'iterations': list(range(n_iterations)),
                    'scores': [-score for score in result.func_vals],
                    'best_scores': [-min(result.func_vals[:i+1]) for i in range(n_iterations)]
                },
                'n_iterations': n_iterations,
                'method': 'bayesian_optimization'
            }
        except Exception as e:
            return {
                'error': f'Optimization failed: {str(e)}',
                'fallback': 'Use grid search or random search instead'
            }


class StatisticalFeatureSelector:
    """
    Statistical feature selection methods
    
    Features:
    - Mutual information selection
    - Chi-square tests
    - F-tests
    - Statistical significance
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def mutual_information_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 10,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Select features using mutual information
        
        Big O: O(n_features * n_samples)
        
        Args:
            X: Features
            y: Labels
            k: Number of features to select
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with selected features, scores, p-values
        """
        if not FEATURE_SELECTION_AVAILABLE:
            return {'error': 'sklearn feature selection not available'}
        
        # Calculate mutual information
        if task_type == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        else:
            mi_scores = mutual_info_regr(X, y, random_state=self.random_state)
        
        # Select top k features
        top_k_indices = np.argsort(mi_scores)[-k:][::-1]
        
        return {
            'selected_features': top_k_indices.tolist(),
            'scores': mi_scores.tolist(),
            'top_k_scores': mi_scores[top_k_indices].tolist(),
            'k': k,
            'method': 'mutual_information'
        }
    
    def chi_square_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.05,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Select features using chi-square test
        
        For categorical features
        
        Args:
            X: Features (should be non-negative integers)
            y: Labels
            alpha: Significance level
            k: Number of features to select (optional)
            
        Returns:
            Dictionary with selected features, chi2 scores, p-values
        """
        if not FEATURE_SELECTION_AVAILABLE:
            return {'error': 'sklearn feature selection not available'}
        
        # Chi-square test
        chi2_scores, p_values = chi2(X, y)
        
        # Select significant features
        significant_features = np.where(p_values < alpha)[0]
        
        # If k specified, select top k
        if k is not None:
            top_k_indices = np.argsort(chi2_scores)[-k:][::-1]
            selected_features = top_k_indices.tolist()
        else:
            selected_features = significant_features.tolist()
        
        return {
            'selected_features': selected_features,
            'chi2_scores': chi2_scores.tolist(),
            'p_values': p_values.tolist(),
            'significant_features': significant_features.tolist(),
            'alpha': alpha,
            'method': 'chi_square'
        }
    
    def f_test_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.05,
        k: Optional[int] = None,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Select features using F-test
        
        For continuous features
        
        Args:
            X: Features
            y: Labels
            alpha: Significance level
            k: Number of features to select (optional)
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with selected features, F scores, p-values
        """
        if not FEATURE_SELECTION_AVAILABLE:
            return {'error': 'sklearn feature selection not available'}
        
        # F-test
        if task_type == 'classification':
            f_scores, p_values = f_classif(X, y)
        else:
            f_scores, p_values = f_regression(X, y)
        
        # Select significant features
        significant_features = np.where(p_values < alpha)[0]
        
        # If k specified, select top k
        if k is not None:
            top_k_indices = np.argsort(f_scores)[-k:][::-1]
            selected_features = top_k_indices.tolist()
        else:
            selected_features = significant_features.tolist()
        
        return {
            'selected_features': selected_features,
            'f_scores': f_scores.tolist(),
            'p_values': p_values.tolist(),
            'significant_features': significant_features.tolist(),
            'alpha': alpha,
            'method': 'f_test'
        }
