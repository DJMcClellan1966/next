"""
A/B Testing Framework for ML Models
Burkov Machine Learning Engineering - A/B Testing

Features:
- Statistical A/B testing
- Traffic splitting
- Metric collection
- Significance testing
- Multiple variants support
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
from datetime import datetime
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import scipy for statistical tests
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")


class ABTest:
    """
    A/B Testing framework for ML models
    
    Compares performance of different model versions
    """
    
    def __init__(
        self,
        test_name: str,
        variants: Dict[str, Any],
        traffic_split: Optional[Dict[str, float]] = None,
        random_seed: int = 42
    ):
        """
        Args:
            test_name: Name of the A/B test
            variants: Dictionary of {variant_name: model}
            traffic_split: Dictionary of {variant_name: percentage} (must sum to 1.0)
            random_seed: Random seed for consistent routing
        """
        self.test_name = test_name
        self.variants = variants
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Default traffic split (equal)
        if traffic_split is None:
            n_variants = len(variants)
            traffic_split = {name: 1.0 / n_variants for name in variants.keys()}
        
        # Validate traffic split
        total = sum(traffic_split.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Traffic split must sum to 1.0, got {total}")
        
        self.traffic_split = traffic_split
        
        # Metrics collection
        self.metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.predictions: Dict[str, List] = defaultdict(list)
        self.labels: Dict[str, List] = defaultdict(list)
        self.timestamps: Dict[str, List] = defaultdict(list)
    
    def route_request(self, request_id: Optional[str] = None) -> str:
        """
        Route request to a variant
        
        Args:
            request_id: Optional request ID for consistent routing
            
        Returns:
            Variant name
        """
        if request_id:
            # Consistent routing based on request ID
            hash_value = hash(request_id) % 10000
            cumulative = 0.0
            for variant, percentage in self.traffic_split.items():
                cumulative += percentage
                if hash_value < cumulative * 10000:
                    return variant
            return list(self.variants.keys())[-1]  # Fallback
        else:
            # Random routing
            rand = np.random.random()
            cumulative = 0.0
            for variant, percentage in self.traffic_split.items():
                cumulative += percentage
                if rand < cumulative:
                    return variant
            return list(self.variants.keys())[-1]  # Fallback
    
    def record_prediction(
        self,
        variant: str,
        prediction: Any,
        label: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Record a prediction and metrics
        
        Args:
            variant: Variant name
            prediction: Model prediction
            label: True label (optional)
            metrics: Additional metrics (optional)
        """
        if variant not in self.variants:
            raise ValueError(f"Unknown variant: {variant}")
        
        self.predictions[variant].append(prediction)
        self.timestamps[variant].append(datetime.now())
        
        if label is not None:
            self.labels[variant].append(label)
        
        if metrics:
            for metric_name, metric_value in metrics.items():
                self.metrics[variant][metric_name].append(metric_value)
    
    def compute_metrics(
        self,
        variant: str,
        metric_name: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Compute metrics for a variant
        
        Args:
            variant: Variant name
            metric_name: Metric to compute ('accuracy', 'precision', 'recall', 'f1', 'mse', 'mae', 'r2')
            
        Returns:
            Dictionary with metric statistics
        """
        if variant not in self.variants:
            raise ValueError(f"Unknown variant: {variant}")
        
        if metric_name in self.metrics[variant]:
            values = self.metrics[variant][metric_name]
        elif len(self.labels[variant]) > 0 and len(self.predictions[variant]) > 0:
            # Compute metric from predictions and labels
            try:
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score,
                    mean_squared_error, mean_absolute_error, r2_score
                )
                
                y_true = np.array(self.labels[variant])
                y_pred = np.array(self.predictions[variant])
                
                if metric_name == 'accuracy':
                    values = [accuracy_score(y_true, y_pred)]
                elif metric_name == 'precision':
                    values = [precision_score(y_true, y_pred, average='weighted', zero_division=0)]
                elif metric_name == 'recall':
                    values = [recall_score(y_true, y_pred, average='weighted', zero_division=0)]
                elif metric_name == 'f1':
                    values = [f1_score(y_true, y_pred, average='weighted', zero_division=0)]
                elif metric_name == 'mse':
                    values = [mean_squared_error(y_true, y_pred)]
                elif metric_name == 'mae':
                    values = [mean_absolute_error(y_true, y_pred)]
                elif metric_name == 'r2':
                    values = [r2_score(y_true, y_pred)]
                else:
                    values = []
            except ImportError:
                # Fallback: simple accuracy
                y_true = np.array(self.labels[variant])
                y_pred = np.array(self.predictions[variant])
                if metric_name == 'accuracy':
                    values = [np.mean(y_true == y_pred)]
                else:
                    values = []
        else:
            values = []
        
        if len(values) == 0:
            return {'error': f'No metrics available for {metric_name}'}
        
        return {
            'variant': variant,
            'metric': metric_name,
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'n_samples': len(values)
        }
    
    def compare_variants(
        self,
        metric_name: str = 'accuracy',
        variant_a: Optional[str] = None,
        variant_b: Optional[str] = None,
        test_type: str = 't_test'
    ) -> Dict[str, Any]:
        """
        Compare two variants statistically
        
        Args:
            metric_name: Metric to compare
            variant_a: First variant (None for first variant)
            variant_b: Second variant (None for second variant)
            test_type: Statistical test ('t_test', 'mannwhitney', 'chi_square')
            
        Returns:
            Dictionary with comparison results
        """
        variants_list = list(self.variants.keys())
        
        if variant_a is None:
            variant_a = variants_list[0]
        if variant_b is None:
            variant_b = variants_list[1] if len(variants_list) > 1 else variants_list[0]
        
        if variant_a not in self.variants or variant_b not in self.variants:
            raise ValueError("Invalid variant names")
        
        # Get metrics for both variants
        metrics_a = self.compute_metrics(variant_a, metric_name)
        metrics_b = self.compute_metrics(variant_b, metric_name)
        
        if 'error' in metrics_a or 'error' in metrics_b:
            return {'error': 'Could not compute metrics for comparison'}
        
        # Get raw metric values
        if metric_name in self.metrics[variant_a] and metric_name in self.metrics[variant_b]:
            values_a = self.metrics[variant_a][metric_name]
            values_b = self.metrics[variant_b][metric_name]
        else:
            # Use computed means
            values_a = [metrics_a['mean']]
            values_b = [metrics_b['mean']]
        
        # Statistical test
        if not SCIPY_AVAILABLE:
            return {
                'error': 'scipy not available for statistical testing',
                'metrics_a': metrics_a,
                'metrics_b': metrics_b
            }
        
        if test_type == 't_test':
            # Independent t-test
            statistic, p_value = stats.ttest_ind(values_a, values_b)
            test_name = 't_test'
        elif test_type == 'mannwhitney':
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
            test_name = 'mannwhitney'
        else:
            # Default to t-test
            statistic, p_value = stats.ttest_ind(values_a, values_b)
            test_name = 't_test'
        
        # Determine significance
        is_significant = p_value < 0.05
        better_variant = variant_a if metrics_a['mean'] > metrics_b['mean'] else variant_b
        
        return {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'metric': metric_name,
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'difference': metrics_a['mean'] - metrics_b['mean'],
            'relative_improvement': (metrics_a['mean'] - metrics_b['mean']) / (metrics_b['mean'] + 1e-10),
            'test_type': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_significant': is_significant,
            'better_variant': better_variant,
            'recommendation': f"Use {better_variant}" if is_significant else "No significant difference"
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get A/B test summary"""
        summary = {
            'test_name': self.test_name,
            'variants': list(self.variants.keys()),
            'traffic_split': self.traffic_split,
            'n_predictions': {v: len(self.predictions[v]) for v in self.variants.keys()},
            'n_labels': {v: len(self.labels[v]) for v in self.variants.keys()},
            'start_time': min([min(ts) for ts in self.timestamps.values()]) if any(self.timestamps.values()) else None,
            'end_time': max([max(ts) for ts in self.timestamps.values()]) if any(self.timestamps.values()) else None
        }
        
        return summary


class MultiVariantTest:
    """
    Multi-variant testing (A/B/C/... testing)
    
    Extends A/B testing to multiple variants
    """
    
    def __init__(
        self,
        test_name: str,
        variants: Dict[str, Any],
        traffic_split: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            test_name: Name of the test
            variants: Dictionary of {variant_name: model}
            traffic_split: Traffic split (must sum to 1.0)
        """
        self.test = ABTest(test_name, variants, traffic_split)
    
    def compare_all(self, metric_name: str = 'accuracy') -> Dict[str, Any]:
        """
        Compare all variants pairwise
        
        Args:
            metric_name: Metric to compare
            
        Returns:
            Dictionary with all pairwise comparisons
        """
        variants = list(self.test.variants.keys())
        comparisons = {}
        
        for i, variant_a in enumerate(variants):
            for variant_b in variants[i+1:]:
                comparison = self.test.compare_variants(metric_name, variant_a, variant_b)
                comparisons[f"{variant_a}_vs_{variant_b}"] = comparison
        
        # Find best variant
        variant_metrics = {}
        for variant in variants:
            metrics = self.test.compute_metrics(variant, metric_name)
            if 'mean' in metrics:
                variant_metrics[variant] = metrics['mean']
        
        best_variant = max(variant_metrics.items(), key=lambda x: x[1])[0] if variant_metrics else None
        
        return {
            'comparisons': comparisons,
            'best_variant': best_variant,
            'variant_metrics': variant_metrics
        }
