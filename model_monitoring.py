"""
Model Monitoring & Drift Detection
Burkov Machine Learning Engineering - Production Monitoring

Features:
- Data drift detection (feature distribution changes)
- Concept drift detection (target relationship changes)
- Performance monitoring (accuracy, latency, throughput)
- Alert systems
- Model degradation detection
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict, deque
import warnings
from datetime import datetime, timedelta
import time

sys.path.insert(0, str(Path(__file__).parent))

# Try to import required libraries
try:
    from scipy import stats
    from scipy.stats import ks_2samp, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class DataDriftDetector:
    """
    Detect data drift (feature distribution changes)
    
    Compares current data distribution to reference (training) distribution
    """
    
    def __init__(self, reference_data: np.ndarray, alpha: float = 0.05):
        """
        Args:
            reference_data: Reference data (training data)
            alpha: Significance level for drift detection
        """
        self.reference_data = np.asarray(reference_data)
        self.alpha = alpha
        self.reference_stats = self._compute_reference_stats()
    
    def _compute_reference_stats(self) -> Dict[str, Any]:
        """Compute statistics for reference data"""
        stats_dict = {}
        
        if len(self.reference_data.shape) == 1:
            self.reference_data = self.reference_data.reshape(-1, 1)
        
        for i in range(self.reference_data.shape[1]):
            feature = self.reference_data[:, i]
            stats_dict[i] = {
                'mean': float(np.mean(feature)),
                'std': float(np.std(feature)),
                'min': float(np.min(feature)),
                'max': float(np.max(feature)),
                'median': float(np.median(feature)),
                'q25': float(np.percentile(feature, 25)),
                'q75': float(np.percentile(feature, 75))
            }
        
        return stats_dict
    
    def detect_drift(
        self,
        current_data: np.ndarray,
        method: str = 'ks_test'
    ) -> Dict[str, Any]:
        """
        Detect data drift
        
        Args:
            current_data: Current data to check
            method: 'ks_test' (Kolmogorov-Smirnov), 'psi' (Population Stability Index)
            
        Returns:
            Dictionary with drift detection results
        """
        current_data = np.asarray(current_data)
        
        if len(current_data.shape) == 1:
            current_data = current_data.reshape(-1, 1)
        
        drift_results = {}
        has_drift = False
        
        for i in range(min(self.reference_data.shape[1], current_data.shape[1])):
            ref_feature = self.reference_data[:, i]
            curr_feature = current_data[:, i]
            
            if method == 'ks_test' and SCIPY_AVAILABLE:
                # Kolmogorov-Smirnov test
                statistic, p_value = ks_2samp(ref_feature, curr_feature)
                is_drift = p_value < self.alpha
                
                drift_results[i] = {
                    'method': 'ks_test',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'is_drift': is_drift,
                    'severity': 'high' if p_value < 0.01 else 'medium' if p_value < 0.05 else 'low'
                }
            elif method == 'psi':
                # Population Stability Index
                psi = self._calculate_psi(ref_feature, curr_feature)
                is_drift = psi > 0.2  # PSI > 0.2 indicates significant drift
                
                drift_results[i] = {
                    'method': 'psi',
                    'psi': float(psi),
                    'is_drift': is_drift,
                    'severity': 'high' if psi > 0.25 else 'medium' if psi > 0.2 else 'low'
                }
            else:
                # Simple statistical comparison
                ref_mean = np.mean(ref_feature)
                curr_mean = np.mean(curr_feature)
                ref_std = np.std(ref_feature)
                
                # Z-score based drift detection
                z_score = abs(curr_mean - ref_mean) / (ref_std + 1e-10)
                is_drift = z_score > 2.0  # 2 standard deviations
                
                drift_results[i] = {
                    'method': 'z_score',
                    'z_score': float(z_score),
                    'is_drift': is_drift,
                    'severity': 'high' if z_score > 3.0 else 'medium' if z_score > 2.0 else 'low'
                }
            
            if drift_results[i]['is_drift']:
                has_drift = True
        
        return {
            'has_drift': has_drift,
            'drift_by_feature': drift_results,
            'n_features_checked': len(drift_results),
            'n_features_with_drift': sum(1 for r in drift_results.values() if r['is_drift']),
            'method': method,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        # Create bins based on reference data
        min_val = min(np.min(reference), np.min(current))
        max_val = max(np.max(reference), np.max(current))
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        curr_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize to probabilities
        ref_prob = ref_hist / (len(reference) + 1e-10)
        curr_prob = curr_hist / (len(current) + 1e-10)
        
        # Calculate PSI
        psi = 0.0
        for i in range(len(ref_prob)):
            if ref_prob[i] > 0:
                psi += (curr_prob[i] - ref_prob[i]) * np.log(curr_prob[i] / (ref_prob[i] + 1e-10))
        
        return abs(psi)


class ConceptDriftDetector:
    """
    Detect concept drift (target relationship changes)
    
    Monitors model performance degradation over time
    """
    
    def __init__(self, baseline_performance: float, threshold: float = 0.1):
        """
        Args:
            baseline_performance: Baseline performance metric (e.g., accuracy)
            threshold: Performance degradation threshold (e.g., 0.1 = 10% drop)
        """
        self.baseline_performance = baseline_performance
        self.threshold = threshold
        self.performance_history = deque(maxlen=100)
    
    def detect_drift(
        self,
        current_performance: float,
        metric_name: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Detect concept drift based on performance degradation
        
        Args:
            current_performance: Current performance metric
            metric_name: Name of metric ('accuracy', 'f1', 'mse', etc.)
            
        Returns:
            Dictionary with drift detection results
        """
        performance_drop = self.baseline_performance - current_performance
        relative_drop = performance_drop / (self.baseline_performance + 1e-10)
        
        has_drift = relative_drop > self.threshold
        
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'performance': current_performance,
            'metric': metric_name
        })
        
        return {
            'has_drift': has_drift,
            'baseline_performance': self.baseline_performance,
            'current_performance': current_performance,
            'performance_drop': float(performance_drop),
            'relative_drop': float(relative_drop),
            'threshold': self.threshold,
            'metric': metric_name,
            'severity': 'high' if relative_drop > 0.2 else 'medium' if relative_drop > 0.1 else 'low',
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Detect performance trend over time
        
        Args:
            window_size: Size of window for trend analysis
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self.performance_history) < window_size:
            return {'error': 'Not enough history for trend analysis'}
        
        recent_performances = [p['performance'] for p in list(self.performance_history)[-window_size:]]
        
        # Linear regression to detect trend
        x = np.arange(len(recent_performances))
        slope = np.polyfit(x, recent_performances, 1)[0]
        
        is_degrading = slope < -0.01  # Negative slope indicates degradation
        
        return {
            'is_degrading': is_degrading,
            'slope': float(slope),
            'trend': 'degrading' if is_degrading else 'stable' if abs(slope) < 0.01 else 'improving',
            'window_size': window_size,
            'recent_performances': recent_performances
        }


class PerformanceMonitor:
    """
    Monitor model performance metrics
    
    Tracks accuracy, latency, throughput, and other metrics
    """
    
    def __init__(self, model_name: str = 'default'):
        """
        Args:
            model_name: Name of the model being monitored
        """
        self.model_name = model_name
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.latency_history = deque(maxlen=1000)
        self.timestamp_history = deque(maxlen=1000)
    
    def record_prediction(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        latency: Optional[float] = None,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Record a prediction and compute metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            latency: Prediction latency in seconds
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with computed metrics
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        timestamp = datetime.now()
        
        # Compute metrics
        if task_type == 'classification':
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
        else:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2)
            }
        
        # Store metrics
        for metric_name, metric_value in metrics.items():
            self.metrics_history[metric_name].append(metric_value)
        
        if latency is not None:
            self.latency_history.append(latency)
        
        self.timestamp_history.append(timestamp)
        
        return {
            'metrics': metrics,
            'latency': latency,
            'timestamp': timestamp.isoformat(),
            'n_samples': len(y_true)
        }
    
    def get_summary(self, window_size: int = 100) -> Dict[str, Any]:
        """
        Get performance summary
        
        Args:
            window_size: Size of window for summary
            
        Returns:
            Dictionary with performance summary
        """
        summary = {}
        
        # Metric summaries
        for metric_name, history in self.metrics_history.items():
            if len(history) > 0:
                recent = list(history)[-window_size:]
                summary[metric_name] = {
                    'current': float(recent[-1]) if recent else None,
                    'mean': float(np.mean(recent)) if recent else None,
                    'std': float(np.std(recent)) if recent else None,
                    'min': float(np.min(recent)) if recent else None,
                    'max': float(np.max(recent)) if recent else None
                }
        
        # Latency summary
        if len(self.latency_history) > 0:
            recent_latency = list(self.latency_history)[-window_size:]
            summary['latency'] = {
                'mean': float(np.mean(recent_latency)),
                'std': float(np.std(recent_latency)),
                'min': float(np.min(recent_latency)),
                'max': float(np.max(recent_latency)),
                'p95': float(np.percentile(recent_latency, 95)),
                'p99': float(np.percentile(recent_latency, 99))
            }
        
        # Throughput (predictions per second)
        if len(self.timestamp_history) > 1:
            time_span = (self.timestamp_history[-1] - self.timestamp_history[0]).total_seconds()
            if time_span > 0:
                throughput = len(self.timestamp_history) / time_span
                summary['throughput'] = {
                    'predictions_per_second': float(throughput)
                }
        
        summary['n_predictions'] = len(self.timestamp_history)
        summary['model_name'] = self.model_name
        
        return summary


class ModelMonitor:
    """
    Comprehensive model monitoring system
    
    Combines data drift, concept drift, and performance monitoring
    """
    
    def __init__(
        self,
        model: Any,
        reference_data: np.ndarray,
        reference_labels: Optional[np.ndarray] = None,
        baseline_performance: Optional[float] = None,
        model_name: str = 'default'
    ):
        """
        Args:
            model: Trained model
            reference_data: Reference (training) data
            reference_labels: Reference labels (for baseline performance)
            baseline_performance: Baseline performance metric
            model_name: Name of the model
        """
        self.model = model
        self.model_name = model_name
        
        # Initialize detectors
        self.data_drift_detector = DataDriftDetector(reference_data)
        
        # Compute baseline performance if not provided
        if baseline_performance is None and reference_labels is not None:
            if hasattr(model, 'predict'):
                ref_predictions = model.predict(reference_data)
                if SKLEARN_AVAILABLE:
                    baseline_performance = accuracy_score(reference_labels, ref_predictions)
                else:
                    baseline_performance = np.mean(reference_labels == ref_predictions)
        
        if baseline_performance is None:
            baseline_performance = 0.9  # Default
        
        self.concept_drift_detector = ConceptDriftDetector(baseline_performance)
        self.performance_monitor = PerformanceMonitor(model_name)
        
        self.alerts = deque(maxlen=100)
    
    def monitor(
        self,
        current_data: np.ndarray,
        current_labels: Optional[np.ndarray] = None,
        check_data_drift: bool = True,
        check_concept_drift: bool = True,
        check_performance: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive monitoring
        
        Args:
            current_data: Current data to monitor
            current_labels: Current labels (for performance monitoring)
            check_data_drift: Whether to check data drift
            check_concept_drift: Whether to check concept drift
            check_performance: Whether to check performance
            
        Returns:
            Dictionary with monitoring results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name
        }
        
        # Data drift detection
        if check_data_drift:
            data_drift_result = self.data_drift_detector.detect_drift(current_data)
            results['data_drift'] = data_drift_result
            
            if data_drift_result['has_drift']:
                self._create_alert('data_drift', data_drift_result)
        
        # Concept drift and performance monitoring
        if check_concept_drift or check_performance:
            if current_labels is not None and hasattr(self.model, 'predict'):
                start_time = time.time()
                predictions = self.model.predict(current_data)
                latency = time.time() - start_time
                
                # Performance monitoring
                if check_performance:
                    perf_result = self.performance_monitor.record_prediction(
                        current_labels, predictions, latency
                    )
                    results['performance'] = perf_result
                
                # Concept drift detection
                if check_concept_drift:
                    if SKLEARN_AVAILABLE:
                        current_performance = accuracy_score(current_labels, predictions)
                    else:
                        current_performance = np.mean(current_labels == predictions)
                    
                    concept_drift_result = self.concept_drift_detector.detect_drift(
                        current_performance
                    )
                    results['concept_drift'] = concept_drift_result
                    
                    if concept_drift_result['has_drift']:
                        self._create_alert('concept_drift', concept_drift_result)
        
        return results
    
    def _create_alert(self, alert_type: str, details: Dict[str, Any]):
        """Create an alert"""
        alert = {
            'type': alert_type,
            'severity': details.get('severity', 'medium'),
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.alerts.append(alert)
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get alerts
        
        Args:
            severity: Filter by severity ('high', 'medium', 'low')
            
        Returns:
            List of alerts
        """
        alerts = list(self.alerts)
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        return alerts
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        return {
            'model_name': self.model_name,
            'performance_summary': self.performance_monitor.get_summary(),
            'concept_drift_trend': self.concept_drift_detector.detect_trend(),
            'n_alerts': len(self.alerts),
            'recent_alerts': list(self.alerts)[-10:],
            'timestamp': datetime.now().isoformat()
        }
