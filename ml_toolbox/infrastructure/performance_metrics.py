"""
Performance Metrics System
Extracted from Lighthouse repository concepts

Features:
- Automated performance auditing
- Performance metrics tracking
- Best practices validation
- Performance optimization suggestions
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import time
import json
import datetime
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class PerformanceMetric:
    """Represents a single performance metric"""
    
    def __init__(self, name: str, value: float, unit: str = "", 
                 threshold: Optional[float] = None, category: str = "general"):
        """
        Initialize performance metric
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            threshold: Performance threshold
            category: Metric category
        """
        self.name = name
        self.value = value
        self.unit = unit
        self.threshold = threshold
        self.category = category
        self.timestamp = datetime.datetime.now()
    
    def is_acceptable(self) -> bool:
        """Check if metric meets threshold"""
        if self.threshold is None:
            return True
        
        # Lower is better for most metrics
        return self.value <= self.threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'threshold': self.threshold,
            'category': self.category,
            'acceptable': self.is_acceptable(),
            'timestamp': self.timestamp.isoformat()
        }


class PerformanceAudit:
    """Represents a performance audit with multiple metrics"""
    
    def __init__(self, audit_name: str, target: str = ""):
        """
        Initialize performance audit
        
        Args:
            audit_name: Name of the audit
            target: Target being audited (e.g., "model_training", "data_processing")
        """
        self.audit_name = audit_name
        self.target = target
        self.metrics: List[PerformanceMetric] = []
        self.start_time = None
        self.end_time = None
        self.score = 0.0
    
    def start(self):
        """Start audit timing"""
        self.start_time = time.time()
    
    def end(self):
        """End audit timing"""
        self.end_time = time.time()
        if self.start_time:
            duration = self.end_time - self.start_time
            self.add_metric("duration", duration, "seconds")
    
    def add_metric(self, name: str, value: float, unit: str = "", 
                   threshold: Optional[float] = None, category: str = "general"):
        """Add metric to audit"""
        metric = PerformanceMetric(name, value, unit, threshold, category)
        self.metrics.append(metric)
    
    def calculate_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        if not self.metrics:
            return 0.0
        
        acceptable_count = sum(1 for m in self.metrics if m.is_acceptable())
        self.score = (acceptable_count / len(self.metrics)) * 100
        return self.score
    
    def get_recommendations(self) -> List[str]:
        """Get performance improvement recommendations"""
        recommendations = []
        
        for metric in self.metrics:
            if not metric.is_acceptable():
                if metric.category == "speed":
                    recommendations.append(
                        f"Optimize {metric.name}: Current {metric.value}{metric.unit}, "
                        f"target: {metric.threshold}{metric.unit}"
                    )
                elif metric.category == "memory":
                    recommendations.append(
                        f"Reduce memory usage for {metric.name}: "
                        f"Current {metric.value}{metric.unit}"
                    )
                elif metric.category == "accuracy":
                    recommendations.append(
                        f"Improve {metric.name}: Current {metric.value}{metric.unit}, "
                        f"target: {metric.threshold}{metric.unit}"
                    )
        
        return recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit to dictionary"""
        return {
            'audit_name': self.audit_name,
            'target': self.target,
            'score': self.calculate_score(),
            'metrics': [m.to_dict() for m in self.metrics],
            'recommendations': self.get_recommendations(),
            'duration': (self.end_time - self.start_time) if self.start_time and self.end_time else None
        }


class PerformanceMonitor:
    """
    Performance Monitoring System
    
    Tracks and audits performance metrics
    """
    
    def __init__(self):
        """Initialize performance monitor"""
        self.audits: List[PerformanceAudit] = []
        self.metric_history: Dict[str, List[float]] = {}
    
    def create_audit(self, audit_name: str, target: str = "") -> PerformanceAudit:
        """Create a new performance audit"""
        audit = PerformanceAudit(audit_name, target)
        self.audits.append(audit)
        return audit
    
    def track_metric(self, name: str, value: float):
        """Track a metric over time"""
        if name not in self.metric_history:
            self.metric_history[name] = []
        self.metric_history[name].append(value)
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a tracked metric"""
        if name not in self.metric_history or not self.metric_history[name]:
            return {}
        
        values = self.metric_history[name]
        return {
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'count': len(values),
            'latest': values[-1]
        }
    
    def audit_function(self, func: Callable, func_name: str, 
                      *args, **kwargs) -> tuple:
        """
        Audit a function's performance
        
        Args:
            func: Function to audit
            func_name: Name of function
            *args, **kwargs: Function arguments
            
        Returns:
            Tuple of (result, audit)
        """
        audit = self.create_audit(f"function_{func_name}", func_name)
        audit.start()
        
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Add metrics
        audit.add_metric("execution_time", end_time - start_time, "seconds", 
                        threshold=5.0, category="speed")
        audit.add_metric("memory_used", end_memory - start_memory, "MB", 
                        threshold=100.0, category="memory")
        audit.add_metric("success", 1.0 if success else 0.0, "", 
                        threshold=1.0, category="reliability")
        
        audit.end()
        
        return result, audit
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def get_latest_audit(self) -> Optional[PerformanceAudit]:
        """Get most recent audit"""
        return self.audits[-1] if self.audits else None
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of all audits"""
        if not self.audits:
            return {'total_audits': 0}
        
        scores = [audit.calculate_score() for audit in self.audits]
        
        return {
            'total_audits': len(self.audits),
            'average_score': sum(scores) / len(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'recent_audits': [audit.to_dict() for audit in self.audits[-5:]]
        }


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create performance monitor instance"""
    return PerformanceMonitor()
