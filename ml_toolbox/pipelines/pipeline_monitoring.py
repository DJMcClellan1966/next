"""
Pipeline Monitoring

Enhanced monitoring for pipeline stages with detailed metrics tracking,
performance monitoring, and debugging capabilities.
"""
from typing import Any, Dict, Optional, List
import logging
import time
from datetime import datetime
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class PipelineMetrics:
    """Metrics for a single pipeline execution"""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.stage_metrics: Dict[str, Dict[str, Any]] = {}
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.data_sizes: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
    
    def end(self):
        """End timing"""
        self.end_time = time.time()
        if self.start_time:
            self.performance_metrics['total_duration'] = self.end_time - self.start_time
    
    def add_stage_metric(self, stage_name: str, metric_name: str, value: Any):
        """Add metric for a stage"""
        if stage_name not in self.stage_metrics:
            self.stage_metrics[stage_name] = {}
        self.stage_metrics[stage_name][metric_name] = value
    
    def add_error(self, stage_name: str, error: Exception, context: Optional[Dict] = None):
        """Add error record"""
        self.errors.append({
            'stage': stage_name,
            'error': str(error),
            'error_type': type(error).__name__,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        })
    
    def add_warning(self, stage_name: str, message: str, context: Optional[Dict] = None):
        """Add warning record"""
        self.warnings.append({
            'stage': stage_name,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        })
    
    def record_data_size(self, stage_name: str, data: Any):
        """Record data size at a stage"""
        if hasattr(data, 'shape'):
            self.data_sizes[stage_name] = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'size_bytes': data.nbytes if hasattr(data, 'nbytes') else None
            }
        elif hasattr(data, '__len__'):
            self.data_sizes[stage_name] = {
                'length': len(data),
                'type': type(data).__name__
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            'pipeline_name': self.pipeline_name,
            'duration': self.performance_metrics.get('total_duration'),
            'stages_count': len(self.stage_metrics),
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings),
            'stage_metrics': self.stage_metrics,
            'performance_metrics': self.performance_metrics,
            'data_sizes': self.data_sizes,
            'errors': self.errors,
            'warnings': self.warnings
        }


class PipelineMonitor:
    """
    Enhanced pipeline monitoring
    
    Tracks:
    - Stage-level metrics
    - Performance metrics
    - Data flow through stages
    - Errors and warnings
    - Resource usage
    """
    
    def __init__(self, enable_tracking: bool = True):
        """
        Initialize pipeline monitor
        
        Parameters
        ----------
        enable_tracking : bool, default=True
            Whether to enable metrics tracking
        """
        self.enable_tracking = enable_tracking
        self.metrics_history: List[PipelineMetrics] = []
        self.stage_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.current_metrics: Optional[PipelineMetrics] = None
    
    def start_pipeline(self, pipeline_name: str) -> PipelineMetrics:
        """Start monitoring a pipeline execution"""
        metrics = PipelineMetrics(pipeline_name)
        metrics.start()
        self.current_metrics = metrics
        return metrics
    
    def end_pipeline(self, metrics: Optional[PipelineMetrics] = None):
        """End monitoring a pipeline execution"""
        if metrics is None:
            metrics = self.current_metrics
        
        if metrics:
            metrics.end()
            self.metrics_history.append(metrics)
            
            # Update stage statistics
            for stage_name, stage_data in metrics.stage_metrics.items():
                if stage_name not in self.stage_stats:
                    self.stage_stats[stage_name] = {
                        'execution_count': 0,
                        'total_duration': 0.0,
                        'error_count': 0,
                        'success_count': 0
                    }
                
                self.stage_stats[stage_name]['execution_count'] += 1
                if 'duration' in stage_data:
                    self.stage_stats[stage_name]['total_duration'] += stage_data['duration']
                if metrics.errors:
                    error_count = sum(1 for e in metrics.errors if e['stage'] == stage_name)
                    self.stage_stats[stage_name]['error_count'] += error_count
                    if error_count == 0:
                        self.stage_stats[stage_name]['success_count'] += 1
    
    def track_stage(self, stage_name: str, start_time: float, end_time: float,
                   input_data: Optional[Any] = None, output_data: Optional[Any] = None,
                   success: bool = True, error: Optional[Exception] = None):
        """Track a stage execution"""
        if not self.enable_tracking or not self.current_metrics:
            return
        
        duration = end_time - start_time
        self.current_metrics.add_stage_metric(stage_name, 'duration', duration)
        self.current_metrics.add_stage_metric(stage_name, 'success', success)
        self.current_metrics.add_stage_metric(stage_name, 'start_time', start_time)
        self.current_metrics.add_stage_metric(stage_name, 'end_time', end_time)
        
        if input_data is not None:
            self.current_metrics.record_data_size(f"{stage_name}_input", input_data)
        if output_data is not None:
            self.current_metrics.record_data_size(f"{stage_name}_output", output_data)
        
        if error:
            self.current_metrics.add_error(stage_name, error)
        elif not success:
            self.current_metrics.add_warning(stage_name, "Stage completed with warnings")
    
    def get_stage_statistics(self, stage_name: str) -> Dict[str, Any]:
        """Get statistics for a specific stage"""
        if stage_name not in self.stage_stats:
            return {}
        
        stats = self.stage_stats[stage_name].copy()
        if stats['execution_count'] > 0:
            stats['average_duration'] = stats['total_duration'] / stats['execution_count']
            stats['success_rate'] = stats['success_count'] / stats['execution_count']
        
        return stats
    
    def get_pipeline_statistics(self, pipeline_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for pipelines"""
        if pipeline_name:
            relevant_metrics = [m for m in self.metrics_history if m.pipeline_name == pipeline_name]
        else:
            relevant_metrics = self.metrics_history
        
        if not relevant_metrics:
            return {}
        
        durations = [m.performance_metrics.get('total_duration', 0) for m in relevant_metrics if m.performance_metrics.get('total_duration')]
        error_counts = [len(m.errors) for m in relevant_metrics]
        
        return {
            'execution_count': len(relevant_metrics),
            'average_duration': np.mean(durations) if durations else 0,
            'min_duration': np.min(durations) if durations else 0,
            'max_duration': np.max(durations) if durations else 0,
            'total_errors': sum(error_counts),
            'average_errors': np.mean(error_counts) if error_counts else 0,
            'stage_statistics': {name: self.get_stage_statistics(name) for name in self.stage_stats.keys()}
        }
    
    def get_recent_metrics(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent pipeline execution metrics"""
        recent = self.metrics_history[-count:] if len(self.metrics_history) > count else self.metrics_history
        return [m.get_summary() for m in recent]
    
    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history = []
        self.stage_stats = defaultdict(dict)
        logger.info("[PipelineMonitor] Metrics history cleared")
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file"""
        import json
        
        export_data = {
            'metrics_history': [m.get_summary() for m in self.metrics_history],
            'stage_statistics': dict(self.stage_stats),
            'export_timestamp': datetime.now().isoformat()
        }
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"[PipelineMonitor] Metrics exported to {filepath}")
        else:
            raise ValueError(f"Unsupported format: {format}")
