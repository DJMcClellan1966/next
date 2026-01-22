"""
Experiment Tracking
Burkov Machine Learning Engineering - Experiment Management

Features:
- Experiment logging
- Parameter tracking
- Metric tracking
- Model artifacts storage
- Experiment comparison
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import json
import pickle
from datetime import datetime
from collections import defaultdict
import hashlib
import os

sys.path.insert(0, str(Path(__file__).parent))


class Experiment:
    """Single experiment record"""
    
    def __init__(
        self,
        experiment_id: str,
        experiment_name: str,
        description: Optional[str] = None
    ):
        """
        Args:
            experiment_id: Unique experiment ID
            experiment_name: Name of the experiment
            description: Experiment description
        """
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.description = description
        self.created_at = datetime.now()
        
        # Tracking data
        self.parameters: Dict[str, Any] = {}
        self.metrics: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.artifacts: Dict[str, str] = {}  # artifact_name -> file_path
        self.tags: List[str] = []
        self.status: str = 'running'  # 'running', 'completed', 'failed'
        self.model_path: Optional[str] = None
        self.notes: List[str] = []
    
    def log_parameter(self, name: str, value: Any):
        """Log a parameter"""
        self.parameters[name] = value
    
    def log_parameters(self, parameters: Dict[str, Any]):
        """Log multiple parameters"""
        self.parameters.update(parameters)
    
    def log_metric(self, name: str, value: float, timestamp: Optional[datetime] = None):
        """Log a metric value"""
        if timestamp is None:
            timestamp = datetime.now()
        self.metrics[name].append((timestamp, value))
    
    def log_metrics(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None):
        """Log multiple metrics"""
        if timestamp is None:
            timestamp = datetime.now()
        for name, value in metrics.items():
            self.metrics[name].append((timestamp, value))
    
    def save_model(self, model: Any, model_name: str = 'model.pkl'):
        """
        Save model artifact
        
        Args:
            model: Model object
            model_name: Name for the model file
        """
        # Create artifacts directory
        artifacts_dir = Path(f"experiments/{self.experiment_id}/artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = artifacts_dir / model_name
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.model_path = str(model_path)
        self.artifacts['model'] = self.model_path
    
    def add_tag(self, tag: str):
        """Add a tag"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def add_note(self, note: str):
        """Add a note"""
        self.notes.append(f"[{datetime.now().isoformat()}] {note}")
    
    def complete(self):
        """Mark experiment as completed"""
        self.status = 'completed'
    
    def fail(self, reason: Optional[str] = None):
        """Mark experiment as failed"""
        self.status = 'failed'
        if reason:
            self.add_note(f"Failed: {reason}")
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest metric values"""
        latest = {}
        for metric_name, values in self.metrics.items():
            if values:
                latest[metric_name] = values[-1][1]  # Last value
        return latest
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'parameters': self.parameters,
            'latest_metrics': self.get_latest_metrics(),
            'tags': self.tags,
            'n_notes': len(self.notes),
            'model_path': self.model_path
        }


class ExperimentTracker:
    """
    Experiment tracking system
    
    Manages multiple experiments
    """
    
    def __init__(self, storage_dir: str = "experiments"):
        """
        Args:
            storage_dir: Directory to store experiments
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiments: List[str] = []
    
    def create_experiment(
        self,
        experiment_name: str,
        description: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> Experiment:
        """
        Create a new experiment
        
        Args:
            experiment_name: Name of the experiment
            description: Experiment description
            experiment_id: Optional experiment ID (auto-generated if None)
            
        Returns:
            Experiment object
        """
        if experiment_id is None:
            # Generate unique ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hash_str = hashlib.md5(f"{experiment_name}_{timestamp}".encode()).hexdigest()[:8]
            experiment_id = f"{experiment_name}_{timestamp}_{hash_str}"
        
        experiment = Experiment(experiment_id, experiment_name, description)
        self.experiments[experiment_id] = experiment
        self.active_experiments.append(experiment_id)
        
        # Create experiment directory
        exp_dir = self.storage_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Experiment:
        """Get experiment by ID"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        return self.experiments[experiment_id]
    
    def list_experiments(
        self,
        status: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List experiments
        
        Args:
            status: Filter by status ('running', 'completed', 'failed')
            tag: Filter by tag
            
        Returns:
            List of experiment dictionaries
        """
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        if tag:
            experiments = [e for e in experiments if tag in e.tags]
        
        return [e.to_dict() for e in experiments]
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Compare experiments on a metric
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metric_name: Metric to compare
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                continue
            
            experiment = self.experiments[exp_id]
            latest_metrics = experiment.get_latest_metrics()
            
            comparison[exp_id] = {
                'experiment_name': experiment.experiment_name,
                'metric_value': latest_metrics.get(metric_name),
                'parameters': experiment.parameters,
                'status': experiment.status
            }
        
        # Find best experiment
        valid_comparisons = {k: v for k, v in comparison.items() if v['metric_value'] is not None}
        if valid_comparisons:
            best_exp_id = max(valid_comparisons.items(), key=lambda x: x[1]['metric_value'])[0]
            comparison['best_experiment'] = best_exp_id
        
        return comparison
    
    def save_experiment(self, experiment_id: str):
        """Save experiment to disk"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        exp_dir = self.storage_dir / experiment_id
        
        # Save experiment metadata
        metadata_path = exp_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(experiment.to_dict(), f, indent=2, default=str)
    
    def load_experiment(self, experiment_id: str) -> Experiment:
        """Load experiment from disk"""
        exp_dir = self.storage_dir / experiment_id
        metadata_path = exp_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise ValueError(f"Experiment {experiment_id} not found on disk")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        experiment = Experiment(
            metadata['experiment_id'],
            metadata['experiment_name'],
            metadata.get('description')
        )
        
        experiment.parameters = metadata.get('parameters', {})
        experiment.status = metadata.get('status', 'completed')
        experiment.tags = metadata.get('tags', [])
        experiment.model_path = metadata.get('model_path')
        
        # Load metrics (simplified - would need full reconstruction)
        
        self.experiments[experiment_id] = experiment
        return experiment
    
    def get_best_experiment(
        self,
        metric_name: str,
        maximize: bool = True
    ) -> Optional[Experiment]:
        """
        Get best experiment by metric
        
        Args:
            metric_name: Metric to optimize
            maximize: Whether to maximize (True) or minimize (False)
            
        Returns:
            Best experiment or None
        """
        best_experiment = None
        best_value = None
        
        for experiment in self.experiments.values():
            latest_metrics = experiment.get_latest_metrics()
            if metric_name in latest_metrics:
                value = latest_metrics[metric_name]
                
                if best_value is None:
                    best_value = value
                    best_experiment = experiment
                elif maximize and value > best_value:
                    best_value = value
                    best_experiment = experiment
                elif not maximize and value < best_value:
                    best_value = value
                    best_experiment = experiment
        
        return best_experiment
