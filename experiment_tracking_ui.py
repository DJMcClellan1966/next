"""
Experiment Tracking UI
Web-based UI for experiment tracking and visualization

Features:
- Experiment dashboard
- Metrics visualization
- Model comparison
- Experiment search and filtering
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import json
import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))


class ExperimentTrackingUI:
    """
    Experiment Tracking UI
    
    Web-based dashboard for experiment tracking
    """
    
    def __init__(self, storage_path: str = "experiments.json"):
        """
        Args:
            storage_path: Path to store experiments
        """
        self.storage_path = storage_path
        self.experiments = []
        self._load_experiments()
    
    def _load_experiments(self):
        """Load experiments from storage"""
        try:
            if Path(self.storage_path).exists():
                with open(self.storage_path, 'r') as f:
                    self.experiments = json.load(f)
        except:
            self.experiments = []
    
    def _save_experiments(self):
        """Save experiments to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.experiments, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving experiments: {e}")
    
    def log_experiment(
        self,
        name: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        model_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log experiment
        
        Args:
            name: Experiment name
            metrics: Dictionary of metrics
            parameters: Hyperparameters
            model_info: Model information
            
        Returns:
            Experiment ID
        """
        experiment = {
            'id': f"exp_{len(self.experiments)}",
            'name': name,
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': metrics,
            'parameters': parameters,
            'model_info': model_info or {}
        }
        
        self.experiments.append(experiment)
        self._save_experiments()
        return experiment['id']
    
    def get_experiments(
        self,
        filter_by: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get experiments with filtering and sorting
        
        Args:
            filter_by: Filter criteria
            sort_by: Sort by metric name
            limit: Maximum number of results
            
        Returns:
            List of experiments
        """
        results = self.experiments.copy()
        
        # Filter
        if filter_by:
            for key, value in filter_by.items():
                results = [exp for exp in results 
                          if exp.get('parameters', {}).get(key) == value]
        
        # Sort
        if sort_by:
            results.sort(key=lambda x: x.get('metrics', {}).get(sort_by, 0), 
                        reverse=True)
        
        # Limit
        if limit:
            results = results[:limit]
        
        return results
    
    def get_best_experiment(self, metric: str = 'accuracy') -> Optional[Dict[str, Any]]:
        """
        Get best experiment by metric
        
        Args:
            metric: Metric name
            
        Returns:
            Best experiment
        """
        if not self.experiments:
            return None
        
        best = max(self.experiments, 
                  key=lambda x: x.get('metrics', {}).get(metric, 0))
        return best
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare experiments
        
        Args:
            experiment_ids: List of experiment IDs
            
        Returns:
            Comparison results
        """
        experiments = [exp for exp in self.experiments 
                      if exp['id'] in experiment_ids]
        
        if not experiments:
            return {'error': 'No experiments found'}
        
        comparison = {
            'experiments': experiments,
            'metrics_comparison': {},
            'parameters_comparison': {}
        }
        
        # Compare metrics
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp.get('metrics', {}).keys())
        
        for metric in all_metrics:
            comparison['metrics_comparison'][metric] = {
                exp['id']: exp.get('metrics', {}).get(metric, 0)
                for exp in experiments
            }
        
        return comparison
    
    def generate_html_dashboard(self) -> str:
        """
        Generate HTML dashboard
        
        Returns:
            HTML string
        """
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Toolbox - Experiment Tracking</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .experiment {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; }}
        .metrics {{ display: flex; gap: 20px; }}
        .metric {{ padding: 5px 10px; background: #f0f0f0; border-radius: 3px; }}
        .best {{ background: #90EE90; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>ML Toolbox - Experiment Tracking Dashboard</h1>
    <h2>Experiments (Total: {total})</h2>
    {experiments}
    <h2>Best Experiment</h2>
    {best}
</body>
</html>
        """
        
        experiments_html = ""
        for exp in self.experiments[-10:]:  # Last 10 experiments
            metrics_html = "".join([
                f'<div class="metric">{k}: {v:.4f}</div>'
                for k, v in exp.get('metrics', {}).items()
            ])
            experiments_html += f"""
            <div class="experiment">
                <h3>{exp['name']} ({exp['id']})</h3>
                <p><strong>Timestamp:</strong> {exp['timestamp']}</p>
                <div class="metrics">{metrics_html}</div>
                <p><strong>Parameters:</strong> {json.dumps(exp.get('parameters', {}), indent=2)}</p>
            </div>
            """
        
        best = self.get_best_experiment()
        best_html = ""
        if best:
            best_metrics = "".join([
                f'<div class="metric best">{k}: {v:.4f}</div>'
                for k, v in best.get('metrics', {}).items()
            ])
            best_html = f"""
            <div class="experiment">
                <h3>{best['name']} ({best['id']})</h3>
                <div class="metrics">{best_metrics}</div>
            </div>
            """
        
        return html_template.format(
            total=len(self.experiments),
            experiments=experiments_html,
            best=best_html
        )
    
    def save_dashboard(self, output_path: str = "experiment_dashboard.html"):
        """Save HTML dashboard to file"""
        html = self.generate_html_dashboard()
        with open(output_path, 'w') as f:
            f.write(html)
        return output_path
