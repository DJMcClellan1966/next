"""
Interactive Visualization Dashboard
Advanced dashboard with Plotly charts, real-time updates, and rich visualizations

Features:
- Interactive charts (Plotly)
- Real-time experiment monitoring
- Model performance visualization
- Hyperparameter sensitivity analysis
- Feature importance plots
- Training curves with zoom/pan
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import json
import datetime
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))


class InteractiveDashboard:
    """
    Interactive Visualization Dashboard
    
    Advanced dashboard with Plotly charts and real-time updates
    """
    
    def __init__(self, storage_path: str = "experiments.json"):
        """
        Args:
            storage_path: Path to store experiments
        """
        self.storage_path = storage_path
        self.experiments = []
        self._load_experiments()
        self._check_plotly()
    
    def _check_plotly(self):
        """Check if Plotly is available"""
        try:
            import plotly
            import plotly.graph_objects as go
            import plotly.express as px
            self.plotly_available = True
        except ImportError:
            self.plotly_available = False
            warnings.warn("Plotly not available. Install with: pip install plotly")
    
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
        model_info: Optional[Dict[str, Any]] = None,
        training_history: Optional[Dict[str, List[float]]] = None
    ) -> str:
        """
        Log experiment with training history
        
        Args:
            name: Experiment name
            metrics: Dictionary of metrics
            parameters: Hyperparameters
            model_info: Model information
            training_history: Training history (loss, accuracy over epochs)
            
        Returns:
            Experiment ID
        """
        experiment = {
            'id': f"exp_{len(self.experiments)}",
            'name': name,
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': metrics,
            'parameters': parameters,
            'model_info': model_info or {},
            'training_history': training_history or {}
        }
        
        self.experiments.append(experiment)
        self._save_experiments()
        return experiment['id']
    
    def generate_interactive_dashboard(self) -> str:
        """
        Generate interactive HTML dashboard with Plotly charts
        
        Returns:
            HTML string with interactive charts
        """
        if not self.plotly_available:
            return self._generate_basic_dashboard()
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        
        # Create dashboard HTML
        html_parts = []
        
        # Header
        html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <title>ML Toolbox - Interactive Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .dashboard-container { max-width: 1400px; margin: 0 auto; }
        .section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        h2 { color: #555; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        .chart-container { margin: 20px 0; }
        .experiment-card { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric-card { background: #f0f0f0; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .metric-label { font-size: 12px; color: #666; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <h1>ML Toolbox - Interactive Experiment Dashboard</h1>
""")
        
        # Summary metrics
        if self.experiments:
            html_parts.append(self._generate_summary_metrics())
        
        # Training curves
        if any(exp.get('training_history') for exp in self.experiments):
            html_parts.append(self._generate_training_curves())
        
        # Metrics comparison
        if len(self.experiments) > 1:
            html_parts.append(self._generate_metrics_comparison())
        
        # Hyperparameter analysis
        if len(self.experiments) > 1:
            html_parts.append(self._generate_hyperparameter_analysis())
        
        # Experiment list
        html_parts.append(self._generate_experiment_list())
        
        # Footer
        html_parts.append("""
    </div>
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function(){ location.reload(); }, 30000);
    </script>
</body>
</html>
""")
        
        return ''.join(html_parts)
    
    def _generate_summary_metrics(self) -> str:
        """Generate summary metrics cards"""
        if not self.experiments:
            return ""
        
        # Get best experiment
        best_exp = max(self.experiments, 
                      key=lambda x: x.get('metrics', {}).get('accuracy', 0))
        
        metrics_html = '<div class="section"><h2>Summary Metrics</h2><div class="metrics-grid">'
        
        for metric_name, value in best_exp.get('metrics', {}).items():
            metrics_html += f"""
            <div class="metric-card">
                <div class="metric-value">{value:.4f}</div>
                <div class="metric-label">{metric_name}</div>
            </div>
            """
        
        metrics_html += '</div></div>'
        return metrics_html
    
    def _generate_training_curves(self) -> str:
        """Generate interactive training curves"""
        if not self.plotly_available:
            return ""
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training Loss', 'Training Accuracy'),
            vertical_spacing=0.1
        )
        
        # Add curves for each experiment
        for exp in self.experiments:
            history = exp.get('training_history', {})
            if not history:
                continue
            
            epochs = list(range(1, len(history.get('loss', [])) + 1))
            
            # Loss curve
            if 'loss' in history:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history['loss'],
                        mode='lines+markers',
                        name=f"{exp['name']} - Loss",
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
            
            # Accuracy curve
            if 'accuracy' in history:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history['accuracy'],
                        mode='lines+markers',
                        name=f"{exp['name']} - Accuracy",
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
        
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        fig.update_layout(height=600, showlegend=True, title_text="Training Curves")
        
        chart_html = fig.to_html(include_plotlyjs='cdn', div_id='training-curves')
        
        return f'<div class="section"><h2>Training Curves</h2><div class="chart-container">{chart_html}</div></div>'
    
    def _generate_metrics_comparison(self) -> str:
        """Generate metrics comparison chart"""
        if not self.plotly_available or len(self.experiments) < 2:
            return ""
        
        import plotly.graph_objects as go
        
        # Get all unique metrics
        all_metrics = set()
        for exp in self.experiments:
            all_metrics.update(exp.get('metrics', {}).keys())
        
        if not all_metrics:
            return ""
        
        # Create bar chart for each metric
        charts_html = []
        for metric_name in all_metrics:
            exp_names = [exp['name'] for exp in self.experiments]
            metric_values = [exp.get('metrics', {}).get(metric_name, 0) 
                           for exp in self.experiments]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=exp_names,
                    y=metric_values,
                    marker_color='#4CAF50',
                    text=metric_values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f'{metric_name} Comparison',
                xaxis_title='Experiment',
                yaxis_title=metric_name,
                height=400
            )
            
            chart_html = fig.to_html(include_plotlyjs=False, div_id=f'metric-{metric_name}')
            charts_html.append(f'<div class="chart-container">{chart_html}</div>')
        
        return f'<div class="section"><h2>Metrics Comparison</h2>{"".join(charts_html)}</div>'
    
    def _generate_hyperparameter_analysis(self) -> str:
        """Generate hyperparameter sensitivity analysis"""
        if not self.plotly_available or len(self.experiments) < 2:
            return ""
        
        import plotly.graph_objects as go
        
        # Get all unique parameters
        all_params = set()
        for exp in self.experiments:
            all_params.update(exp.get('parameters', {}).keys())
        
        if not all_params:
            return ""
        
        # Get primary metric (accuracy or first metric)
        primary_metric = 'accuracy'
        if not any(primary_metric in exp.get('metrics', {}) for exp in self.experiments):
            primary_metric = list(self.experiments[0].get('metrics', {}).keys())[0] if self.experiments else 'accuracy'
        
        charts_html = []
        for param_name in list(all_params)[:5]:  # Limit to 5 parameters
            param_values = []
            metric_values = []
            
            for exp in self.experiments:
                param_val = exp.get('parameters', {}).get(param_name)
                metric_val = exp.get('metrics', {}).get(primary_metric, 0)
                
                if param_val is not None:
                    param_values.append(str(param_val))
                    metric_values.append(metric_val)
            
            if param_values:
                fig = go.Figure(data=[
                    go.Scatter(
                        x=param_values,
                        y=metric_values,
                        mode='markers+lines',
                        marker=dict(size=10, color='#4CAF50'),
                        line=dict(width=2)
                    )
                ])
                
                fig.update_layout(
                    title=f'{param_name} vs {primary_metric}',
                    xaxis_title=param_name,
                    yaxis_title=primary_metric,
                    height=400
                )
                
                chart_html = fig.to_html(include_plotlyjs=False, div_id=f'param-{param_name}')
                charts_html.append(f'<div class="chart-container">{chart_html}</div>')
        
        return f'<div class="section"><h2>Hyperparameter Sensitivity Analysis</h2>{"".join(charts_html)}</div>'
    
    def _generate_experiment_list(self) -> str:
        """Generate experiment list"""
        experiments_html = '<div class="section"><h2>Experiments</h2>'
        
        for exp in self.experiments[-10:]:  # Last 10 experiments
            metrics_html = "".join([
                f'<span style="margin: 0 10px;"><strong>{k}:</strong> {v:.4f}</span>'
                for k, v in exp.get('metrics', {}).items()
            ])
            
            experiments_html += f"""
            <div class="experiment-card">
                <h3>{exp['name']} ({exp['id']})</h3>
                <p><strong>Timestamp:</strong> {exp['timestamp']}</p>
                <p><strong>Metrics:</strong> {metrics_html}</p>
                <details>
                    <summary>Parameters</summary>
                    <pre>{json.dumps(exp.get('parameters', {}), indent=2)}</pre>
                </details>
            </div>
            """
        
        experiments_html += '</div>'
        return experiments_html
    
    def _generate_basic_dashboard(self) -> str:
        """Generate basic dashboard without Plotly"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>ML Toolbox - Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .experiment { border: 1px solid #ddd; padding: 15px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>ML Toolbox - Dashboard</h1>
    <p>Install Plotly for interactive charts: pip install plotly</p>
</body>
</html>
"""
    
    def save_dashboard(self, output_path: str = "interactive_dashboard.html"):
        """Save interactive dashboard to file"""
        html = self.generate_interactive_dashboard()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return output_path
    
    def get_experiments(self, filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get experiments with filtering"""
        results = self.experiments.copy()
        
        if filter_by:
            for key, value in filter_by.items():
                results = [exp for exp in results 
                          if exp.get('parameters', {}).get(key) == value]
        
        return results
    
    def get_best_experiment(self, metric: str = 'accuracy') -> Optional[Dict[str, Any]]:
        """Get best experiment by metric"""
        if not self.experiments:
            return None
        
        best = max(self.experiments, 
                  key=lambda x: x.get('metrics', {}).get(metric, 0))
        return best
