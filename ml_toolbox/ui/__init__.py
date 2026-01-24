"""
ML Toolbox UI Module
Experiment tracking UI and interactive dashboard
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ml_toolbox.ui.experiment_tracking_ui import ExperimentTrackingUI
    from ml_toolbox.ui.interactive_dashboard import InteractiveDashboard
    from ml_toolbox.ui.dashboard_components import (
        DashboardComponent, MetricCard, ChartComponent, TableComponent,
        DashboardLayout, create_wellness_dashboard, get_dashboard_component
    )
    __all__ = [
        'ExperimentTrackingUI', 
        'InteractiveDashboard',
        'DashboardComponent',
        'MetricCard',
        'ChartComponent',
        'TableComponent',
        'DashboardLayout',
        'create_wellness_dashboard',
        'get_dashboard_component'
    ]
except ImportError as e:
    __all__ = []
    import warnings
    warnings.warn(f"UI module imports failed: {e}")
