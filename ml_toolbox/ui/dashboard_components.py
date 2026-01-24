"""
Dashboard Components
Extracted from wellness-dashboard and website repositories

Features:
- Dashboard layouts
- Visualization widgets
- Interactive components
- Web UI frameworks
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import json
import datetime
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("plotly not available. Some visualization features will be limited.")


class DashboardComponent:
    """
    Base dashboard component
    
    Provides common functionality for dashboard widgets
    """
    
    def __init__(self, component_id: str, title: str = ""):
        """
        Initialize dashboard component
        
        Args:
            component_id: Unique identifier
            title: Component title
        """
        self.component_id = component_id
        self.title = title
        self.data = None
        self.config = {}
    
    def update_data(self, data: Any):
        """Update component data"""
        self.data = data
    
    def render(self) -> Dict[str, Any]:
        """Render component (to be implemented by subclasses)"""
        return {
            'component_id': self.component_id,
            'title': self.title,
            'type': 'base'
        }


class MetricCard(DashboardComponent):
    """
    Metric Card Component
    
    Displays a single metric with optional trend indicator
    """
    
    def __init__(self, component_id: str, title: str, value: Any, 
                 trend: Optional[float] = None, unit: str = ""):
        """
        Initialize metric card
        
        Args:
            component_id: Unique identifier
            title: Metric title
            value: Current value
            trend: Trend percentage (positive = up, negative = down)
            unit: Unit of measurement
        """
        super().__init__(component_id, title)
        self.value = value
        self.trend = trend
        self.unit = unit
    
    def render(self) -> Dict[str, Any]:
        """Render metric card"""
        return {
            'component_id': self.component_id,
            'type': 'metric_card',
            'title': self.title,
            'value': self.value,
            'trend': self.trend,
            'unit': self.unit,
            'html': self._generate_html()
        }
    
    def _generate_html(self) -> str:
        """Generate HTML for metric card"""
        trend_class = "trend-up" if self.trend and self.trend > 0 else "trend-down" if self.trend and self.trend < 0 else ""
        trend_icon = "↑" if self.trend and self.trend > 0 else "↓" if self.trend and self.trend < 0 else ""
        
        html = f"""
        <div class="metric-card" id="{self.component_id}">
            <div class="metric-title">{self.title}</div>
            <div class="metric-value">
                {self.value}{self.unit}
                {f'<span class="trend {trend_class}">{trend_icon} {abs(self.trend):.1f}%</span>' if self.trend else ''}
            </div>
        </div>
        """
        return html


class ChartComponent(DashboardComponent):
    """
    Chart Component
    
    Creates various chart types using Plotly
    """
    
    def __init__(self, component_id: str, title: str, chart_type: str = "line"):
        """
        Initialize chart component
        
        Args:
            component_id: Unique identifier
            title: Chart title
            chart_type: Type of chart ('line', 'bar', 'pie', 'scatter', 'heatmap')
        """
        super().__init__(component_id, title)
        self.chart_type = chart_type
        self.fig = None
    
    def create_chart(self, data: Dict[str, Any], **kwargs):
        """
        Create chart from data
        
        Args:
            data: Chart data (format depends on chart_type)
            **kwargs: Additional chart parameters
        """
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available. Chart creation skipped.")
            return
        
        if self.chart_type == 'line':
            self.fig = px.line(data, **kwargs)
        elif self.chart_type == 'bar':
            self.fig = px.bar(data, **kwargs)
        elif self.chart_type == 'pie':
            self.fig = px.pie(data, **kwargs)
        elif self.chart_type == 'scatter':
            self.fig = px.scatter(data, **kwargs)
        elif self.chart_type == 'heatmap':
            self.fig = px.imshow(data, **kwargs)
        
        if self.fig:
            self.fig.update_layout(title=self.title)
    
    def render(self) -> Dict[str, Any]:
        """Render chart component"""
        result = {
            'component_id': self.component_id,
            'type': 'chart',
            'chart_type': self.chart_type,
            'title': self.title
        }
        
        if self.fig:
            result['html'] = self.fig.to_html(include_plotlyjs='cdn', div_id=self.component_id)
            result['json'] = self.fig.to_json()
        
        return result


class TableComponent(DashboardComponent):
    """
    Table Component
    
    Displays data in tabular format
    """
    
    def __init__(self, component_id: str, title: str, columns: List[str], 
                 data: List[List[Any]], sortable: bool = True):
        """
        Initialize table component
        
        Args:
            component_id: Unique identifier
            title: Table title
            columns: Column names
            data: Table data (list of rows)
            sortable: Whether table is sortable
        """
        super().__init__(component_id, title)
        self.columns = columns
        self.data = data
        self.sortable = sortable
    
    def render(self) -> Dict[str, Any]:
        """Render table component"""
        return {
            'component_id': self.component_id,
            'type': 'table',
            'title': self.title,
            'columns': self.columns,
            'data': self.data,
            'sortable': self.sortable,
            'html': self._generate_html()
        }
    
    def _generate_html(self) -> str:
        """Generate HTML for table"""
        html = f'<div class="table-container" id="{self.component_id}">'
        html += f'<h3>{self.title}</h3>'
        html += '<table class="dashboard-table">'
        
        # Header
        html += '<thead><tr>'
        for col in self.columns:
            html += f'<th>{col}</th>'
        html += '</tr></thead>'
        
        # Body
        html += '<tbody>'
        for row in self.data:
            html += '<tr>'
            for cell in row:
                html += f'<td>{cell}</td>'
            html += '</tr>'
        html += '</tbody>'
        
        html += '</table></div>'
        return html


class DashboardLayout:
    """
    Dashboard Layout Manager
    
    Manages layout and arrangement of dashboard components
    """
    
    def __init__(self, layout_type: str = "grid"):
        """
        Initialize dashboard layout
        
        Args:
            layout_type: Layout type ('grid', 'rows', 'columns', 'custom')
        """
        self.layout_type = layout_type
        self.components: List[DashboardComponent] = []
        self.layout_config = {}
    
    def add_component(self, component: DashboardComponent, position: Optional[Dict[str, int]] = None):
        """
        Add component to dashboard
        
        Args:
            component: Dashboard component
            position: Optional position in layout
        """
        self.components.append(component)
        if position:
            self.layout_config[component.component_id] = position
    
    def render(self) -> Dict[str, Any]:
        """Render complete dashboard"""
        return {
            'layout_type': self.layout_type,
            'components': [comp.render() for comp in self.components],
            'config': self.layout_config
        }
    
    def generate_html(self) -> str:
        """Generate complete HTML dashboard"""
        html = '<div class="dashboard-container">'
        
        for component in self.components:
            rendered = component.render()
            if 'html' in rendered:
                html += rendered['html']
        
        html += '</div>'
        return html


def create_wellness_dashboard(metrics: Dict[str, Any]) -> DashboardLayout:
    """
    Create wellness dashboard layout
    
    Args:
        metrics: Dictionary of wellness metrics
        
    Returns:
        DashboardLayout instance
    """
    layout = DashboardLayout(layout_type="grid")
    
    # Add metric cards
    for metric_name, metric_data in metrics.items():
        card = MetricCard(
            component_id=f"metric_{metric_name}",
            title=metric_data.get('title', metric_name),
            value=metric_data.get('value', 0),
            trend=metric_data.get('trend'),
            unit=metric_data.get('unit', '')
        )
        layout.add_component(card)
    
    return layout


def get_dashboard_component(component_type: str, **kwargs) -> DashboardComponent:
    """
    Factory function to create dashboard components
    
    Args:
        component_type: Type of component ('metric', 'chart', 'table')
        **kwargs: Component-specific parameters
        
    Returns:
        DashboardComponent instance
    """
    if component_type == 'metric':
        return MetricCard(**kwargs)
    elif component_type == 'chart':
        return ChartComponent(**kwargs)
    elif component_type == 'table':
        return TableComponent(**kwargs)
    else:
        raise ValueError(f"Unknown component type: {component_type}")
