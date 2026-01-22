"""
Monitoring Dashboard
Web dashboard for ML Toolbox monitoring

Features:
- Real-time metrics visualization
- Drift detection alerts
- Performance monitoring
- Model comparison
- Alert management
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime, timedelta
import json
import asyncio
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available. Install with: pip install fastapi websockets")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")


class MonitoringDashboard:
    """
    Monitoring Dashboard for ML Toolbox
    
    Provides web interface for monitoring ML models
    """
    
    def __init__(
        self,
        model_monitors: Optional[Dict[str, Any]] = None,
        port: int = 8080
    ):
        """
        Args:
            model_monitors: Dictionary of {model_name: ModelMonitor}
            port: Dashboard port
        """
        self.model_monitors = model_monitors or {}
        self.port = port
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.alerts_history: List[Dict[str, Any]] = []
        
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(title="ML Toolbox Monitoring Dashboard")
            self._setup_routes()
        else:
            self.app = None
            warnings.warn("FastAPI not available. Dashboard not initialized.")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        if not FASTAPI_AVAILABLE:
            return
        
        @self.app.get("/")
        async def dashboard():
            """Main dashboard page"""
            html = self._generate_dashboard_html()
            return HTMLResponse(content=html)
        
        @self.app.get("/api/metrics")
        async def get_metrics(model_name: Optional[str] = None):
            """Get metrics for models"""
            if model_name:
                if model_name in self.model_monitors:
                    monitor = self.model_monitors[model_name]
                    summary = monitor.get_summary()
                    return summary
                else:
                    return {"error": f"Model {model_name} not found"}
            else:
                # Return metrics for all models
                all_metrics = {}
                for name, monitor in self.model_monitors.items():
                    all_metrics[name] = monitor.get_summary()
                return all_metrics
        
        @self.app.get("/api/alerts")
        async def get_alerts(severity: Optional[str] = None):
            """Get alerts"""
            alerts = []
            for name, monitor in self.model_monitors.items():
                model_alerts = monitor.get_alerts(severity=severity)
                for alert in model_alerts:
                    alert['model_name'] = name
                    alerts.append(alert)
            
            # Sort by timestamp
            alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return alerts[:100]  # Return latest 100
        
        @self.app.get("/api/drift")
        async def get_drift_status(model_name: Optional[str] = None):
            """Get drift status"""
            drift_status = {}
            for name, monitor in self.model_monitors.items():
                if model_name and name != model_name:
                    continue
                
                summary = monitor.get_summary()
                drift_status[name] = {
                    'has_data_drift': any(
                        a['type'] == 'data_drift' for a in monitor.get_alerts()
                    ),
                    'has_concept_drift': any(
                        a['type'] == 'concept_drift' for a in monitor.get_alerts()
                    ),
                    'performance_trend': summary.get('concept_drift_trend', {}).get('trend', 'unknown')
                }
            
            return drift_status
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await websocket.accept()
            try:
                while True:
                    # Send metrics update every 5 seconds
                    metrics = {}
                    for name, monitor in self.model_monitors.items():
                        summary = monitor.get_summary()
                        metrics[name] = {
                            'performance': summary.get('performance_summary', {}),
                            'alerts_count': summary.get('n_alerts', 0)
                        }
                    
                    await websocket.send_json(metrics)
                    await asyncio.sleep(5)
            except WebSocketDisconnect:
                pass
    
    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>ML Toolbox Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; background: #f9f9f9; }
        .alert { padding: 10px; margin: 5px 0; border-radius: 4px; }
        .alert-high { background: #fee; border-left: 4px solid #f00; }
        .alert-medium { background: #ffe; border-left: 4px solid #fa0; }
        .alert-low { background: #efe; border-left: 4px solid #0a0; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 0; }
    </style>
</head>
<body>
    <h1>ML Toolbox Monitoring Dashboard</h1>
    <div id="dashboard" class="dashboard"></div>
    
    <script>
        async function loadDashboard() {
            const response = await fetch('/api/metrics');
            const metrics = await response.json();
            
            const dashboard = document.getElementById('dashboard');
            dashboard.innerHTML = '';
            
            for (const [modelName, modelMetrics] of Object.entries(metrics)) {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = `
                    <h2>${modelName}</h2>
                    <div id="metrics-${modelName}"></div>
                    <div id="alerts-${modelName}"></div>
                `;
                dashboard.appendChild(card);
                
                // Load alerts
                loadAlerts(modelName);
            }
        }
        
        async function loadAlerts(modelName) {
            const response = await fetch('/api/alerts');
            const alerts = await response.json();
            
            const modelAlerts = alerts.filter(a => a.details?.model_name === modelName);
            const alertsDiv = document.getElementById(`alerts-${modelName}`);
            
            if (modelAlerts.length === 0) {
                alertsDiv.innerHTML = '<p>No alerts</p>';
            } else {
                alertsDiv.innerHTML = '<h3>Recent Alerts</h3>';
                modelAlerts.slice(0, 5).forEach(alert => {
                    const alertDiv = document.createElement('div');
                    alertDiv.className = `alert alert-${alert.severity || 'medium'}`;
                    alertDiv.innerHTML = `
                        <strong>${alert.type}</strong>: ${JSON.stringify(alert.details)}
                    `;
                    alertsDiv.appendChild(alertDiv);
                });
            }
        }
        
        // Load dashboard on page load
        loadDashboard();
        setInterval(loadDashboard, 30000); // Refresh every 30 seconds
    </script>
</body>
</html>
        """
    
    def add_model_monitor(self, model_name: str, monitor: Any):
        """Add a model monitor to the dashboard"""
        self.model_monitors[model_name] = monitor
    
    def run(self, host: str = "0.0.0.0", port: Optional[int] = None):
        """
        Run the dashboard server
        
        Args:
            host: Host address
            port: Port number (uses self.port if None)
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        import uvicorn
        uvicorn.run(self.app, host=host, port=port or self.port)


def create_dashboard_from_monitors(monitors: Dict[str, Any], port: int = 8080) -> MonitoringDashboard:
    """
    Create dashboard from model monitors
    
    Args:
        monitors: Dictionary of {model_name: ModelMonitor}
        port: Dashboard port
        
    Returns:
        MonitoringDashboard instance
    """
    dashboard = MonitoringDashboard(model_monitors=monitors, port=port)
    return dashboard
