"""
Compartment 4: MLOps & Production
Burkov Machine Learning Engineering - Production ML

Components:
- Model Monitoring (drift detection, performance monitoring)
- Model Deployment (serving, APIs, versioning)
- A/B Testing (statistical testing, traffic splitting)
- Experiment Tracking (logging, parameter tracking)
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

# Model Monitoring
try:
    from model_monitoring import (
        DataDriftDetector,
        ConceptDriftDetector,
        PerformanceMonitor,
        ModelMonitor
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    MONITORING_AVAILABLE = False
    print(f"Warning: Could not import model monitoring: {e}")

# Model Deployment
try:
    from model_deployment import (
        ModelVersion,
        ModelRegistry,
        ModelServer,
        BatchInference,
        RealTimeInference,
        CanaryDeployment
    )
    DEPLOYMENT_AVAILABLE = True
except ImportError as e:
    DEPLOYMENT_AVAILABLE = False
    print(f"Warning: Could not import model deployment: {e}")

# A/B Testing
try:
    from ab_testing import ABTest, MultiVariantTest
    AB_TESTING_AVAILABLE = True
except ImportError as e:
    AB_TESTING_AVAILABLE = False
    print(f"Warning: Could not import A/B testing: {e}")

# Experiment Tracking
try:
    from experiment_tracking import Experiment, ExperimentTracker
    EXPERIMENT_TRACKING_AVAILABLE = True
except ImportError as e:
    EXPERIMENT_TRACKING_AVAILABLE = False
    print(f"Warning: Could not import experiment tracking: {e}")


class MLOpsCompartment:
    """
    Compartment 4: MLOps & Production
    
    Burkov Machine Learning Engineering methods for production ML:
    - Model Monitoring & Drift Detection
    - Model Deployment & Serving
    - A/B Testing
    - Experiment Tracking
    """
    
    def __init__(self):
        self.components = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all MLOps components"""
        
        # Model Monitoring
        if MONITORING_AVAILABLE:
            self.components['DataDriftDetector'] = DataDriftDetector
            self.components['ConceptDriftDetector'] = ConceptDriftDetector
            self.components['PerformanceMonitor'] = PerformanceMonitor
            self.components['ModelMonitor'] = ModelMonitor
        
        # Model Deployment
        if DEPLOYMENT_AVAILABLE:
            self.components['ModelRegistry'] = ModelRegistry
            self.components['ModelServer'] = ModelServer
            self.components['BatchInference'] = BatchInference
            self.components['RealTimeInference'] = RealTimeInference
            self.components['CanaryDeployment'] = CanaryDeployment
        
        # A/B Testing
        if AB_TESTING_AVAILABLE:
            self.components['ABTest'] = ABTest
            self.components['MultiVariantTest'] = MultiVariantTest
        
        # Experiment Tracking
        if EXPERIMENT_TRACKING_AVAILABLE:
            self.components['ExperimentTracker'] = ExperimentTracker
            self.components['Experiment'] = Experiment
        
        # Phase 3: Feature Store
        try:
            from feature_store import FeatureStore
            self.components['FeatureStore'] = FeatureStore
        except ImportError as e:
            print(f"Warning: Could not import feature store: {e}")
        
        # Phase 3: Monitoring Dashboard
        try:
            from monitoring_dashboard import MonitoringDashboard, create_dashboard_from_monitors
            self.components['MonitoringDashboard'] = MonitoringDashboard
            self.components['create_dashboard_from_monitors'] = create_dashboard_from_monitors
        except ImportError as e:
            print(f"Warning: Could not import monitoring dashboard: {e}")
        
        # Phase 3: Model Compression
        try:
            from model_compression import ModelCompressor
            self.components['ModelCompressor'] = ModelCompressor
        except ImportError as e:
            print(f"Warning: Could not import model compressor: {e}")
        
        # Component descriptions
        self.component_descriptions = {
            'DataDriftDetector': {
                'description': 'Detect data drift (feature distribution changes)',
                'features': [
                    'Kolmogorov-Smirnov test',
                    'Population Stability Index (PSI)',
                    'Z-score based detection',
                    'Feature-level drift detection'
                ],
                'location': 'model_monitoring.py',
                'category': 'Monitoring',
                'dependencies': ['scipy>=1.11.0', 'numpy>=1.26.0']
            },
            'ConceptDriftDetector': {
                'description': 'Detect concept drift (target relationship changes)',
                'features': [
                    'Performance degradation detection',
                    'Trend analysis',
                    'Threshold-based alerts',
                    'Performance history tracking'
                ],
                'location': 'model_monitoring.py',
                'category': 'Monitoring',
                'dependencies': ['numpy>=1.26.0']
            },
            'PerformanceMonitor': {
                'description': 'Monitor model performance metrics',
                'features': [
                    'Accuracy, precision, recall, F1 tracking',
                    'Latency monitoring',
                    'Throughput calculation',
                    'Metric history'
                ],
                'location': 'model_monitoring.py',
                'category': 'Monitoring',
                'dependencies': ['scikit-learn>=1.5.0', 'numpy>=1.26.0']
            },
            'ModelMonitor': {
                'description': 'Comprehensive model monitoring system',
                'features': [
                    'Combines data drift, concept drift, performance monitoring',
                    'Alert system',
                    'Comprehensive monitoring summary',
                    'Production-ready monitoring'
                ],
                'location': 'model_monitoring.py',
                'category': 'Monitoring',
                'dependencies': ['scipy>=1.11.0', 'scikit-learn>=1.5.0', 'numpy>=1.26.0']
            },
            'ModelRegistry': {
                'description': 'Model version registry',
                'features': [
                    'Model versioning',
                    'Version management',
                    'Active version tracking',
                    'Metadata storage'
                ],
                'location': 'model_deployment.py',
                'category': 'Deployment',
                'dependencies': []
            },
            'ModelServer': {
                'description': 'REST API for model serving',
                'features': [
                    'FastAPI-based REST API',
                    'Real-time inference',
                    'Batch inference',
                    'Model versioning support',
                    'Health checks'
                ],
                'location': 'model_deployment.py',
                'category': 'Deployment',
                'dependencies': ['fastapi>=0.100.0', 'uvicorn>=0.23.0']
            },
            'BatchInference': {
                'description': 'Batch inference processor',
                'features': [
                    'Batch prediction processing',
                    'Configurable batch size',
                    'Progress tracking',
                    'Memory efficient'
                ],
                'location': 'model_deployment.py',
                'category': 'Deployment',
                'dependencies': ['numpy>=1.26.0']
            },
            'RealTimeInference': {
                'description': 'Real-time inference processor',
                'features': [
                    'Single prediction processing',
                    'Low latency',
                    'Probability support',
                    'Real-time serving'
                ],
                'location': 'model_deployment.py',
                'category': 'Deployment',
                'dependencies': ['numpy>=1.26.0']
            },
            'CanaryDeployment': {
                'description': 'Canary deployment manager',
                'features': [
                    'Gradual rollout',
                    'Traffic splitting',
                    'Canary promotion',
                    'Rollback support'
                ],
                'location': 'model_deployment.py',
                'category': 'Deployment',
                'dependencies': ['numpy>=1.26.0']
            },
            'ABTest': {
                'description': 'A/B Testing framework for ML models',
                'features': [
                    'Statistical A/B testing',
                    'Traffic splitting',
                    'Metric collection',
                    'Significance testing',
                    'Variant comparison'
                ],
                'location': 'ab_testing.py',
                'category': 'Testing',
                'dependencies': ['scipy>=1.11.0', 'numpy>=1.26.0']
            },
            'MultiVariantTest': {
                'description': 'Multi-variant testing (A/B/C/...)',
                'features': [
                    'Multiple variant support',
                    'Pairwise comparisons',
                    'Best variant identification',
                    'Traffic splitting'
                ],
                'location': 'ab_testing.py',
                'category': 'Testing',
                'dependencies': ['scipy>=1.11.0', 'numpy>=1.26.0']
            },
            'ExperimentTracker': {
                'description': 'Experiment tracking system',
                'features': [
                    'Experiment logging',
                    'Parameter tracking',
                    'Metric tracking',
                    'Model artifacts storage',
                    'Experiment comparison',
                    'Best experiment identification'
                ],
                'location': 'experiment_tracking.py',
                'category': 'Tracking',
                'dependencies': []
            },
            'Experiment': {
                'description': 'Single experiment record',
                'features': [
                    'Parameter logging',
                    'Metric logging',
                    'Model artifact storage',
                    'Tags and notes',
                    'Status tracking'
                ],
                'location': 'experiment_tracking.py',
                'category': 'Tracking',
                'dependencies': []
            },
            'FeatureStore': {
                'description': 'Feature store for ML models',
                'features': [
                    'Feature storage and retrieval',
                    'Feature versioning',
                    'Feature lineage',
                    'Online/offline serving',
                    'Feature discovery'
                ],
                'location': 'feature_store.py',
                'category': 'Data Management',
                'dependencies': ['numpy>=1.26.0', 'pandas>=2.0.0']
            },
            'MonitoringDashboard': {
                'description': 'Web dashboard for monitoring',
                'features': [
                    'Real-time metrics visualization',
                    'Drift detection alerts',
                    'Performance monitoring',
                    'Model comparison',
                    'Alert management'
                ],
                'location': 'monitoring_dashboard.py',
                'category': 'Monitoring',
                'dependencies': ['fastapi>=0.100.0', 'plotly>=5.0.0']
            },
            'ModelCompressor': {
                'description': 'Model compression for efficient deployment',
                'features': [
                    'Quantization (reduce precision)',
                    'Pruning (remove unnecessary weights)',
                    'Size optimization',
                    'Model size estimation'
                ],
                'location': 'model_compression.py',
                'category': 'Optimization',
                'dependencies': ['torch>=2.3.0']
            }
        }
    
    def get_data_drift_detector(self, reference_data, alpha: float = 0.05):
        """Get data drift detector instance"""
        if 'DataDriftDetector' in self.components:
            return self.components['DataDriftDetector'](reference_data, alpha)
        else:
            raise ImportError("DataDriftDetector not available")
    
    def get_concept_drift_detector(self, baseline_performance: float, threshold: float = 0.1):
        """Get concept drift detector instance"""
        if 'ConceptDriftDetector' in self.components:
            return self.components['ConceptDriftDetector'](baseline_performance, threshold)
        else:
            raise ImportError("ConceptDriftDetector not available")
    
    def get_performance_monitor(self, model_name: str = 'default'):
        """Get performance monitor instance"""
        if 'PerformanceMonitor' in self.components:
            return self.components['PerformanceMonitor'](model_name)
        else:
            raise ImportError("PerformanceMonitor not available")
    
    def get_model_monitor(
        self,
        model,
        reference_data,
        reference_labels=None,
        baseline_performance=None,
        model_name='default'
    ):
        """Get comprehensive model monitor instance"""
        if 'ModelMonitor' in self.components:
            return self.components['ModelMonitor'](
                model, reference_data, reference_labels, baseline_performance, model_name
            )
        else:
            raise ImportError("ModelMonitor not available")
    
    def get_model_registry(self):
        """Get model registry instance"""
        if 'ModelRegistry' in self.components:
            return self.components['ModelRegistry']()
        else:
            raise ImportError("ModelRegistry not available")
    
    def get_model_server(self, model_registry, model_name: str = 'default'):
        """Get model server instance"""
        if 'ModelServer' in self.components:
            return self.components['ModelServer'](model_registry, model_name)
        else:
            raise ImportError("ModelServer not available")
    
    def get_batch_inference(self, model):
        """Get batch inference processor"""
        if 'BatchInference' in self.components:
            return self.components['BatchInference'](model)
        else:
            raise ImportError("BatchInference not available")
    
    def get_realtime_inference(self, model):
        """Get real-time inference processor"""
        if 'RealTimeInference' in self.components:
            return self.components['RealTimeInference'](model)
        else:
            raise ImportError("RealTimeInference not available")
    
    def get_canary_deployment(self, model_registry):
        """Get canary deployment manager"""
        if 'CanaryDeployment' in self.components:
            return self.components['CanaryDeployment'](model_registry)
        else:
            raise ImportError("CanaryDeployment not available")
    
    def get_ab_test(self, test_name: str, variants: Dict[str, Any], traffic_split=None):
        """Get A/B test instance"""
        if 'ABTest' in self.components:
            return self.components['ABTest'](test_name, variants, traffic_split)
        else:
            raise ImportError("ABTest not available")
    
    def get_multivariant_test(self, test_name: str, variants: Dict[str, Any], traffic_split=None):
        """Get multi-variant test instance"""
        if 'MultiVariantTest' in self.components:
            return self.components['MultiVariantTest'](test_name, variants, traffic_split)
        else:
            raise ImportError("MultiVariantTest not available")
    
    def get_experiment_tracker(self, storage_dir: str = "experiments"):
        """Get experiment tracker instance"""
        if 'ExperimentTracker' in self.components:
            return self.components['ExperimentTracker'](storage_dir)
        else:
            raise ImportError("ExperimentTracker not available")
    
    def get_feature_store(self, storage_dir: str = "feature_store", backend: str = 'pickle'):
        """Get feature store instance (Phase 3)"""
        if 'FeatureStore' in self.components:
            return self.components['FeatureStore'](storage_dir, backend)
        else:
            raise ImportError("FeatureStore not available")
    
    def get_monitoring_dashboard(self, model_monitors=None, port: int = 8080):
        """Get monitoring dashboard instance (Phase 3)"""
        if 'MonitoringDashboard' in self.components:
            return self.components['MonitoringDashboard'](model_monitors, port)
        else:
            raise ImportError("MonitoringDashboard not available")
    
    def get_model_compressor(self):
        """Get model compressor instance (Phase 3)"""
        if 'ModelCompressor' in self.components:
            return self.components['ModelCompressor']()
        else:
            raise ImportError("ModelCompressor not available")
    
    def list_components(self) -> List[str]:
        """List all available components"""
        return list(self.components.keys())
    
    def get_component_info(self, component_name: str) -> Dict[str, Any]:
        """Get information about a component"""
        if component_name in self.component_descriptions:
            return self.component_descriptions[component_name]
        else:
            return {'error': f'Component {component_name} not found'}
