"""
Unified ML Pipelines Module

Provides explicit pipeline stages for:
- Feature Pipeline: Data Ingestion → Preprocessing → Feature Engineering → Feature Selection → Feature Store
- Training Pipeline: Model Training → Hyperparameter Tuning → Model Evaluation → Model Validation → Model Registry
- Inference Pipeline: Model Serving → Batch Inference → Real-time Inference → A/B Testing → Monitoring

And a UnifiedMLPipeline that orchestrates all three.

Enhanced features:
- Pipeline Monitoring: Detailed metrics tracking
- Pipeline Persistence: Save/load pipeline state
- Pipeline Retry: Automatic retry with error recovery
- Pipeline Debugger: Debugging and visualization tools
"""
from .base import PipelineStage, BasePipeline
from .data_collection_pipeline import DataCollectionPipeline
from .feature_pipeline import FeaturePipeline
from .training_pipeline import TrainingPipeline
from .inference_pipeline import InferencePipeline
from .unified_pipeline import UnifiedMLPipeline
from .pipeline_state import PipelineState
from .feature_store import FeatureStore

# Enhanced features
try:
    from .pipeline_monitoring import PipelineMonitor, PipelineMetrics
    from .pipeline_persistence import PipelinePersistence
    from .pipeline_retry import RetryHandler, RetryConfig, RetryStrategy
    from .pipeline_debugger import PipelineDebugger
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    PipelineMonitor = None
    PipelineMetrics = None
    PipelinePersistence = None
    RetryHandler = None
    RetryConfig = None
    RetryStrategy = None
    PipelineDebugger = None

__all__ = [
    'PipelineStage',
    'BasePipeline',
    'DataCollectionPipeline',
    'FeaturePipeline',
    'TrainingPipeline',
    'InferencePipeline',
    'UnifiedMLPipeline',
    'PipelineState',
    'FeatureStore'
]

if ENHANCED_FEATURES_AVAILABLE:
    __all__.extend([
        'PipelineMonitor',
        'PipelineMetrics',
        'PipelinePersistence',
        'RetryHandler',
        'RetryConfig',
        'RetryStrategy',
        'PipelineDebugger'
    ])
