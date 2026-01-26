"""
Unified ML Pipeline

Orchestrates Feature Pipeline → Training Pipeline → Inference Pipeline
"""
import numpy as np
from typing import Any, Dict, Optional, List, Union
import logging

from .base import BasePipeline
from .data_collection_pipeline import DataCollectionPipeline
from .feature_pipeline import FeaturePipeline
from .training_pipeline import TrainingPipeline
from .inference_pipeline import InferencePipeline
from .pipeline_state import PipelineState
from .feature_store import FeatureStore

logger = logging.getLogger(__name__)


class UnifiedMLPipeline:
    """
    Unified ML Pipeline
    
    Orchestrates:
    - Data Collection Pipeline (ETL: Extract → Transform → Load)
    - Feature Pipeline (Data → Features)
    - Training Pipeline (Features → Model)
    - Inference Pipeline (Features → Predictions)
    
    Provides:
    - Explicit pipeline stages
    - State management
    - Feature reuse
    - Pipeline versioning
    """
    
    def __init__(self, toolbox=None, enable_feature_store: bool = True,
                 enable_tuning: bool = False, enable_registry: bool = True,
                 enable_monitoring: bool = True, enable_data_collection: bool = False):
        """
        Initialize unified ML pipeline
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
        enable_feature_store : bool, default=True
            Whether to enable feature storage
        enable_tuning : bool, default=False
            Whether to enable hyperparameter tuning
        enable_registry : bool, default=True
            Whether to enable model registry
        enable_monitoring : bool, default=True
            Whether to enable monitoring
        """
        self.toolbox = toolbox
        
        # Initialize feature store
        self.feature_store = FeatureStore(enable_disk_storage=False) if enable_feature_store else None
        
        # Initialize feature pipeline first (needed for data collection pipeline)
        self.feature_pipeline = FeaturePipeline(
            toolbox=toolbox,
            feature_store=self.feature_store,
            enable_feature_store=enable_feature_store
        )
        
        # Initialize data collection pipeline (ETL)
        self.data_collection_pipeline = DataCollectionPipeline(
            toolbox=toolbox,
            feature_pipeline=self.feature_pipeline if enable_data_collection else None,
            enable_monitoring=enable_monitoring
        ) if enable_data_collection else None
        
        self.training_pipeline = TrainingPipeline(
            toolbox=toolbox,
            enable_tuning=enable_tuning,
            enable_registry=enable_registry
        )
        
        self.inference_pipeline = InferencePipeline(
            toolbox=toolbox,
            enable_monitoring=enable_monitoring
        )
        
        # Initialize pipeline state
        self.state = PipelineState(store_history=True)
        
        logger.info("[UnifiedMLPipeline] Initialized unified ML pipeline")
    
    def execute(self, X: Union[np.ndarray, Dict, List, str], y: Optional[np.ndarray] = None,
                mode: str = 'train', feature_name: str = "default",
                model_name: str = "default", reuse_features: bool = True,
                use_data_collection: bool = False, source_type: str = 'auto',
                nosql_client: Optional[Any] = None, nosql_collection: Optional[str] = None,
                nosql_query: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute unified pipeline
        
        Parameters
        ----------
        X : array-like, dict, list, or str
            Input data (can be raw user input or NoSQL query result)
        y : array-like, optional
            Target labels (required for training mode)
        mode : str, default='train'
            Pipeline mode: 'train' or 'inference'
        feature_name : str, default="default"
            Name for stored features
        model_name : str, default="default"
            Name for registered model
        reuse_features : bool, default=True
            Whether to reuse stored features in inference
        use_data_collection : bool, default=False
            Whether to use Data Collection Pipeline (ETL) first
        source_type : str, default='auto'
            Source type for data collection: 'user_input', 'nosql', 'auto'
        nosql_client : Any, optional
            NoSQL database client (if using NoSQL source)
        nosql_collection : str, optional
            NoSQL collection/table name (if using NoSQL source)
        nosql_query : dict, optional
            NoSQL query (if using NoSQL source)
        **kwargs
            Additional parameters for pipelines
            
        Returns
        -------
        result : dict
            Pipeline execution result
        """
        # Use Data Collection Pipeline if enabled and input is not already array
        if use_data_collection and self.data_collection_pipeline:
            if not isinstance(X, np.ndarray) or source_type != 'auto':
                logger.info("[UnifiedMLPipeline] Using Data Collection Pipeline (ETL)...")
                X = self.data_collection_pipeline.execute(
                    X,
                    source_type=source_type,
                    nosql_client=nosql_client,
                    nosql_collection=nosql_collection,
                    nosql_query=nosql_query,
                    feature_name=feature_name,
                    **kwargs
                )
        
        # Ensure X is numpy array
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        
        if mode == 'train':
            return self._execute_training(X, y, feature_name, model_name, **kwargs)
        elif mode == 'inference':
            return self._execute_inference(X, feature_name, model_name, reuse_features, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'inference'")
    
    def _execute_training(self, X: np.ndarray, y: np.ndarray, feature_name: str,
                        model_name: str, **kwargs) -> Dict[str, Any]:
        """Execute training pipeline"""
        logger.info("[UnifiedMLPipeline] Executing training pipeline...")
        
        # Stage 1: Feature Pipeline
        logger.info("[UnifiedMLPipeline] Stage 1: Feature Pipeline")
        X_features = self.feature_pipeline.execute(
            X,
            feature_name=feature_name,
            **kwargs
        )
        
        # Store features in pipeline state
        self.state.store_features(X_features, name=feature_name)
        
        # Stage 2: Training Pipeline
        logger.info("[UnifiedMLPipeline] Stage 2: Training Pipeline")
        training_result = self.training_pipeline.execute(
            X_features,
            y,
            model_name=model_name,
            **kwargs
        )
        
        # Store model in pipeline state
        model = training_result.get('model')
        if model:
            self.state.store_model(model, name=model_name, metadata=training_result.get('metrics', {}))
        
        # Add to history
        self.state.add_to_history('training_completed', {
            'feature_name': feature_name,
            'model_name': model_name,
            'metrics': training_result.get('metrics', {})
        })
        
        result = {
            'mode': 'train',
            'features': X_features,
            'model': model,
            'model_id': training_result.get('model_id'),
            'metrics': training_result.get('metrics', {}),
            'feature_name': feature_name,
            'model_name': model_name
        }
        
        logger.info("[UnifiedMLPipeline] Training pipeline completed")
        return result
    
    def _execute_inference(self, X: np.ndarray, feature_name: str, model_name: str,
                          reuse_features: bool, **kwargs) -> Dict[str, Any]:
        """Execute inference pipeline"""
        logger.info("[UnifiedMLPipeline] Executing inference pipeline...")
        
        # Get or compute features
        if reuse_features and self.feature_store:
            # Try to reuse stored features
            X_features = self.feature_store.get(feature_name)
            if X_features is None:
                logger.info("[UnifiedMLPipeline] Features not found in store, computing...")
                X_features = self.feature_pipeline.execute(X, feature_name=feature_name, **kwargs)
        else:
            # Compute features
            X_features = self.feature_pipeline.execute(X, feature_name=feature_name, **kwargs)
        
        # Get model
        model = self.state.get_model(name=model_name)
        if model is None:
            # Try to get from toolbox registry
            if self.toolbox and hasattr(self.toolbox, 'model_registry') and self.toolbox.model_registry:
                try:
                    model, _ = self.toolbox.get_registered_model(f"{model_name}:latest")
                except:
                    model = None
        
        if model is None:
            raise ValueError(f"Model '{model_name}' not found. Train a model first or provide model.")
        
        # Stage 3: Inference Pipeline
        logger.info("[UnifiedMLPipeline] Stage 3: Inference Pipeline")
        inference_result = self.inference_pipeline.execute(X_features, model, **kwargs)
        
        # Store predictions in pipeline state
        predictions = inference_result.get('predictions')
        if predictions is not None:
            self.state.store_predictions(predictions, name=f"{model_name}_predictions")
        
        # Add to history
        self.state.add_to_history('inference_completed', {
            'feature_name': feature_name,
            'model_name': model_name,
            'prediction_count': len(predictions) if predictions is not None else 0
        })
        
        result = {
            'mode': 'inference',
            'features': X_features,
            'predictions': predictions,
            'monitoring_metrics': inference_result.get('monitoring_metrics', {}),
            'feature_name': feature_name,
            'model_name': model_name
        }
        
        logger.info("[UnifiedMLPipeline] Inference pipeline completed")
        return result
    
    def get_features(self, feature_name: str = "default", version: Optional[str] = None) -> Optional[np.ndarray]:
        """Get stored features"""
        if self.feature_store:
            return self.feature_store.get(feature_name, version)
        return self.state.get_features(feature_name)
    
    def get_model(self, model_name: str = "default", version: Optional[int] = None) -> Optional[Any]:
        """Get stored model"""
        return self.state.get_model(model_name, version)
    
    def get_predictions(self, name: str = "default", version: Optional[int] = None) -> Optional[Any]:
        """Get stored predictions"""
        return self.state.get_predictions(name, version)
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        status = {
            'feature_pipeline': self.feature_pipeline.get_status(),
            'training_pipeline': self.training_pipeline.get_status(),
            'inference_pipeline': self.inference_pipeline.get_status(),
            'state': self.state.get_state_summary(),
            'feature_store': self.feature_store.get_summary() if self.feature_store else None
        }
        
        if self.data_collection_pipeline:
            status['data_collection_pipeline'] = self.data_collection_pipeline.get_status()
        
        return status
    
    def reset(self):
        """Reset pipeline state"""
        self.state.clear()
        self.feature_pipeline.reset()
        self.training_pipeline.reset()
        self.inference_pipeline.reset()
        if self.feature_store:
            self.feature_store.clear()
        logger.info("[UnifiedMLPipeline] Pipeline state reset")
