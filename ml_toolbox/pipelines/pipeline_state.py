"""
Pipeline State Management

Manages state across pipeline stages for reproducibility and debugging.
"""
from typing import Any, Dict, Optional, List
import logging
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)


class PipelineState:
    """
    Manages pipeline state across stages
    
    Stores:
    - Features (from feature pipeline)
    - Models (from training pipeline)
    - Predictions (from inference pipeline)
    - Metadata (versions, timestamps, etc.)
    """
    
    def __init__(self, store_history: bool = True):
        """
        Initialize pipeline state
        
        Parameters
        ----------
        store_history : bool, default=True
            Whether to store execution history
        """
        self.store_history = store_history
        self.features: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.predictions: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.version = 1
    
    def store_features(self, features: Any, name: str = "default", metadata: Optional[Dict] = None):
        """
        Store features from feature pipeline
        
        Parameters
        ----------
        features : Any
            Features to store
        name : str, default="default"
            Feature set name
        metadata : dict, optional
            Additional metadata
        """
        feature_id = f"{name}_v{self.version}"
        self.features[feature_id] = {
            'features': features,
            'name': name,
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        logger.info(f"[PipelineState] Stored features: {feature_id}")
    
    def get_features(self, name: str = "default", version: Optional[int] = None) -> Optional[Any]:
        """
        Get stored features
        
        Parameters
        ----------
        name : str, default="default"
            Feature set name
        version : int, optional
            Specific version (default: latest)
            
        Returns
        -------
        features : Any
            Stored features or None
        """
        if version:
            feature_id = f"{name}_v{version}"
        else:
            # Get latest version
            matching = [k for k in self.features.keys() if k.startswith(f"{name}_v")]
            if not matching:
                return None
            feature_id = max(matching, key=lambda k: self.features[k]['version'])
        
        if feature_id in self.features:
            logger.info(f"[PipelineState] Retrieved features: {feature_id}")
            return self.features[feature_id]['features']
        return None
    
    def store_model(self, model: Any, name: str = "default", metadata: Optional[Dict] = None):
        """
        Store model from training pipeline
        
        Parameters
        ----------
        model : Any
            Model to store
        name : str, default="default"
            Model name
        metadata : dict, optional
            Additional metadata
        """
        model_id = f"{name}_v{self.version}"
        self.models[model_id] = {
            'model': model,
            'name': name,
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        logger.info(f"[PipelineState] Stored model: {model_id}")
    
    def get_model(self, name: str = "default", version: Optional[int] = None) -> Optional[Any]:
        """
        Get stored model
        
        Parameters
        ----------
        name : str, default="default"
            Model name
        version : int, optional
            Specific version (default: latest)
            
        Returns
        -------
        model : Any
            Stored model or None
        """
        if version:
            model_id = f"{name}_v{version}"
        else:
            # Get latest version
            matching = [k for k in self.models.keys() if k.startswith(f"{name}_v")]
            if not matching:
                return None
            model_id = max(matching, key=lambda k: self.models[k]['version'])
        
        if model_id in self.models:
            logger.info(f"[PipelineState] Retrieved model: {model_id}")
            return self.models[model_id]['model']
        return None
    
    def store_predictions(self, predictions: Any, name: str = "default", metadata: Optional[Dict] = None):
        """
        Store predictions from inference pipeline
        
        Parameters
        ----------
        predictions : Any
            Predictions to store
        name : str, default="default"
            Prediction set name
        metadata : dict, optional
            Additional metadata
        """
        pred_id = f"{name}_v{self.version}"
        self.predictions[pred_id] = {
            'predictions': predictions,
            'name': name,
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        logger.info(f"[PipelineState] Stored predictions: {pred_id}")
    
    def add_to_history(self, event: str, data: Optional[Dict] = None):
        """Add event to history"""
        if self.store_history:
            self.history.append({
                'event': event,
                'timestamp': datetime.now().isoformat(),
                'data': data or {},
                'version': self.version
            })
    
    def increment_version(self):
        """Increment pipeline version"""
        self.version += 1
        logger.info(f"[PipelineState] Version incremented to {self.version}")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get state summary"""
        return {
            'version': self.version,
            'features_count': len(self.features),
            'models_count': len(self.models),
            'predictions_count': len(self.predictions),
            'history_count': len(self.history),
            'features': list(self.features.keys()),
            'models': list(self.models.keys()),
            'predictions': list(self.predictions.keys())
        }
    
    def clear(self):
        """Clear all state"""
        self.features = {}
        self.models = {}
        self.predictions = {}
        self.metadata = {}
        self.history = []
        self.version = 1
        logger.info("[PipelineState] State cleared")
