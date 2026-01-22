"""
Model Deployment & Serving
Burkov Machine Learning Engineering - Production Deployment

Features:
- REST API for model serving
- Batch inference
- Real-time inference
- Model versioning
- Canary deployments
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import json
import pickle
from datetime import datetime
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import FastAPI
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from typing import List as TypingList
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available. Install with: pip install fastapi uvicorn")
    print("Model serving API will not be available")


class ModelVersion:
    """Model version information"""
    
    def __init__(
        self,
        model: Any,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            model: Trained model
            version: Version string (e.g., 'v1.0.0')
            metadata: Additional metadata
        """
        self.model = model
        self.version = version
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.is_active = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version': self.version,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active
        }


class ModelRegistry:
    """Model version registry"""
    
    def __init__(self):
        self.models: Dict[str, ModelVersion] = {}
        self.active_version: Optional[str] = None
    
    def register(
        self,
        model: Any,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        set_active: bool = False
    ) -> ModelVersion:
        """
        Register a new model version
        
        Args:
            model: Trained model
            version: Version string
            metadata: Additional metadata
            set_active: Whether to set as active version
            
        Returns:
            ModelVersion object
        """
        model_version = ModelVersion(model, version, metadata)
        self.models[version] = model_version
        
        if set_active or self.active_version is None:
            self.set_active_version(version)
        
        return model_version
    
    def set_active_version(self, version: str):
        """Set active model version"""
        if version not in self.models:
            raise ValueError(f"Version {version} not found")
        
        # Deactivate current active version
        if self.active_version:
            self.models[self.active_version].is_active = False
        
        # Activate new version
        self.models[version].is_active = True
        self.active_version = version
    
    def get_model(self, version: Optional[str] = None) -> Any:
        """
        Get model by version
        
        Args:
            version: Version string (None for active version)
            
        Returns:
            Model object
        """
        if version is None:
            version = self.active_version
        
        if version is None:
            raise ValueError("No active model version")
        
        if version not in self.models:
            raise ValueError(f"Version {version} not found")
        
        return self.models[version].model
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all model versions"""
        return [mv.to_dict() for mv in self.models.values()]


class ModelServer:
    """
    Model serving server
    
    Provides REST API for model inference
    """
    
    def __init__(self, model_registry: ModelRegistry, model_name: str = 'default'):
        """
        Args:
            model_registry: Model registry
            model_name: Name of the model
        """
        self.registry = model_registry
        self.model_name = model_name
        
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(title=f"{model_name} Model Server")
            self._setup_routes()
        else:
            self.app = None
            warnings.warn("FastAPI not available. API routes not set up.")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        if not FASTAPI_AVAILABLE:
            return
        
        # Request models
        class PredictionRequest(BaseModel):
            data: TypingList[TypingList[float]]
            version: Optional[str] = None
        
        class BatchPredictionRequest(BaseModel):
            data: TypingList[TypingList[float]]
            version: Optional[str] = None
        
        @self.app.get("/")
        async def root():
            return {
                "model_name": self.model_name,
                "status": "running",
                "active_version": self.registry.active_version
            }
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        @self.app.get("/versions")
        async def list_versions():
            return {"versions": self.registry.list_versions()}
        
        @self.app.post("/predict")
        async def predict(request: PredictionRequest):
            try:
                model = self.registry.get_model(request.version)
                X = np.array(request.data)
                
                if hasattr(model, 'predict'):
                    predictions = model.predict(X)
                else:
                    raise HTTPException(status_code=500, detail="Model does not support predict")
                
                return {
                    "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                    "version": request.version or self.registry.active_version
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict_proba")
        async def predict_proba(request: PredictionRequest):
            try:
                model = self.registry.get_model(request.version)
                X = np.array(request.data)
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
                    return {
                        "probabilities": probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities,
                        "version": request.version or self.registry.active_version
                    }
                else:
                    raise HTTPException(status_code=500, detail="Model does not support predict_proba")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/batch_predict")
        async def batch_predict(request: BatchPredictionRequest):
            try:
                model = self.registry.get_model(request.version)
                X = np.array(request.data)
                
                if hasattr(model, 'predict'):
                    predictions = model.predict(X)
                    return {
                        "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                        "n_samples": len(predictions),
                        "version": request.version or self.registry.active_version
                    }
                else:
                    raise HTTPException(status_code=500, detail="Model does not support predict")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the model server
        
        Args:
            host: Host address
            port: Port number
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


class BatchInference:
    """Batch inference processor"""
    
    def __init__(self, model: Any):
        """
        Args:
            model: Trained model
        """
        self.model = model
    
    def predict_batch(
        self,
        X: np.ndarray,
        batch_size: int = 1000,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Process batch predictions
        
        Args:
            X: Input data
            batch_size: Batch size
            verbose: Whether to print progress
            
        Returns:
            Predictions
        """
        X = np.asarray(X)
        n_samples = len(X)
        predictions = []
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            
            if hasattr(self.model, 'predict'):
                batch_predictions = self.model.predict(batch)
                predictions.append(batch_predictions)
            else:
                raise ValueError("Model does not support predict")
            
            if verbose:
                print(f"Processed {min(i + batch_size, n_samples)}/{n_samples} samples")
        
        return np.concatenate(predictions) if len(predictions) > 0 else np.array([])


class RealTimeInference:
    """Real-time inference processor"""
    
    def __init__(self, model: Any):
        """
        Args:
            model: Trained model
        """
        self.model = model
    
    def predict(self, x: np.ndarray) -> Union[np.ndarray, float, int]:
        """
        Single prediction (real-time)
        
        Args:
            x: Single sample (1D array)
            
        Returns:
            Prediction
        """
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(x)
            return prediction[0] if len(prediction) == 1 else prediction
        else:
            raise ValueError("Model does not support predict")
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Single prediction with probabilities (real-time)
        
        Args:
            x: Single sample (1D array)
            
        Returns:
            Probabilities
        """
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(x)
            return probabilities[0] if probabilities.shape[0] == 1 else probabilities
        else:
            raise ValueError("Model does not support predict_proba")


class CanaryDeployment:
    """
    Canary deployment manager
    
    Gradually roll out new model versions to a subset of traffic
    """
    
    def __init__(self, model_registry: ModelRegistry):
        """
        Args:
            model_registry: Model registry
        """
        self.registry = model_registry
        self.canary_version: Optional[str] = None
        self.canary_percentage: float = 0.0  # 0.0 to 1.0
    
    def start_canary(
        self,
        version: str,
        percentage: float = 0.1
    ):
        """
        Start canary deployment
        
        Args:
            version: New model version
            percentage: Percentage of traffic to route to canary (0.0 to 1.0)
        """
        if version not in self.registry.models:
            raise ValueError(f"Version {version} not found")
        
        self.canary_version = version
        self.canary_percentage = max(0.0, min(1.0, percentage))
    
    def get_model_for_request(self, request_id: Optional[str] = None) -> Tuple[Any, str]:
        """
        Get model for a request (canary or production)
        
        Args:
            request_id: Optional request ID for consistent routing
            
        Returns:
            Tuple of (model, version)
        """
        if self.canary_version and np.random.random() < self.canary_percentage:
            model = self.registry.get_model(self.canary_version)
            return model, self.canary_version
        else:
            model = self.registry.get_model()  # Active version
            return model, self.registry.active_version
    
    def promote_canary(self):
        """Promote canary to production"""
        if self.canary_version:
            self.registry.set_active_version(self.canary_version)
            self.canary_version = None
            self.canary_percentage = 0.0
    
    def rollback_canary(self):
        """Rollback canary deployment"""
        self.canary_version = None
        self.canary_percentage = 0.0
