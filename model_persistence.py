"""
Model Persistence & Loading
Standardized model serialization for ML Toolbox

Features:
- Standardized save/load
- Model metadata storage
- Version tracking
- Model validation on load
- Cross-platform compatibility
"""
import pickle
import json
import joblib
from pathlib import Path
from typing import Any, Optional, Dict, Union, List
from datetime import datetime
import warnings

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class ModelPersistence:
    """
    Standardized model persistence
    
    Supports multiple formats: pickle, joblib, JSON metadata
    """
    
    def __init__(
        self,
        storage_dir: Union[str, Path] = "models",
        format: str = 'pickle',
        compress: bool = False,
        include_metadata: bool = True
    ):
        """
        Args:
            storage_dir: Directory to store models
            format: Storage format ('pickle', 'joblib')
            compress: Whether to compress models
            include_metadata: Whether to include metadata
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.compress = compress
        self.include_metadata = include_metadata
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save model with metadata
        
        Args:
            model: Model to save
            model_name: Name of the model
            version: Model version (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Dictionary with save information
        """
        # Create model directory
        if version:
            model_dir = self.storage_dir / model_name / version
        else:
            model_dir = self.storage_dir / model_name
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.format == 'pickle':
            model_path = model_dir / 'model.pkl'
            with open(model_path, 'wb') as f:
                if self.compress:
                    import gzip
                    with gzip.open(model_path, 'wb') as gz:
                        pickle.dump(model, gz)
                else:
                    pickle.dump(model, f)
        elif self.format == 'joblib':
            model_path = model_dir / 'model.joblib'
            joblib.dump(model, model_path, compress=self.compress)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        
        # Save metadata
        if self.include_metadata:
            metadata_dict = {
                'model_name': model_name,
                'version': version or 'latest',
                'format': self.format,
                'compressed': self.compress,
                'saved_at': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'metadata': metadata or {}
            }
            
            metadata_path = model_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        
        return {
            'model_path': str(model_path),
            'metadata_path': str(model_dir / 'metadata.json') if self.include_metadata else None,
            'model_name': model_name,
            'version': version or 'latest',
            'format': self.format
        }
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Load model with metadata
        
        Args:
            model_name: Name of the model
            version: Model version (optional, defaults to 'latest')
            validate: Whether to validate model on load
            
        Returns:
            Dictionary with model and metadata
        """
        # Find model directory
        if version:
            model_dir = self.storage_dir / model_name / version
        else:
            # Find latest version
            model_dir = self.storage_dir / model_name
            if model_dir.exists():
                versions = [d for d in model_dir.iterdir() if d.is_dir()]
                if versions:
                    # Sort by modification time
                    model_dir = max(versions, key=lambda p: p.stat().st_mtime)
                else:
                    # No version subdirectories, use model_dir directly
                    pass
            else:
                raise FileNotFoundError(f"Model {model_name} not found")
        
        # Load model
        if self.format == 'pickle':
            model_path = model_dir / 'model.pkl'
            if not model_path.exists():
                # Try compressed
                model_path = model_dir / 'model.pkl.gz'
                if model_path.exists():
                    import gzip
                    with gzip.open(model_path, 'rb') as f:
                        model = pickle.load(f)
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
        elif self.format == 'joblib':
            model_path = model_dir / 'model.joblib'
            model = joblib.load(model_path)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        
        # Load metadata
        metadata = None
        metadata_path = model_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Validate model
        if validate:
            self._validate_model(model)
        
        return {
            'model': model,
            'metadata': metadata,
            'model_path': str(model_path),
            'model_name': model_name,
            'version': version or 'latest'
        }
    
    def _validate_model(self, model: Any):
        """
        Validate loaded model
        
        Args:
            model: Model to validate
            
        Raises:
            ValueError: If model is invalid
        """
        if model is None:
            raise ValueError("Model is None")
        
        # Check for required methods
        if not hasattr(model, 'predict'):
            warnings.warn("Model does not have 'predict' method")
        
        # Check if model is callable or has predict
        if not (callable(model) or hasattr(model, 'predict')):
            warnings.warn("Model may not be usable")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for model_dir in self.storage_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                
                # Check for versions
                versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
                if versions:
                    for version in versions:
                        metadata_path = model_dir / version / 'metadata.json'
                        if metadata_path.exists():
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            models.append({
                                'model_name': model_name,
                                'version': version,
                                'metadata': metadata
                            })
                else:
                    # No version subdirectories
                    metadata_path = model_dir / 'metadata.json'
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        models.append({
                            'model_name': model_name,
                            'version': 'latest',
                            'metadata': metadata
                        })
        
        return models
    
    def delete_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ):
        """
        Delete a saved model
        
        Args:
            model_name: Name of the model
            version: Model version (optional)
        """
        if version:
            model_dir = self.storage_dir / model_name / version
        else:
            model_dir = self.storage_dir / model_name
        
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
        else:
            raise FileNotFoundError(f"Model {model_name} (version {version}) not found")
