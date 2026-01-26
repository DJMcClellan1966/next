"""
Pipeline Persistence

Save and load pipeline state, configurations, and models for reproducibility.
"""
from typing import Any, Dict, Optional, List
import logging
import pickle
import json
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelinePersistence:
    """
    Persist pipeline state, configurations, and models
    
    Provides:
    - Save/load pipeline state
    - Save/load pipeline configurations
    - Save/load models
    - Pipeline versioning
    - Reproducibility
    """
    
    def __init__(self, storage_dir: str = "pipeline_storage", enable_compression: bool = False):
        """
        Initialize pipeline persistence
        
        Parameters
        ----------
        storage_dir : str, default="pipeline_storage"
            Directory for storing pipeline data
        enable_compression : bool, default=False
            Whether to compress stored data
        """
        self.storage_dir = Path(storage_dir)
        self.enable_compression = enable_compression
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.state_dir = self.storage_dir / "states"
        self.config_dir = self.storage_dir / "configs"
        self.model_dir = self.storage_dir / "models"
        self.metrics_dir = self.storage_dir / "metrics"
        
        for dir_path in [self.state_dir, self.config_dir, self.model_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[PipelinePersistence] Initialized with storage: {self.storage_dir}")
    
    def save_pipeline_state(self, pipeline_name: str, state: Dict[str, Any],
                           version: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """
        Save pipeline state
        
        Parameters
        ----------
        pipeline_name : str
            Pipeline name
        state : dict
            Pipeline state to save
        version : str, optional
            Version identifier (default: timestamp)
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        state_id : str
            State identifier
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        state_id = f"{pipeline_name}_{version}"
        filepath = self.state_dir / f"{state_id}.pkl"
        
        save_data = {
            'pipeline_name': pipeline_name,
            'version': version,
            'state': state,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"[PipelinePersistence] Saved pipeline state: {state_id}")
        return state_id
    
    def load_pipeline_state(self, pipeline_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load pipeline state
        
        Parameters
        ----------
        pipeline_name : str
            Pipeline name
        version : str, optional
            Version identifier (default: latest)
            
        Returns
        -------
        state : dict or None
            Pipeline state or None if not found
        """
        if version:
            state_id = f"{pipeline_name}_{version}"
            filepath = self.state_dir / f"{state_id}.pkl"
        else:
            # Find latest version
            pattern = f"{pipeline_name}_*.pkl"
            matching_files = list(self.state_dir.glob(pattern))
            if not matching_files:
                return None
            filepath = max(matching_files, key=lambda p: p.stat().st_mtime)
        
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"[PipelinePersistence] Loaded pipeline state: {filepath.name}")
            return data['state']
        
        return None
    
    def save_pipeline_config(self, pipeline_name: str, config: Dict[str, Any],
                            version: Optional[str] = None) -> str:
        """Save pipeline configuration"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        config_id = f"{pipeline_name}_{version}"
        filepath = self.config_dir / f"{config_id}.json"
        
        save_data = {
            'pipeline_name': pipeline_name,
            'version': version,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"[PipelinePersistence] Saved pipeline config: {config_id}")
        return config_id
    
    def load_pipeline_config(self, pipeline_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load pipeline configuration"""
        if version:
            config_id = f"{pipeline_name}_{version}"
            filepath = self.config_dir / f"{config_id}.json"
        else:
            pattern = f"{pipeline_name}_*.json"
            matching_files = list(self.config_dir.glob(pattern))
            if not matching_files:
                return None
            filepath = max(matching_files, key=lambda p: p.stat().st_mtime)
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"[PipelinePersistence] Loaded pipeline config: {filepath.name}")
            return data['config']
        
        return None
    
    def save_model(self, model: Any, model_name: str, version: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> str:
        """Save model"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_id = f"{model_name}_{version}"
        filepath = self.model_dir / f"{model_id}.pkl"
        
        save_data = {
            'model_name': model_name,
            'version': version,
            'model': model,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"[PipelinePersistence] Saved model: {model_id}")
        return model_id
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """Load model"""
        if version:
            model_id = f"{model_name}_{version}"
            filepath = self.model_dir / f"{model_id}.pkl"
        else:
            pattern = f"{model_name}_*.pkl"
            matching_files = list(self.model_dir.glob(pattern))
            if not matching_files:
                return None
            filepath = max(matching_files, key=lambda p: p.stat().st_mtime)
        
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"[PipelinePersistence] Loaded model: {filepath.name}")
            return data['model']
        
        return None
    
    def list_versions(self, pipeline_name: str, component: str = 'state') -> List[str]:
        """List available versions for a pipeline component"""
        if component == 'state':
            pattern = f"{pipeline_name}_*.pkl"
            files = list(self.state_dir.glob(pattern))
        elif component == 'config':
            pattern = f"{pipeline_name}_*.json"
            files = list(self.config_dir.glob(pattern))
        elif component == 'model':
            pattern = f"{pipeline_name}_*.pkl"
            files = list(self.model_dir.glob(pattern))
        else:
            return []
        
        versions = []
        for file in files:
            # Extract version from filename: pipeline_name_version.ext
            parts = file.stem.split('_', 1)
            if len(parts) == 2:
                versions.append(parts[1])
        
        return sorted(versions)
    
    def delete_version(self, pipeline_name: str, version: str, component: str = 'state'):
        """Delete a specific version"""
        if component == 'state':
            filepath = self.state_dir / f"{pipeline_name}_{version}.pkl"
        elif component == 'config':
            filepath = self.config_dir / f"{pipeline_name}_{version}.json"
        elif component == 'model':
            filepath = self.model_dir / f"{pipeline_name}_{version}.pkl"
        else:
            return False
        
        if filepath.exists():
            filepath.unlink()
            logger.info(f"[PipelinePersistence] Deleted {component}: {filepath.name}")
            return True
        
        return False
