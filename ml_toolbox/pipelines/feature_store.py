"""
Feature Store

Stores and manages features for reuse across training and inference pipelines.
"""
from typing import Any, Dict, Optional, List
import logging
import numpy as np
from datetime import datetime
import hashlib
import pickle
import os

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Feature Store for persisting and retrieving features
    
    Provides:
    - Feature storage and retrieval
    - Feature versioning
    - Feature metadata tracking
    - Feature reuse across pipelines
    """
    
    def __init__(self, storage_path: Optional[str] = None, enable_disk_storage: bool = False):
        """
        Initialize feature store
        
        Parameters
        ----------
        storage_path : str, optional
            Path for disk storage (if enabled)
        enable_disk_storage : bool, default=False
            Whether to persist features to disk
        """
        self.storage_path = storage_path or "feature_store"
        self.enable_disk_storage = enable_disk_storage
        self.features: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        if enable_disk_storage:
            os.makedirs(self.storage_path, exist_ok=True)
            logger.info(f"[FeatureStore] Disk storage enabled: {self.storage_path}")
    
    def store(self, features: np.ndarray, name: str, version: Optional[str] = None,
              metadata: Optional[Dict] = None) -> str:
        """
        Store features
        
        Parameters
        ----------
        features : array-like
            Features to store
        name : str
            Feature set name
        version : str, optional
            Version identifier (default: auto-generated)
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        feature_id : str
            Feature identifier
        """
        features = np.asarray(features)
        
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        feature_id = f"{name}:{version}"
        
        # Compute feature hash for deduplication
        feature_hash = self._compute_hash(features)
        
        # Store features
        self.features[feature_id] = {
            'features': features,
            'name': name,
            'version': version,
            'hash': feature_hash,
            'timestamp': datetime.now().isoformat(),
            'shape': features.shape,
            'dtype': str(features.dtype),
            'metadata': metadata or {}
        }
        
        # Store metadata
        self.metadata[feature_id] = {
            'name': name,
            'version': version,
            'hash': feature_hash,
            'timestamp': datetime.now().isoformat(),
            'shape': features.shape,
            'dtype': str(features.dtype),
            'metadata': metadata or {}
        }
        
        # Persist to disk if enabled
        if self.enable_disk_storage:
            self._save_to_disk(feature_id, features, metadata)
        
        logger.info(f"[FeatureStore] Stored features: {feature_id} (shape: {features.shape})")
        return feature_id
    
    def get(self, name: str, version: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get stored features
        
        Parameters
        ----------
        name : str
            Feature set name
        version : str, optional
            Version identifier (default: latest)
            
        Returns
        -------
        features : array-like or None
            Stored features or None if not found
        """
        if version:
            feature_id = f"{name}:{version}"
        else:
            # Get latest version
            matching = [k for k in self.features.keys() if k.startswith(f"{name}:")]
            if not matching:
                return None
            feature_id = max(matching, key=lambda k: self.features[k]['timestamp'])
        
        if feature_id in self.features:
            logger.info(f"[FeatureStore] Retrieved features: {feature_id}")
            return self.features[feature_id]['features']
        
        # Try loading from disk if enabled
        if self.enable_disk_storage:
            features = self._load_from_disk(feature_id)
            if features is not None:
                return features
        
        return None
    
    def list_features(self, name: Optional[str] = None) -> List[str]:
        """
        List stored features
        
        Parameters
        ----------
        name : str, optional
            Filter by feature set name
            
        Returns
        -------
        feature_ids : list of str
            List of feature identifiers
        """
        if name:
            return [k for k in self.features.keys() if k.startswith(f"{name}:")]
        return list(self.features.keys())
    
    def get_metadata(self, name: str, version: Optional[str] = None) -> Optional[Dict]:
        """
        Get feature metadata
        
        Parameters
        ----------
        name : str
            Feature set name
        version : str, optional
            Version identifier (default: latest)
            
        Returns
        -------
        metadata : dict or None
            Feature metadata or None if not found
        """
        if version:
            feature_id = f"{name}:{version}"
        else:
            matching = [k for k in self.metadata.keys() if k.startswith(f"{name}:")]
            if not matching:
                return None
            feature_id = max(matching, key=lambda k: self.metadata[k]['timestamp'])
        
        return self.metadata.get(feature_id)
    
    def _compute_hash(self, features: np.ndarray) -> str:
        """Compute hash of features for deduplication"""
        features_bytes = features.tobytes()
        return hashlib.md5(features_bytes).hexdigest()
    
    def _save_to_disk(self, feature_id: str, features: np.ndarray, metadata: Optional[Dict]):
        """Save features to disk"""
        try:
            file_path = os.path.join(self.storage_path, f"{feature_id}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'features': features,
                    'metadata': metadata or {}
                }, f)
            logger.debug(f"[FeatureStore] Saved to disk: {file_path}")
        except Exception as e:
            logger.warning(f"[FeatureStore] Failed to save to disk: {e}")
    
    def _load_from_disk(self, feature_id: str) -> Optional[np.ndarray]:
        """Load features from disk"""
        try:
            file_path = os.path.join(self.storage_path, f"{feature_id}.pkl")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    logger.debug(f"[FeatureStore] Loaded from disk: {file_path}")
                    return data['features']
        except Exception as e:
            logger.warning(f"[FeatureStore] Failed to load from disk: {e}")
        return None
    
    def clear(self, name: Optional[str] = None):
        """
        Clear stored features
        
        Parameters
        ----------
        name : str, optional
            Clear only features with this name (default: clear all)
        """
        if name:
            keys_to_remove = [k for k in self.features.keys() if k.startswith(f"{name}:")]
            for key in keys_to_remove:
                del self.features[key]
                del self.metadata[key]
            logger.info(f"[FeatureStore] Cleared features: {name}")
        else:
            self.features = {}
            self.metadata = {}
            logger.info("[FeatureStore] Cleared all features")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get feature store summary"""
        return {
            'total_features': len(self.features),
            'feature_sets': list(set(k.split(':')[0] for k in self.features.keys())),
            'disk_storage_enabled': self.enable_disk_storage,
            'storage_path': self.storage_path if self.enable_disk_storage else None
        }
