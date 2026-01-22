"""
Feature Store
Store, version, and serve features for ML models

Features:
- Feature storage and retrieval
- Feature versioning
- Feature lineage
- Online/offline feature serving
- Feature discovery
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("Warning: h5py not available. Install with: pip install h5py")


class FeatureStore:
    """
    Feature Store for ML Toolbox
    
    Stores features with versioning and lineage tracking
    """
    
    def __init__(
        self,
        storage_dir: Union[str, Path] = "feature_store",
        backend: str = 'pickle'  # 'pickle', 'hdf5', 'parquet'
    ):
        """
        Args:
            storage_dir: Directory to store features
            backend: Storage backend ('pickle', 'hdf5', 'parquet')
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        
        # Feature registry
        self.feature_registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load feature registry from disk"""
        registry_path = self.storage_dir / 'registry.json'
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.feature_registry = json.load(f)
    
    def _save_registry(self):
        """Save feature registry to disk"""
        registry_path = self.storage_dir / 'registry.json'
        with open(registry_path, 'w') as f:
            json.dump(self.feature_registry, f, indent=2, default=str)
    
    def register_feature(
        self,
        feature_name: str,
        features: Union[np.ndarray, pd.DataFrame],
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register and store features
        
        Args:
            feature_name: Name of the feature set
            features: Feature data (numpy array or DataFrame)
            version: Feature version (auto-generated if None)
            metadata: Additional metadata
            tags: Feature tags
            
        Returns:
            Feature version string
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to numpy if DataFrame
        if isinstance(features, pd.DataFrame):
            feature_array = features.values
            feature_names = features.columns.tolist()
        else:
            feature_array = np.asarray(features)
            feature_names = None
        
        # Create feature directory
        feature_dir = self.storage_dir / feature_name / version
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        if self.backend == 'pickle':
            feature_path = feature_dir / 'features.pkl'
            with open(feature_path, 'wb') as f:
                pickle.dump(feature_array, f)
        elif self.backend == 'hdf5' and H5PY_AVAILABLE:
            feature_path = feature_dir / 'features.h5'
            with h5py.File(feature_path, 'w') as f:
                f.create_dataset('features', data=feature_array)
        elif self.backend == 'parquet' and isinstance(features, pd.DataFrame):
            feature_path = feature_dir / 'features.parquet'
            features.to_parquet(feature_path)
        else:
            # Fallback to pickle
            feature_path = feature_dir / 'features.pkl'
            with open(feature_path, 'wb') as f:
                pickle.dump(feature_array, f)
        
        # Save metadata
        feature_metadata = {
            'feature_name': feature_name,
            'version': version,
            'shape': feature_array.shape,
            'dtype': str(feature_array.dtype),
            'feature_names': feature_names,
            'created_at': datetime.now().isoformat(),
            'backend': self.backend,
            'metadata': metadata or {},
            'tags': tags or []
        }
        
        metadata_path = feature_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(feature_metadata, f, indent=2, default=str)
        
        # Update registry
        if feature_name not in self.feature_registry:
            self.feature_registry[feature_name] = {
                'versions': [],
                'latest_version': None,
                'tags': []
            }
        
        self.feature_registry[feature_name]['versions'].append(version)
        self.feature_registry[feature_name]['latest_version'] = version
        if tags:
            self.feature_registry[feature_name]['tags'].extend(tags)
            self.feature_registry[feature_name]['tags'] = list(set(
                self.feature_registry[feature_name]['tags']
            ))
        
        self._save_registry()
        
        return version
    
    def get_feature(
        self,
        feature_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get features by name and version
        
        Args:
            feature_name: Name of the feature set
            version: Feature version (uses latest if None)
            
        Returns:
            Dictionary with features and metadata
        """
        if feature_name not in self.feature_registry:
            raise ValueError(f"Feature {feature_name} not found")
        
        if version is None:
            version = self.feature_registry[feature_name]['latest_version']
        
        feature_dir = self.storage_dir / feature_name / version
        
        if not feature_dir.exists():
            raise ValueError(f"Feature {feature_name} version {version} not found")
        
        # Load metadata
        metadata_path = feature_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load features
        if self.backend == 'pickle' or (feature_dir / 'features.pkl').exists():
            feature_path = feature_dir / 'features.pkl'
            with open(feature_path, 'rb') as f:
                features = pickle.load(f)
        elif self.backend == 'hdf5' and H5PY_AVAILABLE:
            feature_path = feature_dir / 'features.h5'
            with h5py.File(feature_path, 'r') as f:
                features = f['features'][:]
        elif (feature_dir / 'features.parquet').exists():
            feature_path = feature_dir / 'features.parquet'
            features = pd.read_parquet(feature_path)
        else:
            raise ValueError(f"Feature file not found for {feature_name} version {version}")
        
        # Convert to DataFrame if feature names available
        if metadata.get('feature_names') and isinstance(features, np.ndarray):
            features = pd.DataFrame(features, columns=metadata['feature_names'])
        
        return {
            'features': features,
            'metadata': metadata,
            'feature_name': feature_name,
            'version': version
        }
    
    def list_features(
        self,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all features
        
        Args:
            tag: Filter by tag (optional)
            
        Returns:
            List of feature information dictionaries
        """
        features = []
        
        for feature_name, info in self.feature_registry.items():
            if tag and tag not in info.get('tags', []):
                continue
            
            features.append({
                'feature_name': feature_name,
                'latest_version': info['latest_version'],
                'versions': info['versions'],
                'tags': info.get('tags', [])
            })
        
        return features
    
    def get_feature_lineage(
        self,
        feature_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get feature lineage (how features were created)
        
        Args:
            feature_name: Name of the feature set
            version: Feature version (uses latest if None)
            
        Returns:
            Dictionary with lineage information
        """
        feature_data = self.get_feature(feature_name, version)
        metadata = feature_data['metadata']
        
        lineage = {
            'feature_name': feature_name,
            'version': version or self.feature_registry[feature_name]['latest_version'],
            'created_at': metadata.get('created_at'),
            'source_metadata': metadata.get('metadata', {}),
            'tags': metadata.get('tags', []),
            'shape': metadata.get('shape'),
            'dtype': metadata.get('dtype')
        }
        
        return lineage
    
    def serve_features_online(
        self,
        feature_name: str,
        indices: Optional[np.ndarray] = None,
        version: Optional[str] = None
    ) -> np.ndarray:
        """
        Serve features online (low latency)
        
        Args:
            feature_name: Name of the feature set
            indices: Indices to retrieve (None for all)
            version: Feature version (uses latest if None)
            
        Returns:
            Feature array
        """
        feature_data = self.get_feature(feature_name, version)
        features = feature_data['features']
        
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        if indices is not None:
            return features[indices]
        else:
            return features
    
    def serve_features_offline(
        self,
        feature_name: str,
        version: Optional[str] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Serve features offline (batch processing)
        
        Args:
            feature_name: Name of the feature set
            version: Feature version (uses latest if None)
            
        Returns:
            Feature array or DataFrame
        """
        feature_data = self.get_feature(feature_name, version)
        return feature_data['features']
    
    def delete_feature(
        self,
        feature_name: str,
        version: Optional[str] = None
    ):
        """
        Delete a feature version
        
        Args:
            feature_name: Name of the feature set
            version: Feature version (deletes all if None)
        """
        if feature_name not in self.feature_registry:
            raise ValueError(f"Feature {feature_name} not found")
        
        if version:
            feature_dir = self.storage_dir / feature_name / version
            if feature_dir.exists():
                import shutil
                shutil.rmtree(feature_dir)
                # Update registry
                if version in self.feature_registry[feature_name]['versions']:
                    self.feature_registry[feature_name]['versions'].remove(version)
                    if self.feature_registry[feature_name]['latest_version'] == version:
                        versions = self.feature_registry[feature_name]['versions']
                        self.feature_registry[feature_name]['latest_version'] = (
                            versions[-1] if versions else None
                        )
                self._save_registry()
        else:
            # Delete all versions
            feature_dir = self.storage_dir / feature_name
            if feature_dir.exists():
                import shutil
                shutil.rmtree(feature_dir)
                del self.feature_registry[feature_name]
                self._save_registry()
