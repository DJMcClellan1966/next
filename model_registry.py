"""
Model Registry & Versioning
Production-ready model management with semantic versioning, staging, and deployment workflows

Features:
- Semantic versioning (MAJOR.MINOR.PATCH)
- Model staging (dev, staging, production)
- Model lineage tracking
- Model metadata management
- Deployment workflows
- A/B testing integration
- Model rollback capabilities
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import json
import datetime
import pickle
import hashlib
import shutil
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent))


class ModelStage(Enum):
    """Model deployment stages"""
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelVersion:
    """Model version with metadata"""
    
    def __init__(
        self,
        version: str,
        model: Any,
        metadata: Dict[str, Any],
        stage: ModelStage = ModelStage.DEV
    ):
        """
        Args:
            version: Semantic version (e.g., "1.2.3")
            model: Model object
            metadata: Model metadata
            stage: Deployment stage
        """
        self.version = version
        self.model = model
        self.metadata = metadata
        self.stage = stage
        self.created_at = datetime.datetime.now()
        self.updated_at = datetime.datetime.now()
        self.model_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate model hash for integrity checking"""
        try:
            model_bytes = pickle.dumps(self.model)
            return hashlib.sha256(model_bytes).hexdigest()[:16]
        except:
            return "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version': self.version,
            'stage': self.stage.value,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'model_hash': self.model_hash
        }


class ModelRegistry:
    """
    Model Registry & Versioning
    
    Production-ready model management
    """
    
    def __init__(self, registry_path: str = "model_registry"):
        """
        Args:
            registry_path: Path to store models and metadata
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "metadata.json"
        self.versions: Dict[str, ModelVersion] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for version_str, version_data in data.items():
                        # Load model from disk
                        model_path = self.registry_path / version_str / "model.pkl"
                        if model_path.exists():
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                            
                            version = ModelVersion(
                                version=version_data['version'],
                                model=model,
                                metadata=version_data['metadata'],
                                stage=ModelStage(version_data['stage'])
                            )
                            version.created_at = datetime.datetime.fromisoformat(version_data['created_at'])
                            version.updated_at = datetime.datetime.fromisoformat(version_data['updated_at'])
                            self.versions[version_str] = version
        except Exception as e:
            print(f"Error loading registry: {e}")
            self.versions = {}
    
    def _save_registry(self):
        """Save registry to disk"""
        try:
            metadata = {}
            for version_str, version in self.versions.items():
                # Save model to disk
                model_dir = self.registry_path / version_str
                model_dir.mkdir(parents=True, exist_ok=True)
                
                model_path = model_dir / "model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(version.model, f)
                
                # Save metadata
                metadata[version_str] = version.to_dict()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving registry: {e}")
    
    def register_model(
        self,
        model: Any,
        metadata: Dict[str, Any],
        version: Optional[str] = None,
        stage: ModelStage = ModelStage.DEV
    ) -> str:
        """
        Register a new model version
        
        Args:
            model: Model object
            metadata: Model metadata (metrics, parameters, etc.)
            version: Version string (auto-increment if None)
            stage: Initial deployment stage
            
        Returns:
            Version string
        """
        if version is None:
            version = self._get_next_version()
        
        # Validate semantic version
        if not self._is_valid_version(version):
            raise ValueError(f"Invalid version format: {version}. Use MAJOR.MINOR.PATCH")
        
        version_obj = ModelVersion(
            version=version,
            model=model,
            metadata=metadata,
            stage=stage
        )
        
        self.versions[version] = version_obj
        self._save_registry()
        
        return version
    
    def _get_next_version(self) -> str:
        """Get next version (increment patch)"""
        if not self.versions:
            return "1.0.0"
        
        # Get latest version
        latest = self._get_latest_version()
        if latest:
            major, minor, patch = map(int, latest.split('.'))
            return f"{major}.{minor}.{patch + 1}"
        
        return "1.0.0"
    
    def _get_latest_version(self) -> Optional[str]:
        """Get latest version string"""
        if not self.versions:
            return None
        
        versions = [v for v in self.versions.keys() if self._is_valid_version(v)]
        if not versions:
            return None
        
        # Sort by version
        versions.sort(key=lambda v: tuple(map(int, v.split('.'))))
        return versions[-1]
    
    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid semantic version"""
        try:
            parts = version.split('.')
            if len(parts) != 3:
                return False
            for part in parts:
                int(part)
            return True
        except:
            return False
    
    def get_model(self, version: str) -> Optional[ModelVersion]:
        """Get model by version"""
        return self.versions.get(version)
    
    def get_models_by_stage(self, stage: ModelStage) -> List[ModelVersion]:
        """Get all models in a stage"""
        return [v for v in self.versions.values() if v.stage == stage]
    
    def promote_model(self, version: str, target_stage: ModelStage) -> bool:
        """
        Promote model to target stage
        
        Args:
            version: Model version
            target_stage: Target deployment stage
            
        Returns:
            True if successful
        """
        if version not in self.versions:
            return False
        
        model_version = self.versions[version]
        
        # Validate promotion path
        if not self._can_promote(model_version.stage, target_stage):
            raise ValueError(
                f"Cannot promote from {model_version.stage.value} to {target_stage.value}"
            )
        
        model_version.stage = target_stage
        model_version.updated_at = datetime.datetime.now()
        self._save_registry()
        
        return True
    
    def _can_promote(self, from_stage: ModelStage, to_stage: ModelStage) -> bool:
        """Check if promotion is allowed"""
        promotion_paths = {
            ModelStage.DEV: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.ARCHIVED],
            ModelStage.PRODUCTION: [ModelStage.ARCHIVED],
            ModelStage.ARCHIVED: []
        }
        return to_stage in promotion_paths.get(from_stage, [])
    
    def rollback_production(self, target_version: str) -> bool:
        """
        Rollback production to target version
        
        Args:
            target_version: Version to rollback to
            
        Returns:
            True if successful
        """
        if target_version not in self.versions:
            return False
        
        # Archive current production
        production_models = self.get_models_by_stage(ModelStage.PRODUCTION)
        for model in production_models:
            model.stage = ModelStage.ARCHIVED
            model.updated_at = datetime.datetime.now()
        
        # Promote target to production
        target_model = self.versions[target_version]
        target_model.stage = ModelStage.PRODUCTION
        target_model.updated_at = datetime.datetime.now()
        
        self._save_registry()
        return True
    
    def list_versions(
        self,
        stage: Optional[ModelStage] = None,
        sort_by: str = 'version'
    ) -> List[ModelVersion]:
        """
        List model versions
        
        Args:
            stage: Filter by stage (optional)
            sort_by: Sort by 'version', 'created_at', or 'updated_at'
            
        Returns:
            List of model versions
        """
        versions = list(self.versions.values())
        
        # Filter by stage
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        # Sort
        if sort_by == 'version':
            versions.sort(key=lambda v: tuple(map(int, v.version.split('.'))))
        elif sort_by == 'created_at':
            versions.sort(key=lambda v: v.created_at)
        elif sort_by == 'updated_at':
            versions.sort(key=lambda v: v.updated_at)
        
        return versions
    
    def get_model_lineage(self, version: str) -> Dict[str, Any]:
        """
        Get model lineage (parent models, experiments, etc.)
        
        Args:
            version: Model version
            
        Returns:
            Lineage information
        """
        if version not in self.versions:
            return {}
        
        model_version = self.versions[version]
        metadata = model_version.metadata
        
        lineage = {
            'version': version,
            'parent_version': metadata.get('parent_version'),
            'experiment_id': metadata.get('experiment_id'),
            'base_model': metadata.get('base_model'),
            'fine_tuned_from': metadata.get('fine_tuned_from'),
            'created_at': model_version.created_at.isoformat(),
            'stage': model_version.stage.value
        }
        
        return lineage
    
    def delete_model(self, version: str) -> bool:
        """
        Delete model version
        
        Args:
            version: Model version to delete
            
        Returns:
            True if successful
        """
        if version not in self.versions:
            return False
        
        # Don't allow deleting production models
        if self.versions[version].stage == ModelStage.PRODUCTION:
            raise ValueError("Cannot delete production models. Archive first.")
        
        # Delete model files
        model_dir = self.registry_path / version
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del self.versions[version]
        self._save_registry()
        
        return True
    
    def compare_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions
        
        Args:
            version1: First model version
            version2: Second model version
            
        Returns:
            Comparison results
        """
        if version1 not in self.versions or version2 not in self.versions:
            return {'error': 'One or both versions not found'}
        
        model1 = self.versions[version1]
        model2 = self.versions[version2]
        
        comparison = {
            'versions': [version1, version2],
            'metadata_diff': self._compare_metadata(
                model1.metadata,
                model2.metadata
            ),
            'metrics_diff': self._compare_metrics(
                model1.metadata.get('metrics', {}),
                model2.metadata.get('metrics', {})
            ),
            'stages': [model1.stage.value, model2.stage.value],
            'created_at': [
                model1.created_at.isoformat(),
                model2.created_at.isoformat()
            ]
        }
        
        return comparison
    
    def _compare_metadata(self, metadata1: Dict, metadata2: Dict) -> Dict[str, Any]:
        """Compare metadata dictionaries"""
        all_keys = set(metadata1.keys()) | set(metadata2.keys())
        diff = {}
        
        for key in all_keys:
            val1 = metadata1.get(key)
            val2 = metadata2.get(key)
            
            if val1 != val2:
                diff[key] = {
                    'version1': val1,
                    'version2': val2
                }
        
        return diff
    
    def _compare_metrics(self, metrics1: Dict, metrics2: Dict) -> Dict[str, Any]:
        """Compare metrics dictionaries"""
        all_keys = set(metrics1.keys()) | set(metrics2.keys())
        diff = {}
        
        for key in all_keys:
            val1 = metrics1.get(key, 0)
            val2 = metrics2.get(key, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff[key] = {
                    'version1': val1,
                    'version2': val2,
                    'difference': val2 - val1,
                    'percent_change': ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                }
        
        return diff
    
    def export_model(self, version: str, output_path: str) -> bool:
        """
        Export model to external path
        
        Args:
            version: Model version
            output_path: Output path
            
        Returns:
            True if successful
        """
        if version not in self.versions:
            return False
        
        model_version = self.versions[version]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy model
        model_path = self.registry_path / version / "model.pkl"
        if model_path.exists():
            shutil.copy(model_path, output_path)
        
        # Save metadata
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_version.to_dict(), f, indent=2, default=str)
        
        return True
