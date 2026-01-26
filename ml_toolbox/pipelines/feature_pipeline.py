"""
Feature Pipeline

Stages:
1. Data Ingestion
2. Preprocessing
3. Feature Engineering
4. Feature Selection
5. Feature Store
"""
import numpy as np
from typing import Any, Dict, Optional, List
import logging

from .base import BasePipeline, PipelineStage
from .feature_store import FeatureStore

logger = logging.getLogger(__name__)


class DataIngestionStage(PipelineStage):
    """Stage 1: Data Ingestion"""
    
    def __init__(self, toolbox=None):
        super().__init__("data_ingestion")
        self.toolbox = toolbox
    
    def execute(self, input_data: Any, **kwargs) -> np.ndarray:
        """Ingest and validate data"""
        if isinstance(input_data, np.ndarray):
            data = input_data
        else:
            data = np.asarray(input_data)
        
        # Validate data
        if data.size == 0:
            raise ValueError("Empty data provided")
        
        self.metadata['shape'] = data.shape
        self.metadata['dtype'] = str(data.dtype)
        
        return data


class PreprocessingStage(PipelineStage):
    """Stage 2: Preprocessing"""
    
    def __init__(self, toolbox=None, method: str = 'standardize'):
        super().__init__("preprocessing")
        self.toolbox = toolbox
        self.method = method
    
    def execute(self, input_data: np.ndarray, **kwargs) -> np.ndarray:
        """Preprocess data"""
        method = kwargs.get('method', self.method)
        
        if self.toolbox and hasattr(self.toolbox, 'preprocess'):
            # Use toolbox preprocessing
            processed = self.toolbox.preprocess(input_data, method=method)
        elif self.toolbox and hasattr(self.toolbox, 'computational_kernel') and self.toolbox.computational_kernel:
            # Use computational kernel
            if method == 'standardize':
                processed = self.toolbox.computational_kernel.standardize(input_data)
            elif method == 'normalize':
                processed = self.toolbox.computational_kernel.normalize(input_data)
            else:
                processed = input_data
        else:
            # Fallback to NumPy
            if method == 'standardize':
                mean = np.mean(input_data, axis=0, keepdims=True)
                std = np.std(input_data, axis=0, keepdims=True)
                std = np.where(std < 1e-10, 1.0, std)
                processed = (input_data - mean) / std
            elif method == 'normalize':
                min_val = np.min(input_data, axis=0, keepdims=True)
                max_val = np.max(input_data, axis=0, keepdims=True)
                range_val = max_val - min_val
                range_val = np.where(range_val < 1e-10, 1.0, range_val)
                processed = (input_data - min_val) / range_val
            else:
                processed = input_data
        
        self.metadata['method'] = method
        self.metadata['shape'] = processed.shape
        
        return processed


class FeatureEngineeringStage(PipelineStage):
    """Stage 3: Feature Engineering"""
    
    def __init__(self, toolbox=None):
        super().__init__("feature_engineering")
        self.toolbox = toolbox
    
    def execute(self, input_data: np.ndarray, **kwargs) -> np.ndarray:
        """Engineer features"""
        if self.toolbox and hasattr(self.toolbox, 'feature_kernel') and self.toolbox.feature_kernel:
            # Use feature engineering kernel
            operations = kwargs.get('operations', ['standardize'])
            engineered = self.toolbox.feature_kernel.transform(input_data, operations=operations)
        else:
            # Pass through for now (can add polynomial features, etc.)
            engineered = input_data
        
        self.metadata['shape'] = engineered.shape
        self.metadata['operations'] = kwargs.get('operations', [])
        
        return engineered


class FeatureSelectionStage(PipelineStage):
    """Stage 4: Feature Selection"""
    
    def __init__(self, toolbox=None, max_features: Optional[int] = None,
                 method: str = 'mutual_information'):
        """
        Initialize feature selection stage
        
        Parameters
        ----------
        toolbox : MLToolbox
            ML Toolbox instance
        max_features : int, optional
            Maximum number of features to select
        method : str
            Selection method ('mutual_information', 'variance', 'information_gain')
        """
        super().__init__("feature_selection")
        self.toolbox = toolbox
        self.max_features = max_features
        self.method = method
    
    def execute(self, input_data: np.ndarray, **kwargs) -> np.ndarray:
        """Select features using information-theoretic or variance-based methods"""
        max_features = kwargs.get('max_features', self.max_features)
        method = kwargs.get('selection_method', kwargs.get('method', self.method))  # Allow both for compatibility
        target = kwargs.get('target')  # Target variable for MI/IG-based selection
        
        if max_features and input_data.shape[1] > max_features:
            if method == 'mutual_information' and target is not None:
                # Mutual Information-based selection
                try:
                    from ml_toolbox.textbook_concepts.information_theory import mutual_information
                    
                    target = np.asarray(target).ravel()
                    mi_scores = []
                    
                    for feature_idx in range(input_data.shape[1]):
                        feature = input_data[:, feature_idx]
                        mi = mutual_information(feature, target, n_bins=10)
                        mi_scores.append(mi)
                    
                    mi_scores = np.array(mi_scores)
                    top_indices = np.argsort(mi_scores)[-max_features:]
                    selected = input_data[:, top_indices]
                    
                    self.metadata['method'] = 'mutual_information'
                    self.metadata['mi_scores'] = mi_scores[top_indices].tolist()
                    self.metadata['selected_indices'] = top_indices.tolist()
                    
                except Exception as e:
                    logger.warning(f"[FeatureSelection] MI selection failed: {e}, falling back to variance")
                    method = 'variance'
            
            if method == 'information_gain' and target is not None:
                # Information Gain-based selection (for discrete features)
                try:
                    from ml_toolbox.textbook_concepts.information_theory import information_gain, entropy
                    
                    target = np.asarray(target).ravel()
                    ig_scores = []
                    
                    for feature_idx in range(input_data.shape[1]):
                        feature = input_data[:, feature_idx]
                        # Discretize feature for IG calculation
                        feature_binned = np.digitize(feature, np.linspace(feature.min(), feature.max(), 5))
                        
                        # Calculate information gain
                        unique_values = np.unique(feature_binned)
                        y_splits = [target[feature_binned == val] for val in unique_values]
                        ig = information_gain(target, y_splits)
                        ig_scores.append(ig)
                    
                    ig_scores = np.array(ig_scores)
                    top_indices = np.argsort(ig_scores)[-max_features:]
                    selected = input_data[:, top_indices]
                    
                    self.metadata['method'] = 'information_gain'
                    self.metadata['ig_scores'] = ig_scores[top_indices].tolist()
                    self.metadata['selected_indices'] = top_indices.tolist()
                    
                except Exception as e:
                    logger.warning(f"[FeatureSelection] IG selection failed: {e}, falling back to variance")
                    method = 'variance'
            
            if method == 'variance':
                # Variance-based selection (fallback)
                variances = np.var(input_data, axis=0)
                top_indices = np.argsort(variances)[-max_features:]
                selected = input_data[:, top_indices]
                
                self.metadata['method'] = 'variance'
                self.metadata['variances'] = variances[top_indices].tolist()
                self.metadata['selected_indices'] = top_indices.tolist()
            
            self.metadata['max_features'] = max_features
        else:
            selected = input_data
            self.metadata['max_features'] = None
            self.metadata['method'] = 'none'
        
        self.metadata['shape'] = selected.shape
        
        return selected


class FeatureStoreStage(PipelineStage):
    """Stage 5: Feature Store"""
    
    def __init__(self, toolbox=None, feature_store: Optional[FeatureStore] = None, store_features: bool = True):
        super().__init__("feature_store")
        self.toolbox = toolbox
        self.feature_store = feature_store
        self.store_features = store_features
    
    def execute(self, input_data: np.ndarray, **kwargs) -> np.ndarray:
        """Store features"""
        if self.store_features and self.feature_store:
            feature_name = kwargs.get('feature_name', 'default')
            version = kwargs.get('version')
            metadata = kwargs.get('metadata', {})
            
            feature_id = self.feature_store.store(
                input_data,
                name=feature_name,
                version=version,
                metadata=metadata
            )
            
            self.metadata['feature_id'] = feature_id
            self.metadata['stored'] = True
        else:
            self.metadata['stored'] = False
        
        return input_data


class FeaturePipeline(BasePipeline):
    """
    Feature Pipeline
    
    Orchestrates:
    1. Data Ingestion
    2. Preprocessing
    3. Feature Engineering
    4. Feature Selection
    5. Feature Store
    """
    
    def __init__(self, toolbox=None, feature_store: Optional[FeatureStore] = None,
                 enable_feature_store: bool = True):
        """
        Initialize feature pipeline
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
        feature_store : FeatureStore, optional
            Feature store instance (default: create new)
        enable_feature_store : bool, default=True
            Whether to enable feature storage
        """
        super().__init__("feature_pipeline", toolbox)
        
        # Create feature store if not provided
        if feature_store is None and enable_feature_store:
            feature_store = FeatureStore(enable_disk_storage=False)
        
        # Add stages
        self.add_stage(DataIngestionStage(toolbox))
        self.add_stage(PreprocessingStage(toolbox))
        self.add_stage(FeatureEngineeringStage(toolbox))
        self.add_stage(FeatureSelectionStage(toolbox))
        self.add_stage(FeatureStoreStage(toolbox, feature_store, enable_feature_store))
        
        self.feature_store = feature_store
    
    def execute(self, X: np.ndarray, feature_name: str = "default", **kwargs) -> np.ndarray:
        """
        Execute feature pipeline
        
        Parameters
        ----------
        X : array-like
            Input data
        feature_name : str, default="default"
            Name for stored features
        **kwargs
            Additional parameters for stages
            
        Returns
        -------
        X_features : array-like
            Processed features
        """
        X = np.asarray(X)
        
        # Start monitoring if enabled
        if self.monitor:
            metrics = self.monitor.start_pipeline(self.name)
        
        # Execute stages sequentially
        result = X
        for stage in self.stages:
            if stage.enabled:
                result = stage.run(result, monitor=self.monitor, retry_handler=self.retry_handler,
                                  debugger=self.debugger, feature_name=feature_name, **kwargs)
                self.state[stage.name] = {
                    'output_shape': result.shape,
                    'metadata': stage.metadata
                }
        
        # End monitoring if enabled
        if self.monitor and self.monitor.current_metrics:
            self.monitor.end_pipeline()
        
        # Store final features in state
        self.state['final_features'] = result
        self.state['feature_name'] = feature_name
        
        logger.info(f"[FeaturePipeline] Pipeline completed. Output shape: {result.shape}")
        
        return result
    
    def get_stored_features(self, feature_name: str = "default", version: Optional[str] = None) -> Optional[np.ndarray]:
        """Get stored features from feature store"""
        if self.feature_store:
            return self.feature_store.get(feature_name, version)
        return None
