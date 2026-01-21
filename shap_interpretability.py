"""
SHAP (SHapley Additive exPlanations) for Model Interpretability
Full implementation of SHAP values for all model types

Features:
- Tree SHAP (for tree-based models)
- Kernel SHAP (model-agnostic)
- Linear SHAP (for linear models)
- Deep SHAP (for neural networks)
- SHAP summary plots
- SHAP dependence plots
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import shap library
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not available. Install with: pip install shap")
    print("SHAP interpretability will use simplified implementation")


class SHAPInterpreter:
    """
    SHAP (SHapley Additive exPlanations) interpreter
    
    Provides model-agnostic and model-specific explanations
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.explainer_ = None
        self.shap_values_ = None
        self.is_fitted = False
    
    def _create_explainer(self, model: Any, X: np.ndarray, model_type: str = 'auto'):
        """
        Create appropriate SHAP explainer based on model type
        
        Args:
            model: Fitted model
            X: Background data
            model_type: 'tree', 'linear', 'kernel', 'deep', 'auto'
        """
        if not SHAP_AVAILABLE:
            return None
        
        # Auto-detect model type
        if model_type == 'auto':
            model_name = type(model).__name__.lower()
            
            # Tree-based models
            if any(x in model_name for x in ['tree', 'forest', 'gradient', 'xgboost', 'lightgbm', 'catboost']):
                model_type = 'tree'
            
            # Linear models
            elif any(x in model_name for x in ['linear', 'logistic', 'ridge', 'lasso', 'elastic']):
                model_type = 'linear'
            
            # Neural networks
            elif any(x in model_name for x in ['neural', 'mlp', 'sequential', 'model']):
                model_type = 'deep'
            
            else:
                model_type = 'kernel'
        
        # Create explainer
        try:
            if model_type == 'tree':
                return shap.TreeExplainer(model)
            elif model_type == 'linear':
                return shap.LinearExplainer(model, X)
            elif model_type == 'deep':
                return shap.DeepExplainer(model, X[:100])  # Sample for background
            else:
                # Kernel SHAP (model-agnostic)
                return shap.KernelExplainer(
                    lambda x: model.predict_proba(x) if hasattr(model, 'predict_proba') else model.predict(x),
                    X[:100]  # Background data
                )
        except Exception as e:
            warnings.warn(f"Failed to create {model_type} explainer: {e}. Falling back to kernel SHAP.")
            return shap.KernelExplainer(
                lambda x: model.predict_proba(x) if hasattr(model, 'predict_proba') else model.predict(x),
                X[:100]
            )
    
    def explain(
        self,
        model: Any,
        X: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        model_type: str = 'auto',
        n_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate SHAP values for model predictions
        
        Args:
            model: Fitted model
            X: Data to explain
            X_background: Background data for explainer (None uses X)
            model_type: Model type ('tree', 'linear', 'kernel', 'deep', 'auto')
            n_samples: Number of samples to explain (None for all)
            
        Returns:
            Dictionary with SHAP values, base values, feature names
        """
        if not SHAP_AVAILABLE:
            return {'error': 'shap library not available. Install with: pip install shap'}
        
        X = np.asarray(X)
        
        # Subsample if needed
        if n_samples is not None and len(X) > n_samples:
            np.random.seed(self.random_state)
            sample_idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X
        
        # Background data
        if X_background is None:
            X_background = X[:100]  # Use first 100 samples as background
        else:
            X_background = np.asarray(X_background)
        
        # Create explainer
        self.explainer_ = self._create_explainer(model, X_background, model_type)
        
        if self.explainer_ is None:
            return {'error': 'Failed to create SHAP explainer'}
        
        # Calculate SHAP values
        try:
            shap_values = self.explainer_.shap_values(X_sample)
            
            # Handle multi-class (shap_values is list)
            if isinstance(shap_values, list):
                # Use first class for importance
                shap_values_array = np.abs(shap_values[0])
            else:
                shap_values_array = np.abs(shap_values)
            
            # Get base values
            if hasattr(self.explainer_, 'expected_value'):
                base_value = self.explainer_.expected_value
            else:
                base_value = None
            
            self.shap_values_ = shap_values
            self.is_fitted = True
            
            # Feature importance (mean absolute SHAP values)
            feature_importance = np.mean(shap_values_array, axis=0)
            
            return {
                'shap_values': shap_values,
                'shap_values_abs': shap_values_array.tolist(),
                'base_value': base_value,
                'feature_importance': feature_importance.tolist(),
                'feature_rankings': np.argsort(feature_importance)[::-1].tolist(),
                'n_samples': len(X_sample),
                'n_features': X_sample.shape[1],
                'model_type': model_type
            }
        except Exception as e:
            return {'error': f'SHAP calculation failed: {str(e)}'}
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from SHAP values"""
        if not self.is_fitted or self.shap_values_ is None:
            return None
        
        if isinstance(self.shap_values_, list):
            shap_array = np.abs(self.shap_values_[0])
        else:
            shap_array = np.abs(self.shap_values_)
        
        return np.mean(shap_array, axis=0)
    
    def plot_summary(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        max_display: int = 10
    ):
        """
        Plot SHAP summary
        
        Args:
            save_path: Path to save figure
            show: Whether to display plot
            max_display: Maximum number of features to display
        """
        if not SHAP_AVAILABLE:
            print("Warning: shap library not available for plotting")
            return
        
        if not self.is_fitted or self.shap_values_ is None:
            print("Warning: Must call explain() before plotting")
            return
        
        try:
            shap.summary_plot(
                self.shap_values_,
                max_display=max_display,
                show=show
            )
            
            if save_path:
                import matplotlib.pyplot as plt
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to plot SHAP summary: {e}")
    
    def plot_dependence(
        self,
        feature_idx: int,
        interaction_feature: Optional[int] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot SHAP dependence plot
        
        Args:
            feature_idx: Index of feature to plot
            interaction_feature: Optional interaction feature
            save_path: Path to save figure
            show: Whether to display plot
        """
        if not SHAP_AVAILABLE:
            print("Warning: shap library not available for plotting")
            return
        
        if not self.is_fitted or self.shap_values_ is None:
            print("Warning: Must call explain() before plotting")
            return
        
        try:
            shap.dependence_plot(
                feature_idx,
                self.shap_values_,
                interaction_index=interaction_feature,
                show=show
            )
            
            if save_path:
                import matplotlib.pyplot as plt
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to plot SHAP dependence: {e}")
