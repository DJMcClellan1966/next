"""
LIME (Local Interpretable Model-agnostic Explanations)
Local interpretability for any model

Features:
- LIME for tabular data
- LIME for text data
- Feature importance explanations
- Local model explanations
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import lime library
try:
    from lime import lime_tabular, lime_text
    from lime.lime_base import LimeBase
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: lime not available. Install with: pip install lime")
    print("LIME interpretability will use simplified implementation")


class LIMEInterpreter:
    """
    LIME (Local Interpretable Model-agnostic Explanations) interpreter
    
    Provides local explanations for individual predictions
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.explainer_ = None
        self.is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        mode: str = 'classification'
    ):
        """
        Fit LIME explainer
        
        Args:
            X: Training data
            y: Labels (optional)
            feature_names: Feature names
            mode: 'classification' or 'regression'
        """
        if not LIME_AVAILABLE:
            raise ImportError("lime library not available. Install with: pip install lime")
        
        X = np.asarray(X)
        
        # Create LIME explainer
        self.explainer_ = lime_tabular.LimeTabularExplainer(
            X,
            feature_names=feature_names,
            mode=mode,
            random_state=self.random_state
        )
        
        self.is_fitted = True
        return self
    
    def explain_instance(
        self,
        instance: np.ndarray,
        model: Any,
        num_features: int = 10,
        top_labels: int = 1
    ) -> Dict[str, Any]:
        """
        Explain a single instance
        
        Args:
            instance: Instance to explain (1D array)
            model: Fitted model
            num_features: Number of features to include in explanation
            top_labels: Number of top labels to explain
            
        Returns:
            Dictionary with explanation, feature importance, prediction
        """
        if not self.is_fitted:
            raise ValueError("Must fit before explain_instance")
        
        if not LIME_AVAILABLE:
            return {'error': 'lime library not available'}
        
        instance = np.asarray(instance)
        
        try:
            # Get explanation
            explanation = self.explainer_.explain_instance(
                instance,
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                num_features=num_features,
                top_labels=top_labels
            )
            
            # Extract explanation data
            explanation_list = explanation.as_list()
            
            # Parse explanation
            feature_importance = {}
            for feature, importance in explanation_list:
                feature_importance[feature] = importance
            
            # Get prediction
            if hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(instance.reshape(1, -1))[0]
            else:
                prediction = model.predict(instance.reshape(1, -1))[0]
            
            return {
                'explanation': explanation_list,
                'feature_importance': feature_importance,
                'prediction': prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
                'num_features': num_features,
                'explanation_object': explanation
            }
        except Exception as e:
            return {'error': f'LIME explanation failed: {str(e)}'}
    
    def explain_multiple(
        self,
        X: np.ndarray,
        model: Any,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain multiple instances
        
        Args:
            X: Instances to explain
            model: Fitted model
            num_features: Number of features to include
            
        Returns:
            Dictionary with explanations for all instances
        """
        X = np.asarray(X)
        explanations = []
        
        for i, instance in enumerate(X):
            explanation = self.explain_instance(instance, model, num_features)
            if 'error' not in explanation:
                explanations.append(explanation)
        
        # Aggregate feature importance across instances
        all_importances = defaultdict(list)
        for exp in explanations:
            for feature, importance in exp.get('feature_importance', {}).items():
                all_importances[feature].append(abs(importance))
        
        # Average importance
        avg_importance = {feat: np.mean(imps) for feat, imps in all_importances.items()}
        
        return {
            'explanations': explanations,
            'average_feature_importance': avg_importance,
            'n_instances': len(explanations),
            'feature_rankings': sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        }
    
    def plot_explanation(
        self,
        explanation: Any,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot LIME explanation
        
        Args:
            explanation: Explanation object from explain_instance
            save_path: Path to save figure
            show: Whether to display plot
        """
        if not LIME_AVAILABLE:
            print("Warning: lime library not available for plotting")
            return
        
        try:
            if isinstance(explanation, dict) and 'explanation_object' in explanation:
                explanation = explanation['explanation_object']
            
            explanation.show_in_notebook(show_table=True)
            
            if save_path:
                # LIME doesn't have direct save, would need to use matplotlib
                import matplotlib.pyplot as plt
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to plot LIME explanation: {e}")
