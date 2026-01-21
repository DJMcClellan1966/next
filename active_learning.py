"""
Active Learning
Select most informative samples for labeling

Methods:
- Uncertainty sampling (least confident, margin, entropy)
- Query-by-committee
- Expected model change
- Diversity sampling
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class ActiveLearner:
    """
    Active Learning: Select most informative samples for labeling
    
    Reduces labeling costs by selecting samples that provide most information
    """
    
    def __init__(
        self,
        model: Any,
        method: str = 'uncertainty',
        uncertainty_method: str = 'entropy',
        random_state: int = 42
    ):
        """
        Args:
            model: Base model for uncertainty estimation
            method: 'uncertainty', 'query_by_committee', 'expected_model_change'
            uncertainty_method: 'least_confident', 'margin', 'entropy'
            random_state: Random seed
        """
        self.model = model
        self.method = method
        self.uncertainty_method = uncertainty_method
        self.random_state = random_state
        np.random.seed(random_state)
    
    def _uncertainty_least_confident(self, proba: np.ndarray) -> np.ndarray:
        """
        Least confident uncertainty
        
        Uncertainty = 1 - max(P(y|x))
        """
        max_proba = np.max(proba, axis=1)
        return 1 - max_proba
    
    def _uncertainty_margin(self, proba: np.ndarray) -> np.ndarray:
        """
        Margin-based uncertainty
        
        Uncertainty = P(y1|x) - P(y2|x) where y1 and y2 are top 2 classes
        """
        sorted_proba = np.sort(proba, axis=1)[:, ::-1]
        if sorted_proba.shape[1] >= 2:
            margin = sorted_proba[:, 0] - sorted_proba[:, 1]
        else:
            margin = sorted_proba[:, 0]
        return 1 - margin  # Lower margin = higher uncertainty
    
    def _uncertainty_entropy(self, proba: np.ndarray) -> np.ndarray:
        """
        Entropy-based uncertainty
        
        Uncertainty = -sum(P(y|x) * log(P(y|x)))
        """
        # Avoid log(0)
        proba = proba + 1e-10
        proba = proba / proba.sum(axis=1, keepdims=True)
        
        entropy = -np.sum(proba * np.log2(proba), axis=1)
        return entropy
    
    def _calculate_uncertainty(self, proba: np.ndarray) -> np.ndarray:
        """Calculate uncertainty based on method"""
        if self.uncertainty_method == 'least_confident':
            return self._uncertainty_least_confident(proba)
        elif self.uncertainty_method == 'margin':
            return self._uncertainty_margin(proba)
        elif self.uncertainty_method == 'entropy':
            return self._uncertainty_entropy(proba)
        else:
            return self._uncertainty_entropy(proba)  # Default
    
    def select_samples(
        self,
        X_unlabeled: np.ndarray,
        n_samples: int = 10,
        X_labeled: Optional[np.ndarray] = None,
        y_labeled: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Select most informative samples for labeling
        
        Args:
            X_unlabeled: Unlabeled data
            n_samples: Number of samples to select
            X_labeled: Already labeled data (optional, for retraining)
            y_labeled: Labels for labeled data (optional)
            
        Returns:
            Dictionary with selected indices, uncertainty scores
        """
        X_unlabeled = np.asarray(X_unlabeled)
        
        # Retrain model if labeled data provided
        if X_labeled is not None and y_labeled is not None:
            try:
                model_copy = self._clone_model(self.model)
                model_copy.fit(X_labeled, y_labeled)
                model_to_use = model_copy
            except:
                model_to_use = self.model
        else:
            model_to_use = self.model
        
        if self.method == 'uncertainty':
            return self._uncertainty_sampling(X_unlabeled, n_samples, model_to_use)
        elif self.method == 'query_by_committee':
            return self._query_by_committee(X_unlabeled, n_samples)
        elif self.method == 'expected_model_change':
            return self._expected_model_change(X_unlabeled, n_samples, model_to_use)
        else:
            return self._uncertainty_sampling(X_unlabeled, n_samples, model_to_use)
    
    def _uncertainty_sampling(
        self,
        X_unlabeled: np.ndarray,
        n_samples: int,
        model: Any
    ) -> Dict[str, Any]:
        """Uncertainty sampling"""
        # Get predictions
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_unlabeled)
        else:
            # For regression, use prediction variance as uncertainty
            predictions = model.predict(X_unlabeled)
            # Simple uncertainty: distance from mean
            uncertainty = np.abs(predictions - np.mean(predictions))
            selected_indices = np.argsort(uncertainty)[-n_samples:][::-1]
            return {
                'selected_indices': selected_indices.tolist(),
                'uncertainty_scores': uncertainty[selected_indices].tolist(),
                'method': 'uncertainty'
            }
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(proba)
        
        # Select samples with highest uncertainty
        selected_indices = np.argsort(uncertainty)[-n_samples:][::-1]
        
        return {
            'selected_indices': selected_indices.tolist(),
            'uncertainty_scores': uncertainty[selected_indices].tolist(),
            'method': 'uncertainty',
            'uncertainty_method': self.uncertainty_method
        }
    
    def _query_by_committee(
        self,
        X_unlabeled: np.ndarray,
        n_samples: int
    ) -> Dict[str, Any]:
        """
        Query-by-committee
        
        Uses multiple models and selects samples with highest disagreement
        """
        # For simplicity, create committee by training on bootstrap samples
        # In practice, would use different model types or training sets
        
        warnings.warn("Query-by-committee requires multiple models. Using uncertainty sampling as fallback.")
        return self._uncertainty_sampling(X_unlabeled, n_samples, self.model)
    
    def _expected_model_change(
        self,
        X_unlabeled: np.ndarray,
        n_samples: int,
        model: Any
    ) -> Dict[str, Any]:
        """
        Expected model change
        
        Selects samples that would change model most if labeled
        """
        # Simplified: use uncertainty as proxy
        warnings.warn("Expected model change requires gradient computation. Using uncertainty sampling as fallback.")
        return self._uncertainty_sampling(X_unlabeled, n_samples, model)
    
    def _clone_model(self, model: Any) -> Any:
        """Clone model"""
        if hasattr(model, 'get_params'):
            from sklearn.base import clone
            return clone(model)
        else:
            import copy
            return copy.deepcopy(model)
    
    def iterative_labeling(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        n_iterations: int = 10,
        n_samples_per_iteration: int = 5
    ) -> Dict[str, Any]:
        """
        Iterative active learning
        
        Repeatedly selects samples, simulates labeling, retrains model
        
        Args:
            X_unlabeled: Unlabeled data
            X_labeled: Initially labeled data
            y_labeled: Initial labels
            n_iterations: Number of active learning iterations
            n_samples_per_iteration: Samples to select per iteration
            
        Returns:
            Dictionary with selection history, final model performance
        """
        X_unlabeled = np.asarray(X_unlabeled).copy()
        X_labeled = np.asarray(X_labeled).copy()
        y_labeled = np.asarray(y_labeled).copy()
        
        selection_history = []
        
        for iteration in range(n_iterations):
            if len(X_unlabeled) == 0:
                break
            
            # Retrain model
            model_copy = self._clone_model(self.model)
            model_copy.fit(X_labeled, y_labeled)
            
            # Select samples
            selection_result = self.select_samples(
                X_unlabeled,
                n_samples=n_samples_per_iteration,
                X_labeled=X_labeled,
                y_labeled=y_labeled
            )
            
            selected_indices = selection_result['selected_indices']
            
            # Simulate labeling (in practice, would get labels from human)
            # For demo, use model predictions as "true" labels
            selected_X = X_unlabeled[selected_indices]
            if hasattr(model_copy, 'predict'):
                simulated_labels = model_copy.predict(selected_X)
            else:
                simulated_labels = np.zeros(len(selected_X))
            
            # Add to labeled set
            X_labeled = np.vstack([X_labeled, selected_X])
            y_labeled = np.hstack([y_labeled, simulated_labels])
            
            # Remove from unlabeled set
            X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
            
            selection_history.append({
                'iteration': iteration,
                'selected_indices': selected_indices,
                'uncertainty_scores': selection_result.get('uncertainty_scores', []),
                'n_labeled': len(X_labeled),
                'n_unlabeled': len(X_unlabeled)
            })
        
        return {
            'selection_history': selection_history,
            'final_n_labeled': len(X_labeled),
            'final_n_unlabeled': len(X_unlabeled),
            'n_iterations': len(selection_history)
        }
