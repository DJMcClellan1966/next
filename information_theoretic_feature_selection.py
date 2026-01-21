"""
Information-Theoretic Feature Selection
Entropy-based feature selection methods

Methods:
- Shannon entropy for feature importance
- Conditional entropy for feature interactions
- Information gain for decision trees
- Mutual information (extend existing)
- KL divergence for feature selection
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import Counter, defaultdict
import warnings
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regr
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class InformationTheoreticFeatureSelector:
    """
    Information-theoretic feature selection using entropy measures
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def _entropy(self, values: np.ndarray) -> float:
        """
        Calculate Shannon entropy
        
        H(X) = -sum(p(x) * log2(p(x)))
        """
        value_counts = Counter(values)
        n = len(values)
        entropy = 0.0
        
        for count in value_counts.values():
            p = count / n
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _conditional_entropy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate conditional entropy H(Y|X)
        
        H(Y|X) = sum(p(x) * H(Y|X=x))
        """
        value_counts = Counter(X)
        n = len(X)
        conditional_entropy = 0.0
        
        for value, count in value_counts.items():
            p_x = count / n
            y_conditioned = y[X == value]
            if len(y_conditioned) > 0:
                h_y_given_x = self._entropy(y_conditioned)
                conditional_entropy += p_x * h_y_given_x
        
        return conditional_entropy
    
    def _mutual_information(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate mutual information I(X;Y)
        
        I(X;Y) = H(Y) - H(Y|X)
        """
        h_y = self._entropy(y)
        h_y_given_x = self._conditional_entropy(X, y)
        return h_y - h_y_given_x
    
    def _information_gain(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate information gain (same as mutual information for classification)
        
        IG(X;Y) = H(Y) - H(Y|X)
        """
        return self._mutual_information(X, y)
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Kullback-Leibler divergence
        
        KL(P||Q) = sum(p(x) * log2(p(x) / q(x)))
        """
        # Avoid division by zero
        p = p + 1e-10
        q = q + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        
        kl = np.sum(p * np.log2(p / q))
        return kl
    
    def entropy_based_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 10,
        method: str = 'mutual_information',
        discretize: bool = True,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Select features using entropy-based methods
        
        Args:
            X: Features
            y: Labels
            k: Number of features to select
            method: 'mutual_information', 'information_gain', 'entropy'
            discretize: Whether to discretize continuous features
            n_bins: Number of bins for discretization
            
        Returns:
            Dictionary with selected features, scores, rankings
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]
        
        # Discretize if needed
        if discretize:
            X_discrete = self._discretize(X, n_bins)
        else:
            X_discrete = X
        
        scores = []
        
        for i in range(n_features):
            X_feature = X_discrete[:, i]
            
            if method == 'mutual_information':
                score = self._mutual_information(X_feature, y)
            elif method == 'information_gain':
                score = self._information_gain(X_feature, y)
            elif method == 'entropy':
                score = self._entropy(X_feature)
            else:
                score = self._mutual_information(X_feature, y)
            
            scores.append(score)
        
        scores = np.array(scores)
        
        # Select top k features
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        return {
            'selected_features': top_k_indices.tolist(),
            'scores': scores.tolist(),
            'top_k_scores': scores[top_k_indices].tolist(),
            'k': k,
            'method': method,
            'feature_rankings': top_k_indices.tolist()
        }
    
    def conditional_entropy_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 10,
        discretize: bool = True,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Select features using conditional entropy
        
        Selects features that minimize H(Y|X)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]
        
        if discretize:
            X_discrete = self._discretize(X, n_bins)
        else:
            X_discrete = X
        
        # Calculate H(Y)
        h_y = self._entropy(y)
        
        # Calculate H(Y|X) for each feature
        conditional_entropies = []
        for i in range(n_features):
            h_y_given_x = self._conditional_entropy(X_discrete[:, i], y)
            conditional_entropies.append(h_y_given_x)
        
        conditional_entropies = np.array(conditional_entropies)
        
        # Select features with lowest conditional entropy (highest information gain)
        # Information gain = H(Y) - H(Y|X)
        information_gains = h_y - conditional_entropies
        
        # Select top k
        top_k_indices = np.argsort(information_gains)[-k:][::-1]
        
        return {
            'selected_features': top_k_indices.tolist(),
            'conditional_entropies': conditional_entropies.tolist(),
            'information_gains': information_gains.tolist(),
            'top_k_scores': information_gains[top_k_indices].tolist(),
            'k': k,
            'method': 'conditional_entropy'
        }
    
    def _discretize(self, X: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Discretize continuous features"""
        X_discrete = X.copy()
        for i in range(X.shape[1]):
            if len(np.unique(X[:, i])) > n_bins:
                X_discrete[:, i] = np.digitize(X[:, i], np.linspace(X[:, i].min(), X[:, i].max(), n_bins))
        return X_discrete
    
    def feature_interaction_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_pairs: Optional[List[Tuple[int, int]]] = None,
        discretize: bool = True,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze feature interactions using information theory
        
        Measures how much information two features together provide
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]
        
        if discretize:
            X_discrete = self._discretize(X, n_bins)
        else:
            X_discrete = X
        
        # Generate feature pairs if not provided
        if feature_pairs is None:
            feature_pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]
        
        interactions = []
        
        for i, j in feature_pairs:
            # Individual mutual information
            mi_i = self._mutual_information(X_discrete[:, i], y)
            mi_j = self._mutual_information(X_discrete[:, j], y)
            
            # Combined mutual information (simplified: concatenate features)
            # In practice, would need joint distribution
            combined = X_discrete[:, i] * 1000 + X_discrete[:, j]  # Simple combination
            mi_combined = self._mutual_information(combined, y)
            
            # Interaction: how much more information do we get from both?
            interaction = mi_combined - (mi_i + mi_j)
            
            interactions.append({
                'feature1': i,
                'feature2': j,
                'mi_feature1': mi_i,
                'mi_feature2': mi_j,
                'mi_combined': mi_combined,
                'interaction_strength': interaction
            })
        
        # Sort by interaction strength
        interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
        
        return {
            'interactions': interactions,
            'n_pairs': len(interactions),
            'top_interactions': interactions[:10]
        }
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 10,
        method: str = 'mutual_information',
        **kwargs
    ) -> np.ndarray:
        """
        Select features using information-theoretic methods
        
        Args:
            X: Features
            y: Labels
            k: Number of features to select
            method: Selection method
            **kwargs: Additional parameters
            
        Returns:
            Selected feature indices
        """
        if method in ['mutual_information', 'information_gain', 'entropy']:
            result = self.entropy_based_selection(X, y, k=k, method=method, **kwargs)
        elif method == 'conditional_entropy':
            result = self.conditional_entropy_selection(X, y, k=k, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return np.array(result['selected_features'])
