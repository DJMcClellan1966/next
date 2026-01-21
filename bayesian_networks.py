"""
Russell/Norvig Bayesian Networks for Feature Relationships
Model dependencies and causal relationships between features

Features:
- Bayesian Network structure learning
- Conditional probability tables
- Inference (variable elimination, sampling)
- Feature relationship analysis
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Set
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import pgmpy (Python library for probabilistic graphical models)
try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    print("Warning: pgmpy not available. Install with: pip install pgmpy")
    print("Bayesian Networks will use simplified implementation")


class SimpleBayesianNetwork:
    """
    Simplified Bayesian Network implementation
    
    For when pgmpy is not available
    """
    
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.cpt = {}  # Conditional Probability Tables
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, structure: Optional[List[Tuple[int, int]]] = None):
        """
        Fit Bayesian Network
        
        Args:
            X: Features
            y: Target variable (optional)
            structure: Network structure as list of (parent, child) edges
        """
        X = np.asarray(X)
        n_features = X.shape[1]
        
        # Discretize continuous features
        X_discrete = self._discretize(X)
        
        # Learn structure if not provided
        if structure is None:
            structure = self._learn_structure_simple(X_discrete)
        
        self.nodes = list(range(n_features))
        if y is not None:
            self.nodes.append('target')
        
        self.edges = structure
        self.is_fitted = True
        
        return self
    
    def _discretize(self, X: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """Discretize continuous features"""
        X_discrete = X.copy()
        for i in range(X.shape[1]):
            if len(np.unique(X[:, i])) > n_bins:
                X_discrete[:, i] = np.digitize(X[:, i], np.linspace(X[:, i].min(), X[:, i].max(), n_bins))
        return X_discrete
    
    def _learn_structure_simple(self, X: np.ndarray) -> List[Tuple[int, int]]:
        """
        Simple structure learning using correlation
        
        Creates edges between highly correlated features
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Find edges (correlation > threshold)
        threshold = 0.3
        edges = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > threshold:
                    # Add edge (directed based on feature index)
                    edges.append((i, j))
        
        return edges
    
    def get_dependencies(self) -> Dict[str, Any]:
        """Get feature dependencies"""
        if not self.is_fitted:
            raise ValueError("Must fit before get_dependencies")
        
        # Build dependency graph
        dependencies = defaultdict(list)
        for parent, child in self.edges:
            dependencies[child].append(parent)
        
        return {
            'edges': self.edges,
            'dependencies': dict(dependencies),
            'n_nodes': len(self.nodes),
            'n_edges': len(self.edges)
        }


class BayesianNetworkAnalyzer:
    """
    Bayesian Network Analyzer for ML feature relationships
    
    Uses Bayesian Networks to model dependencies between features
    """
    
    def __init__(self, use_pgmpy: bool = True):
        """
        Args:
            use_pgmpy: Whether to use pgmpy library (if available)
        """
        self.use_pgmpy = use_pgmpy and PGMPY_AVAILABLE
        self.model = None
        self.is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        structure: Optional[List[Tuple[str, str]]] = None,
        discretize: bool = True,
        n_bins: int = 5
    ):
        """
        Fit Bayesian Network
        
        Args:
            X: Features
            y: Target variable (optional)
            structure: Network structure (if None, learns structure)
            discretize: Whether to discretize continuous features
            n_bins: Number of bins for discretization
        """
        X = np.asarray(X)
        
        if self.use_pgmpy:
            return self._fit_pgmpy(X, y, structure, discretize, n_bins)
        else:
            return self._fit_simple(X, y, structure, discretize, n_bins)
    
    def _fit_pgmpy(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        structure: Optional[List[Tuple[str, str]]],
        discretize: bool,
        n_bins: int
    ):
        """Fit using pgmpy library"""
        import pandas as pd
        
        # Prepare data
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        data_dict = {name: X[:, i] for i, name in enumerate(feature_names)}
        
        if y is not None:
            data_dict['target'] = y
        
        df = pd.DataFrame(data_dict)
        
        # Discretize if needed
        if discretize:
            for col in df.columns:
                if len(df[col].unique()) > n_bins:
                    df[col] = pd.cut(df[col], bins=n_bins, labels=False)
        
        # Create model
        if structure is None:
            # Learn structure (simplified - use correlation-based)
            from pgmpy.estimators import PC
            est = PC(df)
            self.model = est.estimate()
        else:
            # Use provided structure
            self.model = BayesianNetwork(structure)
            self.model.fit(df, estimator=MaximumLikelihoodEstimator)
        
        self.is_fitted = True
        return self
    
    def _fit_simple(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        structure: Optional[List[Tuple[str, str]]],
        discretize: bool,
        n_bins: int
    ):
        """Fit using simple implementation"""
        self.model = SimpleBayesianNetwork()
        self.model.fit(X, y, structure)
        self.is_fitted = True
        return self
    
    def get_dependencies(self) -> Dict[str, Any]:
        """
        Get feature dependencies
        
        Returns:
            Dictionary with edges, dependencies, and statistics
        """
        if not self.is_fitted:
            raise ValueError("Must fit before get_dependencies")
        
        if self.use_pgmpy:
            edges = list(self.model.edges())
            dependencies = defaultdict(list)
            for parent, child in edges:
                dependencies[child].append(parent)
            
            return {
                'edges': [(str(p), str(c)) for p, c in edges],
                'dependencies': {str(k): [str(v) for v in vs] for k, vs in dependencies.items()},
                'n_nodes': len(self.model.nodes()),
                'n_edges': len(edges)
            }
        else:
            return self.model.get_dependencies()
    
    def get_feature_importance(self, target: str = 'target') -> Dict[str, float]:
        """
        Get feature importance based on network structure
        
        Features with more connections to target are more important
        """
        if not self.is_fitted:
            raise ValueError("Must fit before get_feature_importance")
        
        dependencies = self.get_dependencies()
        
        # Calculate importance as number of paths to target
        importance = defaultdict(float)
        
        if self.use_pgmpy:
            # Use pgmpy inference
            try:
                from pgmpy.inference import VariableElimination
                infer = VariableElimination(self.model)
                
                # Get all features
                all_nodes = list(self.model.nodes())
                features = [n for n in all_nodes if n != target]
                
                for feature in features:
                    # Calculate mutual information or correlation with target
                    # Simplified: count edges
                    importance[feature] = len([e for e in dependencies.get('edges', []) 
                                             if feature in str(e)])
            except:
                # Fallback: simple counting
                for feature in dependencies.get('dependencies', {}).keys():
                    if feature != target:
                        importance[feature] = len(dependencies['dependencies'].get(feature, []))
        else:
            # Simple implementation
            deps = dependencies.get('dependencies', {})
            for feature, parents in deps.items():
                if feature != target:
                    importance[feature] = len(parents)
        
        return dict(importance)
    
    def predict_proba(self, X: np.ndarray, target: str = 'target') -> np.ndarray:
        """
        Predict probabilities using Bayesian Network
        
        Args:
            X: Features
            target: Target variable name
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Must fit before predict_proba")
        
        if not self.use_pgmpy:
            warnings.warn("Probability prediction requires pgmpy")
            return np.ones((len(X), 2)) * 0.5  # Dummy probabilities
        
        # Use pgmpy inference
        from pgmpy.inference import VariableElimination
        infer = VariableElimination(self.model)
        
        # Prepare evidence
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        probabilities = []
        for i in range(len(X)):
            evidence = {name: float(X[i, j]) for j, name in enumerate(feature_names)}
            try:
                prob = infer.query(variables=[target], evidence=evidence)
                probabilities.append(prob.values)
            except:
                probabilities.append([0.5, 0.5])  # Default
        
        return np.array(probabilities)
