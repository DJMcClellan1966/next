"""
Causal Discovery
Learn causal relationships from observational data

Methods:
- PC Algorithm (constraint-based)
- GES Algorithm (score-based)
- Causal graph learning
- D-separation testing
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Set
import numpy as np
from collections import defaultdict
import warnings
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

# Try to import pgmpy
try:
    from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
    from pgmpy.models import BayesianNetwork
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    print("Warning: pgmpy not available. Install with: pip install pgmpy")
    print("Causal discovery will use simplified implementation")


class CausalDiscovery:
    """
    Causal discovery from observational data
    
    Learns causal graph structure
    """
    
    def __init__(
        self,
        method: str = 'pc',
        alpha: float = 0.05,
        random_state: int = 42
    ):
        """
        Args:
            method: 'pc' (constraint-based) or 'ges' (score-based)
            alpha: Significance level for independence tests
            random_state: Random seed
        """
        self.method = method
        self.alpha = alpha
        self.random_state = random_state
        np.random.seed(random_state)
        self.causal_graph_ = None
        self.is_fitted = False
    
    def _independence_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None
    ) -> Tuple[float, bool]:
        """
        Test conditional independence X ⟂ Y | Z
        
        Args:
            X: Variable 1
            Y: Variable 2
            Z: Conditioning set (optional)
            
        Returns:
            (p_value, is_independent)
        """
        X = np.asarray(X).ravel()
        Y = np.asarray(Y).ravel()
        
        if Z is None or len(Z) == 0:
            # Unconditional independence test
            # Use correlation test
            if len(np.unique(X)) < 10 and len(np.unique(Y)) < 10:
                # Categorical: chi-square test
                from scipy.stats import chi2_contingency
                contingency = np.histogram2d(X, Y, bins=[len(np.unique(X)), len(np.unique(Y))])[0]
                chi2, p_value, _, _ = chi2_contingency(contingency)
                return p_value, p_value > self.alpha
            else:
                # Continuous: correlation test
                corr, p_value = stats.pearsonr(X, Y)
                return p_value, p_value > self.alpha
        else:
            # Conditional independence (simplified: partial correlation)
            # In practice, would use more sophisticated tests
            from scipy.stats import pearsonr
            
            # Partial correlation approximation
            # Residualize X and Y on Z
            if len(Z.shape) == 1:
                Z = Z.reshape(-1, 1)
            
            # Simple linear regression residuals
            from sklearn.linear_model import LinearRegression
            
            # Residualize X
            lr_x = LinearRegression()
            lr_x.fit(Z, X)
            X_residual = X - lr_x.predict(Z)
            
            # Residualize Y
            lr_y = LinearRegression()
            lr_y.fit(Z, Y)
            Y_residual = Y - lr_y.predict(Z)
            
            # Test correlation of residuals
            _, p_value = pearsonr(X_residual, Y_residual)
            return p_value, p_value > self.alpha
    
    def pc_algorithm(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        PC Algorithm (constraint-based causal discovery)
        
        Args:
            X: Data (n_samples, n_features)
            feature_names: Feature names
            
        Returns:
            Dictionary with causal graph, edges, structure
        """
        X = np.asarray(X)
        n_features = X.shape[1]
        
        if feature_names is None:
            feature_names = [f'X{i}' for i in range(n_features)]
        
        if PGMPY_AVAILABLE:
            return self._pc_algorithm_pgmpy(X, feature_names)
        else:
            return self._pc_algorithm_simple(X, feature_names)
    
    def _pc_algorithm_pgmpy(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """PC algorithm using pgmpy"""
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        
        # Run PC algorithm
        est = PC(df)
        self.causal_graph_ = est.estimate()
        
        edges = list(self.causal_graph_.edges())
        
        return {
            'edges': [(str(u), str(v)) for u, v in edges],
            'nodes': list(self.causal_graph_.nodes()),
            'n_edges': len(edges),
            'n_nodes': len(self.causal_graph_.nodes()),
            'method': 'pc_pgmpy',
            'graph': self.causal_graph_
        }
    
    def _pc_algorithm_simple(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Simplified PC algorithm implementation"""
        n_features = X.shape[1]
        
        # Initialize fully connected graph
        edges = set()
        for i in range(n_features):
            for j in range(i + 1, n_features):
                edges.add((i, j))
        
        # Phase 1: Remove edges based on unconditional independence
        edges_to_remove = set()
        for i, j in edges:
            p_value, is_independent = self._independence_test(X[:, i], X[:, j])
            if is_independent:
                edges_to_remove.add((i, j))
        
        edges = edges - edges_to_remove
        
        # Phase 2: Remove edges based on conditional independence (simplified)
        # In full PC, would test all conditioning sets of increasing size
        # Here, just test with one conditioning variable
        
        edges_to_remove_2 = set()
        for i, j in list(edges):
            # Try conditioning on each other variable
            for k in range(n_features):
                if k != i and k != j:
                    p_value, is_independent = self._independence_test(
                        X[:, i], X[:, j], X[:, k]
                    )
                    if is_independent:
                        edges_to_remove_2.add((i, j))
                        break
        
        edges = edges - edges_to_remove_2
        
        # Convert to directed edges (simplified: use feature order)
        directed_edges = []
        for i, j in edges:
            # Simple heuristic: earlier feature causes later feature
            if i < j:
                directed_edges.append((feature_names[i], feature_names[j]))
            else:
                directed_edges.append((feature_names[j], feature_names[i]))
        
        return {
            'edges': directed_edges,
            'nodes': feature_names,
            'n_edges': len(directed_edges),
            'n_nodes': n_features,
            'method': 'pc_simple'
        }
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Learn causal graph
        
        Args:
            X: Data
            feature_names: Feature names
            
        Returns:
            self
        """
        if self.method == 'pc':
            result = self.pc_algorithm(X, feature_names)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def get_causal_graph(self) -> Dict[str, Any]:
        """Get learned causal graph"""
        if not self.is_fitted:
            raise ValueError("Must fit before get_causal_graph")
        
        if self.causal_graph_ is not None:
            edges = list(self.causal_graph_.edges())
            return {
                'edges': [(str(u), str(v)) for u, v in edges],
                'nodes': list(self.causal_graph_.nodes()),
                'n_edges': len(edges),
                'n_nodes': len(self.causal_graph_.nodes())
            }
        else:
            return {'error': 'Causal graph not available'}
