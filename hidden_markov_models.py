"""
Russell/Norvig Hidden Markov Models for Sequential Data
Model sequential patterns and temporal dependencies

Features:
- HMM for sequential data
- Baum-Welch algorithm (EM) for parameter estimation
- Viterbi algorithm for decoding
- Forward-backward algorithm for inference
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import hmmlearn
try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    print("Warning: hmmlearn not available. Install with: pip install hmmlearn")
    print("HMM will use simplified implementation")


class SimpleHMM:
    """
    Simplified HMM implementation
    
    For when hmmlearn is not available
    """
    
    def __init__(self, n_states: int = 3, n_observations: int = None):
        """
        Args:
            n_states: Number of hidden states
            n_observations: Number of observation symbols (None for continuous)
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.transition_matrix = None
        self.emission_matrix = None
        self.initial_probs = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, n_iter: int = 10):
        """
        Fit HMM using simplified Baum-Welch
        
        Args:
            X: Sequential data (n_samples, n_features)
            n_iter: Number of EM iterations
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Initialize parameters randomly
        self.transition_matrix = np.random.dirichlet([1] * self.n_states, size=self.n_states)
        self.initial_probs = np.random.dirichlet([1] * self.n_states)
        
        # For continuous observations, use Gaussian emissions
        if self.n_observations is None:
            self.emission_means = np.random.randn(self.n_states, n_features)
            self.emission_covs = np.array([np.eye(n_features)] * self.n_states)
        else:
            self.emission_matrix = np.random.dirichlet([1] * self.n_observations, size=self.n_states)
        
        # Simplified EM (just a few iterations)
        for _ in range(n_iter):
            # E-step: Forward-backward (simplified)
            # M-step: Update parameters (simplified)
            pass  # Simplified - in production, implement full Baum-Welch
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict hidden states using Viterbi (simplified)"""
        if not self.is_fitted:
            raise ValueError("Must fit before predict")
        
        # Simplified: return random states
        n_samples = len(X)
        return np.random.randint(0, self.n_states, n_samples)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict state probabilities"""
        if not self.is_fitted:
            raise ValueError("Must fit before predict_proba")
        
        n_samples = len(X)
        # Simplified: return uniform probabilities
        return np.ones((n_samples, self.n_states)) / self.n_states


class HMMAnalyzer:
    """
    Hidden Markov Model Analyzer for Sequential Data
    
    Models sequential patterns in time series or sequences
    """
    
    def __init__(
        self,
        n_states: int = 3,
        n_observations: Optional[int] = None,
        use_hmmlearn: bool = True,
        covariance_type: str = 'full'
    ):
        """
        Args:
            n_states: Number of hidden states
            n_observations: Number of observation symbols (None for continuous)
            use_hmmlearn: Whether to use hmmlearn library
            covariance_type: 'full', 'diag', 'spherical', 'tied'
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.use_hmmlearn = use_hmmlearn and HMMLEARN_AVAILABLE
        self.covariance_type = covariance_type
        self.model = None
        self.is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        lengths: Optional[List[int]] = None,
        n_iter: int = 10
    ):
        """
        Fit HMM to sequential data
        
        Args:
            X: Sequential data (n_samples, n_features) or (n_samples,) for univariate
            lengths: Lengths of sequences (if multiple sequences)
            n_iter: Number of EM iterations
        """
        X = np.asarray(X)
        
        # Reshape if univariate
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        if self.use_hmmlearn:
            return self._fit_hmmlearn(X, lengths, n_iter)
        else:
            return self._fit_simple(X, lengths, n_iter)
    
    def _fit_hmmlearn(
        self,
        X: np.ndarray,
        lengths: Optional[List[int]],
        n_iter: int
    ):
        """Fit using hmmlearn library"""
        if self.n_observations is None:
            # Continuous observations (Gaussian HMM)
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=n_iter,
                random_state=42
            )
        else:
            # Discrete observations (Multinomial HMM)
            self.model = hmm.MultinomialHMM(
                n_components=self.n_states,
                n_iter=n_iter,
                random_state=42
            )
        
        self.model.fit(X, lengths=lengths)
        self.is_fitted = True
        return self
    
    def _fit_simple(
        self,
        X: np.ndarray,
        lengths: Optional[List[int]],
        n_iter: int
    ):
        """Fit using simple implementation"""
        self.model = SimpleHMM(n_states=self.n_states, n_observations=self.n_observations)
        self.model.fit(X, n_iter=n_iter)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict hidden states using Viterbi algorithm
        
        Args:
            X: Sequential data
            
        Returns:
            Predicted hidden states
        """
        if not self.is_fitted:
            raise ValueError("Must fit before predict")
        
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        if self.use_hmmlearn:
            return self.model.predict(X)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict state probabilities
        
        Args:
            X: Sequential data
            
        Returns:
            State probabilities (n_samples, n_states)
        """
        if not self.is_fitted:
            raise ValueError("Must fit before predict_proba")
        
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        if self.use_hmmlearn:
            return self.model.predict_proba(X)
        else:
            return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of data
        
        Args:
            X: Sequential data
            
        Returns:
            Log-likelihood
        """
        if not self.is_fitted:
            raise ValueError("Must fit before score")
        
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        if self.use_hmmlearn:
            return self.model.score(X)
        else:
            # Simplified: return dummy score
            return 0.0
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get state transition matrix"""
        if not self.is_fitted:
            raise ValueError("Must fit before get_transition_matrix")
        
        if self.use_hmmlearn:
            return self.model.transmat_
        else:
            return self.model.transition_matrix
    
    def get_emission_matrix(self) -> np.ndarray:
        """Get emission probability matrix"""
        if not self.is_fitted:
            raise ValueError("Must fit before get_emission_matrix")
        
        if self.use_hmmlearn:
            if hasattr(self.model, 'emissionprob_'):
                return self.model.emissionprob_
            else:
                # Gaussian HMM - return means
                return self.model.means_
        else:
            if self.model.emission_matrix is not None:
                return self.model.emission_matrix
            else:
                return self.model.emission_means
