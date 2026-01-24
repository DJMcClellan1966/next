"""
Probabilistic Machine Learning (PRML - Bishop)

Implements:
- EM Algorithm
- Variational Inference
- Bayesian Learning
- Graphical Models
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EMAlgorithm:
    """
    Expectation-Maximization Algorithm
    
    For latent variable models
    """
    
    def __init__(self, n_components: int = 2, max_iter: int = 100, tol: float = 1e-6):
        """
        Initialize EM
        
        Parameters
        ----------
        n_components : int
            Number of components
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
    
    def fit(self, X: np.ndarray):
        """
        Fit EM algorithm (Gaussian Mixture Model)
        
        Parameters
        ----------
        X : array
            Training data
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities
            responsibilities = self._e_step(X)
            
            # M-step: Update parameters
            self._m_step(X, responsibilities)
            
            # Check convergence
            log_likelihood = self._log_likelihood(X)
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            
            prev_log_likelihood = log_likelihood
        
        logger.info(f"[EMAlgorithm] Converged in {iteration + 1} iterations")
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """E-step: Compute responsibilities"""
        n_samples = len(X)
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Gaussian probability
            diff = X - self.means_[k]
            inv_cov = np.linalg.inv(self.covariances_[k])
            exp_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            det = np.linalg.det(self.covariances_[k])
            prob = np.exp(exp_term) / np.sqrt((2 * np.pi) ** X.shape[1] * det)
            
            responsibilities[:, k] = self.weights_[k] * prob
        
        # Normalize
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """M-step: Update parameters"""
        n_samples = len(X)
        
        for k in range(self.n_components):
            # Update weights
            self.weights_[k] = np.mean(responsibilities[:, k])
            
            # Update means
            self.means_[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / \
                            np.sum(responsibilities[:, k])
            
            # Update covariances
            diff = X - self.means_[k]
            self.covariances_[k] = np.sum(
                responsibilities[:, k:k+1] * diff[:, :, np.newaxis] * diff[:, np.newaxis, :],
                axis=0
            ) / np.sum(responsibilities[:, k])
    
    def _log_likelihood(self, X: np.ndarray) -> float:
        """Compute log-likelihood"""
        likelihood = 0.0
        
        for k in range(self.n_components):
            diff = X - self.means_[k]
            inv_cov = np.linalg.inv(self.covariances_[k])
            exp_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            det = np.linalg.det(self.covariances_[k])
            prob = np.exp(exp_term) / np.sqrt((2 * np.pi) ** X.shape[1] * det)
            
            likelihood += self.weights_[k] * prob
        
        return np.sum(np.log(likelihood + 1e-10))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict component assignments"""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)


class VariationalInference:
    """
    Variational Inference
    
    Approximate Bayesian inference
    """
    
    @staticmethod
    def elbo(q_params: Dict[str, np.ndarray], p_params: Dict[str, np.ndarray],
            data: np.ndarray) -> float:
        """
        Evidence Lower BOund (ELBO)
        
        ELBO = E_q[log p(x|z)] - KL(q(z) || p(z))
        
        Parameters
        ----------
        q_params : dict
            Variational distribution parameters
        p_params : dict
            Prior distribution parameters
        data : array
            Observed data
            
        Returns
        -------
        elbo : float
            ELBO value
        """
        # Simplified ELBO computation
        # In practice, more sophisticated computation needed
        
        # Reconstruction term (simplified)
        recon_term = -np.mean((data - q_params.get('mean', 0)) ** 2)
        
        # KL divergence term
        q_mean = q_params.get('mean', 0)
        q_var = q_params.get('var', 1)
        p_mean = p_params.get('mean', 0)
        p_var = p_params.get('var', 1)
        
        kl_term = 0.5 * (np.log(p_var / q_var) + (q_var + (q_mean - p_mean) ** 2) / p_var - 1)
        
        return recon_term - kl_term
    
    @staticmethod
    def update_variational_params(q_params: Dict, p_params: Dict, data: np.ndarray,
                                 learning_rate: float = 0.01) -> Dict:
        """Update variational parameters"""
        # Simplified update (gradient ascent on ELBO)
        # In practice, use reparameterization trick
        
        new_params = q_params.copy()
        
        # Update mean
        if 'mean' in new_params:
            new_params['mean'] += learning_rate * (data.mean() - new_params['mean'])
        
        # Update variance
        if 'var' in new_params:
            new_params['var'] = np.maximum(new_params['var'] * 0.99, 0.01)
        
        return new_params


class BayesianLearning:
    """
    Bayesian Learning
    
    Learning with uncertainty
    """
    
    @staticmethod
    def bayesian_linear_regression(X: np.ndarray, y: np.ndarray,
                                  prior_precision: float = 1.0,
                                  noise_precision: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Bayesian Linear Regression
        
        Parameters
        ----------
        X : array
            Training data
        y : array
            Target values
        prior_precision : float
            Prior precision (inverse variance)
        noise_precision : float
            Noise precision
            
        Returns
        -------
        posterior : dict
            Posterior distribution parameters
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        # Posterior precision
        posterior_precision = prior_precision * np.eye(X.shape[1]) + \
                            noise_precision * X.T @ X
        
        # Posterior mean
        posterior_mean = np.linalg.solve(posterior_precision,
                                       noise_precision * X.T @ y)
        
        # Posterior covariance
        posterior_cov = np.linalg.inv(posterior_precision)
        
        return {
            'mean': posterior_mean,
            'covariance': posterior_cov,
            'precision': posterior_precision
        }
    
    @staticmethod
    def predictive_distribution(X_new: np.ndarray, posterior: Dict[str, np.ndarray],
                               noise_precision: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Predictive distribution
        
        Parameters
        ----------
        X_new : array
            New data points
        posterior : dict
            Posterior parameters
        noise_precision : float
            Noise precision
            
        Returns
        -------
        predictive : dict
            Predictive distribution parameters
        """
        X_new = np.asarray(X_new)
        
        # Predictive mean
        pred_mean = X_new @ posterior['mean']
        
        # Predictive variance
        pred_var = 1 / noise_precision + np.diag(X_new @ posterior['covariance'] @ X_new.T)
        
        return {
            'mean': pred_mean,
            'variance': pred_var
        }


class GraphicalModels:
    """
    Graphical Models
    
    Probabilistic graphical models
    """
    
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str]] = []
    
    def add_node(self, node: str, distribution: str = 'gaussian',
                parameters: Dict = None):
        """Add node to graphical model"""
        self.nodes[node] = {
            'distribution': distribution,
            'parameters': parameters or {}
        }
    
    def add_edge(self, parent: str, child: str):
        """Add edge (parent -> child)"""
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Nodes must exist before adding edge")
        
        self.edges.append((parent, child))
    
    def get_parents(self, node: str) -> List[str]:
        """Get parent nodes"""
        return [p for p, c in self.edges if c == node]
    
    def get_children(self, node: str) -> List[str]:
        """Get child nodes"""
        return [c for p, c in self.edges if p == node]
    
    def factorize(self) -> List[Tuple]:
        """
        Factorize joint distribution
        
        P(X1, X2, ..., Xn) = Π P(Xi | parents(Xi))
        
        Returns
        -------
        factors : list
            List of (node, parents) tuples
        """
        factors = []
        for node in self.nodes:
            parents = self.get_parents(node)
            factors.append((node, parents))
        return factors
