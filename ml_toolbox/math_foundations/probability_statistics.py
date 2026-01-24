"""
Probability & Statistics

Implements:
- Probability Distributions
- Bayesian Inference
- Maximum Likelihood Estimation (MLE)
- Expectation, Variance, Covariance
"""
import numpy as np
from typing import Callable, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)

# Try to import scipy, fallback to numpy
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Some features may be limited.")


class ProbabilityDistribution:
    """Base class for probability distributions"""
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function"""
        raise NotImplementedError
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function"""
        raise NotImplementedError
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from distribution"""
        raise NotImplementedError


class Gaussian(ProbabilityDistribution):
    """
    Gaussian (Normal) Distribution
    """
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """
        Initialize Gaussian distribution
        
        Parameters
        ----------
        mean : float
            Mean
        std : float
            Standard deviation
        """
        self.mean = mean
        self.std = std
        self.var = std ** 2
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function"""
        return (1 / (self.std * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((x - self.mean) / self.std) ** 2)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function"""
        if SCIPY_AVAILABLE:
            return stats.norm.cdf(x, self.mean, self.std)
        else:
            # Approximation using error function
            z = (x - self.mean) / self.std
            return 0.5 * (1 + np.sign(z) * (1 - np.exp(-2 * z ** 2 / np.pi)))
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from distribution"""
        return np.random.normal(self.mean, self.std, n)


class BayesianInference:
    """
    Bayesian Inference
    
    Update prior beliefs with evidence
    """
    
    @staticmethod
    def bayes_rule(prior: float, likelihood: float, evidence: float) -> float:
        """
        Bayes' rule
        
        P(A|B) = P(B|A) * P(A) / P(B)
        
        Parameters
        ----------
        prior : float
            Prior probability P(A)
        likelihood : float
            Likelihood P(B|A)
        evidence : float
            Evidence P(B)
            
        Returns
        -------
        posterior : float
            Posterior probability P(A|B)
        """
        if evidence == 0:
            return 0.0
        return (likelihood * prior) / evidence
    
    @staticmethod
    def update_posterior(prior_mean: float, prior_var: float,
                        data_mean: float, data_var: float,
                        n_samples: int) -> Tuple[float, float]:
        """
        Update posterior for Gaussian (conjugate prior)
        
        Parameters
        ----------
        prior_mean : float
            Prior mean
        prior_var : float
            Prior variance
        data_mean : float
            Data mean
        data_var : float
            Data variance
        n_samples : int
            Number of samples
            
        Returns
        -------
        posterior_mean : float
            Posterior mean
        posterior_var : float
            Posterior variance
        """
        # Conjugate update for Gaussian
        prior_precision = 1 / prior_var
        data_precision = n_samples / data_var
        
        posterior_precision = prior_precision + data_precision
        posterior_var = 1 / posterior_precision
        
        posterior_mean = (prior_precision * prior_mean + data_precision * data_mean) / \
                         posterior_precision
        
        return posterior_mean, posterior_var


class MLE:
    """
    Maximum Likelihood Estimation
    """
    
    @staticmethod
    def gaussian_mle(data: np.ndarray) -> Tuple[float, float]:
        """
        MLE for Gaussian distribution
        
        Parameters
        ----------
        data : array
            Data samples
            
        Returns
        -------
        mean : float
            MLE mean
        std : float
            MLE standard deviation
        """
        mean = np.mean(data)
        std = np.std(data, ddof=0)  # MLE uses N, not N-1
        return mean, std
    
    @staticmethod
    def bernoulli_mle(data: np.ndarray) -> float:
        """
        MLE for Bernoulli distribution
        
        Parameters
        ----------
        data : array
            Binary data (0 or 1)
            
        Returns
        -------
        p : float
            MLE probability
        """
        return np.mean(data)
    
    @staticmethod
    def poisson_mle(data: np.ndarray) -> float:
        """
        MLE for Poisson distribution
        
        Parameters
        ----------
        data : array
            Count data
            
        Returns
        -------
        lambda : float
            MLE rate parameter
        """
        return np.mean(data)


def expectation(distribution: ProbabilityDistribution, 
               function: Optional[Callable] = None) -> float:
    """
    Expectation (expected value)
    
    E[f(X)] = ∫ f(x) * p(x) dx
    
    Parameters
    ----------
    distribution : ProbabilityDistribution
        Distribution
    function : callable, optional
        Function to apply (default: identity)
        
    Returns
    -------
    expectation : float
        Expected value
    """
    if function is None:
        function = lambda x: x
    
    # Monte Carlo estimation
    samples = distribution.sample(10000)
    return np.mean([function(s) for s in samples])


def variance(distribution: ProbabilityDistribution) -> float:
    """
    Variance
    
    Var(X) = E[X²] - E[X]²
    
    Parameters
    ----------
    distribution : ProbabilityDistribution
        Distribution
        
    Returns
    -------
    variance : float
        Variance
    """
    if isinstance(distribution, Gaussian):
        return distribution.var
    
    # Monte Carlo estimation
    samples = distribution.sample(10000)
    return np.var(samples)


def covariance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Covariance
    
    Cov(X, Y) = E[XY] - E[X]E[Y]
    
    Parameters
    ----------
    X : array
        First variable
    Y : array
        Second variable
        
    Returns
    -------
    covariance : float
        Covariance
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    return np.mean((X - np.mean(X)) * (Y - np.mean(Y)))
