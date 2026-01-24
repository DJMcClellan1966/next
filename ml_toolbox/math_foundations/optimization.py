"""
Optimization Algorithms

Implements:
- Gradient Descent
- Stochastic Gradient Descent
- Adam Optimizer
- Convex Optimization
"""
import numpy as np
from typing import Callable, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def gradient_descent(f: Callable, grad_f: Callable, x0: np.ndarray,
                    learning_rate: float = 0.01, max_iter: int = 1000,
                    tol: float = 1e-6) -> Tuple[np.ndarray, list]:
    """
    Gradient Descent
    
    Parameters
    ----------
    f : callable
        Objective function
    grad_f : callable
        Gradient function
    x0 : array
        Initial point
    learning_rate : float
        Learning rate
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
        
    Returns
    -------
    x : array
        Optimal point
    history : list
        Optimization history
    """
    x = np.asarray(x0).copy()
    history = []
    
    for i in range(max_iter):
        grad = grad_f(x)
        x_new = x - learning_rate * grad
        
        history.append({
            'iteration': i,
            'x': x.copy(),
            'f(x)': f(x),
            'grad_norm': np.linalg.norm(grad)
        })
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new
    
    return x, history


def stochastic_gradient_descent(f: Callable, grad_f: Callable, data: np.ndarray,
                                x0: np.ndarray, learning_rate: float = 0.01,
                                batch_size: int = 32, max_iter: int = 1000,
                                learning_rate_decay: float = 1.0) -> Tuple[np.ndarray, list]:
    """
    Stochastic Gradient Descent
    
    Parameters
    ----------
    f : callable
        Objective function (takes data batch)
    grad_f : callable
        Gradient function (takes data batch)
    data : array
        Training data
    x0 : array
        Initial point
    learning_rate : float
        Initial learning rate
    batch_size : int
        Batch size
    max_iter : int
        Maximum iterations
    learning_rate_decay : float
        Learning rate decay factor
        
    Returns
    -------
    x : array
        Optimal point
    history : list
        Optimization history
    """
    x = np.asarray(x0).copy()
    history = []
    n_samples = len(data)
    current_lr = learning_rate
    
    for i in range(max_iter):
        # Sample batch
        batch_indices = np.random.choice(n_samples, batch_size, replace=False)
        batch = data[batch_indices]
        
        # Compute gradient on batch
        grad = grad_f(x, batch)
        
        # Update
        x = x - current_lr * grad
        
        # Decay learning rate
        current_lr *= learning_rate_decay
        
        if i % 100 == 0:
            history.append({
                'iteration': i,
                'x': x.copy(),
                'f(x)': f(x, batch),
                'learning_rate': current_lr
            })
    
    return x, history


def adam_optimizer(f: Callable, grad_f: Callable, x0: np.ndarray,
                  learning_rate: float = 0.001, beta1: float = 0.9,
                  beta2: float = 0.999, eps: float = 1e-8,
                  max_iter: int = 1000) -> Tuple[np.ndarray, list]:
    """
    Adam Optimizer
    
    Adaptive moment estimation
    
    Parameters
    ----------
    f : callable
        Objective function
    grad_f : callable
        Gradient function
    x0 : array
        Initial point
    learning_rate : float
        Learning rate
    beta1 : float
        Exponential decay for first moment
    beta2 : float
        Exponential decay for second moment
    eps : float
        Small value for numerical stability
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    x : array
        Optimal point
    history : list
        Optimization history
    """
    x = np.asarray(x0).copy()
    m = np.zeros_like(x)  # First moment
    v = np.zeros_like(x)  # Second moment
    history = []
    
    for i in range(1, max_iter + 1):
        grad = grad_f(x)
        
        # Update biased first moment
        m = beta1 * m + (1 - beta1) * grad
        
        # Update biased second moment
        v = beta2 * v + (1 - beta2) * grad ** 2
        
        # Bias correction
        m_hat = m / (1 - beta1 ** i)
        v_hat = v / (1 - beta2 ** i)
        
        # Update parameters
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        
        if i % 100 == 0:
            history.append({
                'iteration': i,
                'x': x.copy(),
                'f(x)': f(x)
            })
    
    return x, history


def convex_optimization(f: Callable, grad_f: Callable, x0: np.ndarray,
                       method: str = 'gradient_descent', **kwargs) -> Tuple[np.ndarray, list]:
    """
    Convex Optimization
    
    Optimize convex function
    
    Parameters
    ----------
    f : callable
        Convex objective function
    grad_f : callable
        Gradient function
    x0 : array
        Initial point
    method : str
        Optimization method
    **kwargs
        Additional parameters
        
    Returns
    -------
    x : array
        Optimal point
    history : list
        Optimization history
    """
    if method == 'gradient_descent':
        return gradient_descent(f, grad_f, x0, **kwargs)
    elif method == 'adam':
        return adam_optimizer(f, grad_f, x0, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
