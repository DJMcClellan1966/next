"""
Calculus Operations

Implements:
- Derivatives
- Gradients
- Chain Rule
- Jacobian
- Hessian
"""
import numpy as np
from typing import Callable, Tuple
import logging

logger = logging.getLogger(__name__)


def derivative(f: Callable, x: float, h: float = 1e-5) -> float:
    """
    Numerical derivative
    
    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    
    Parameters
    ----------
    f : callable
        Function
    x : float
        Point
    h : float
        Step size
        
    Returns
    -------
    derivative : float
        Numerical derivative
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def gradient(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Numerical gradient
    
    Parameters
    ----------
    f : callable
        Function f(x) -> float
    x : array
        Point
    h : float
        Step size
        
    Returns
    -------
    gradient : array
        Gradient vector
    """
    x = np.asarray(x)
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad


def chain_rule(f: Callable, g: Callable, x: float, h: float = 1e-5) -> float:
    """
    Chain rule
    
    (f(g(x)))' = f'(g(x)) * g'(x)
    
    Parameters
    ----------
    f : callable
        Outer function
    g : callable
        Inner function
    x : float
        Point
    h : float
        Step size
        
    Returns
    -------
    derivative : float
        Derivative of composition
    """
    g_x = g(x)
    f_prime_g = derivative(f, g_x, h)
    g_prime_x = derivative(g, x, h)
    
    return f_prime_g * g_prime_x


def jacobian(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Jacobian matrix
    
    J_ij = ∂f_i / ∂x_j
    
    Parameters
    ----------
    f : callable
        Vector-valued function f(x) -> array
    x : array
        Point
    h : float
        Step size
        
    Returns
    -------
    jacobian : array
        Jacobian matrix
    """
    x = np.asarray(x)
    f_x = np.asarray(f(x))
    
    n_outputs = len(f_x) if f_x.ndim == 1 else f_x.size
    n_inputs = len(x)
    
    J = np.zeros((n_outputs, n_inputs))
    
    for j in range(n_inputs):
        x_plus = x.copy()
        x_plus[j] += h
        x_minus = x.copy()
        x_minus[j] -= h
        
        f_plus = np.asarray(f(x_plus))
        f_minus = np.asarray(f(x_minus))
        
        if f_plus.ndim > 1:
            f_plus = f_plus.flatten()
        if f_minus.ndim > 1:
            f_minus = f_minus.flatten()
        
        J[:, j] = (f_plus - f_minus) / (2 * h)
    
    return J


def hessian(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Hessian matrix
    
    H_ij = ∂²f / ∂x_i ∂x_j
    
    Parameters
    ----------
    f : callable
        Scalar function f(x) -> float
    x : array
        Point
    h : float
        Step size
        
    Returns
    -------
    hessian : array
        Hessian matrix
    """
    x = np.asarray(x)
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Second-order finite difference
            x_ij = x.copy()
            x_ij[i] += h
            x_ij[j] += h
            
            x_i = x.copy()
            x_i[i] += h
            
            x_j = x.copy()
            x_j[j] += h
            
            H[i, j] = (f(x_ij) - f(x_i) - f(x_j) + f(x)) / (h * h)
    
    return H
