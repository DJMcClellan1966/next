"""
Linear Algebra Operations

Implements:
- Vectors
- Matrices
- SVD (Singular Value Decomposition)
- Eigen-decomposition
"""
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Vector:
    """Vector operations"""
    
    @staticmethod
    def dot(v1: np.ndarray, v2: np.ndarray) -> float:
        """Dot product"""
        return np.dot(v1, v2)
    
    @staticmethod
    def norm(v: np.ndarray, ord: int = 2) -> float:
        """Vector norm"""
        return np.linalg.norm(v, ord=ord)
    
    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize vector"""
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v


class Matrix:
    """Matrix operations"""
    
    @staticmethod
    def multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication"""
        return A @ B
    
    @staticmethod
    def transpose(A: np.ndarray) -> np.ndarray:
        """Matrix transpose"""
        return A.T
    
    @staticmethod
    def inverse(A: np.ndarray) -> np.ndarray:
        """Matrix inverse"""
        return np.linalg.inv(A)
    
    @staticmethod
    def determinant(A: np.ndarray) -> float:
        """Matrix determinant"""
        return np.linalg.det(A)
    
    @staticmethod
    def rank(A: np.ndarray) -> int:
        """Matrix rank"""
        return np.linalg.matrix_rank(A)


def svd(A: np.ndarray, full_matrices: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Singular Value Decomposition
    
    A = U @ S @ V^T
    
    Parameters
    ----------
    A : array
        Input matrix
    full_matrices : bool
        Whether to return full matrices
        
    Returns
    -------
    U : array
        Left singular vectors
    S : array
        Singular values
    Vt : array
        Right singular vectors (transposed)
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=full_matrices)
    S = np.diag(s)
    return U, S, Vt


def eigendecomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigen-decomposition
    
    A = Q @ Lambda @ Q^-1
    
    Parameters
    ----------
    A : array
        Input matrix
        
    Returns
    -------
    eigenvalues : array
        Eigenvalues
    eigenvectors : array
        Eigenvectors (columns)
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication"""
    return Matrix.multiply(A, B)


def transpose(A: np.ndarray) -> np.ndarray:
    """Matrix transpose"""
    return Matrix.transpose(A)
