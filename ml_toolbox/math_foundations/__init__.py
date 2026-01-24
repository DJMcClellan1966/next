"""
Essential Mathematical Foundations

Implements:
- Linear Algebra (Vectors, Matrices, SVD, Eigen-decomposition)
- Calculus (Derivatives, Gradients, Chain Rule)
- Probability & Statistics (Distributions, Bayesian, MLE)
- Optimization (Gradient Descent, Convex Optimization)
"""
try:
    from .linear_algebra import (
        Vector, Matrix, svd, eigendecomposition, matrix_multiply, transpose
    )
    from .calculus import (
        derivative, gradient, chain_rule, jacobian, hessian
    )
    from .probability_statistics import (
        ProbabilityDistribution, Gaussian, BayesianInference, MLE,
        expectation, variance, covariance
    )
    from .optimization import (
        gradient_descent, stochastic_gradient_descent, adam_optimizer,
        convex_optimization
    )
    __all__ = [
        'Vector',
        'Matrix',
        'svd',
        'eigendecomposition',
        'matrix_multiply',
        'transpose',
        'derivative',
        'gradient',
        'chain_rule',
        'jacobian',
        'hessian',
        'ProbabilityDistribution',
        'Gaussian',
        'BayesianInference',
        'MLE',
        'expectation',
        'variance',
        'covariance',
        'gradient_descent',
        'stochastic_gradient_descent',
        'adam_optimizer',
        'convex_optimization'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Some math foundations not available: {e}")
    __all__ = []
