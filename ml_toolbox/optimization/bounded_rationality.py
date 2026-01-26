"""
Bounded Rationality & Satisficing - Inspired by Herbert Simon

Implements:
- Satisficing Optimizers
- Heuristic Model Selection
- Adaptive Aspiration Levels
- Fast Approximate Solutions
"""
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SatisficingOptimizer:
    """
    Satisficing Optimizer - "Good enough" solutions instead of optimal
    
    Stops when solution meets satisfaction threshold
    """
    
    def __init__(
        self,
        objective_function: Callable,
        satisfaction_threshold: float,
        initial_solution: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        max_iterations: int = 1000,
        step_size: float = 0.1
    ):
        """
        Initialize satisficing optimizer
        
        Args:
            objective_function: Function to minimize
            satisfaction_threshold: Acceptable objective value
            initial_solution: Starting point
            bounds: Optional bounds for each dimension
            max_iterations: Maximum iterations
            step_size: Step size for search
        """
        self.objective_function = objective_function
        self.satisfaction_threshold = satisfaction_threshold
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.step_size = step_size
        
        if initial_solution is None:
            if bounds:
                self.current_solution = np.array([(min_val + max_val) / 2 
                                                 for min_val, max_val in bounds])
            else:
                self.current_solution = np.random.random(10)
        else:
            self.current_solution = np.array(initial_solution).copy()
        
        self.current_value = self.objective_function(self.current_solution)
        self.history = [self.current_value]
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run satisficing optimization
        
        Returns:
            Solution and history
        """
        iteration = 0
        
        while iteration < self.max_iterations:
            # Check if satisfied
            if self.current_value <= self.satisfaction_threshold:
                logger.info(f"Solution satisfied at iteration {iteration}")
                break
            
            # Generate neighbor
            neighbor = self.current_solution + np.random.normal(0, self.step_size, 
                                                                size=self.current_solution.shape)
            
            # Apply bounds
            if self.bounds:
                for i, (min_val, max_val) in enumerate(self.bounds):
                    neighbor[i] = np.clip(neighbor[i], min_val, max_val)
            
            neighbor_value = self.objective_function(neighbor)
            
            # Accept if better or if close enough (satisficing)
            if neighbor_value < self.current_value or neighbor_value <= self.satisfaction_threshold:
                self.current_solution = neighbor
                self.current_value = neighbor_value
                self.history.append(self.current_value)
            
            iteration += 1
        
        return {
            'solution': self.current_solution,
            'value': self.current_value,
            'satisfied': self.current_value <= self.satisfaction_threshold,
            'iterations': iteration,
            'history': self.history
        }


class AdaptiveAspirationLevel:
    """
    Adaptive aspiration levels for dynamic goal adjustment
    """
    
    def __init__(
        self,
        initial_aspiration: float,
        adaptation_rate: float = 0.1,
        min_aspiration: float = 0.0,
        max_aspiration: float = float('inf')
    ):
        """
        Initialize adaptive aspiration
        
        Args:
            initial_aspiration: Initial aspiration level
            adaptation_rate: Rate of adaptation (0-1)
            min_aspiration: Minimum aspiration
            max_aspiration: Maximum aspiration
        """
        self.aspiration = initial_aspiration
        self.initial_aspiration = initial_aspiration
        self.adaptation_rate = adaptation_rate
        self.min_aspiration = min_aspiration
        self.max_aspiration = max_aspiration
        self.history = [initial_aspiration]
    
    def update(self, current_performance: float) -> float:
        """
        Update aspiration based on current performance
        
        Args:
            current_performance: Current performance metric
        
        Returns:
            Updated aspiration level
        """
        # If performance exceeds aspiration, raise aspiration
        # If performance below aspiration, lower aspiration
        if current_performance >= self.aspiration:
            # Raise aspiration (optimistic)
            self.aspiration = self.aspiration + self.adaptation_rate * \
                             (current_performance - self.aspiration)
        else:
            # Lower aspiration (realistic)
            self.aspiration = self.aspiration - self.adaptation_rate * \
                             (self.aspiration - current_performance)
        
        # Apply bounds
        self.aspiration = np.clip(self.aspiration, self.min_aspiration, self.max_aspiration)
        self.history.append(self.aspiration)
        
        return self.aspiration
    
    def is_satisfied(self, performance: float) -> bool:
        """Check if performance satisfies aspiration"""
        return performance >= self.aspiration


class HeuristicModelSelector:
    """
    Fast heuristic model selection (satisficing approach)
    """
    
    def __init__(
        self,
        models: List[Any],
        satisfaction_threshold: float = 0.8,
        max_evaluations: int = 10
    ):
        """
        Initialize heuristic model selector
        
        Args:
            models: List of models to choose from
            satisfaction_threshold: Minimum acceptable performance
            max_evaluations: Maximum models to evaluate
        """
        self.models = models
        self.satisfaction_threshold = satisfaction_threshold
        self.max_evaluations = max_evaluations
    
    def select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: Callable = None
    ) -> Dict[str, Any]:
        """
        Select model using heuristic (satisficing)
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            metric: Evaluation metric (default: accuracy for classification, R2 for regression)
        
        Returns:
            Selected model and information
        """
        from sklearn.metrics import accuracy_score, r2_score
        
        if metric is None:
            # Default metric based on task
            if len(np.unique(y_train)) < 20:
                metric = lambda y_true, y_pred: accuracy_score(y_true, y_pred)
            else:
                metric = lambda y_true, y_pred: r2_score(y_true, y_pred)
        
        # Evaluate models until satisfied
        best_model = None
        best_score = float('-inf')
        evaluated = []
        
        # Shuffle order for diversity
        model_indices = np.random.permutation(len(self.models))
        
        for i, model_idx in enumerate(model_indices[:self.max_evaluations]):
            model = self.models[model_idx]
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_val)
                score = metric(y_val, y_pred)
                
                evaluated.append({
                    'model_idx': model_idx,
                    'model': model,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_model = model
                
                # Check if satisfied
                if score >= self.satisfaction_threshold:
                    logger.info(f"Found satisfactory model at evaluation {i+1}")
                    break
            except Exception as e:
                logger.warning(f"Model {model_idx} failed: {e}")
                continue
        
        return {
            'selected_model': best_model,
            'selected_score': best_score,
            'satisfied': best_score >= self.satisfaction_threshold,
            'evaluated_models': len(evaluated),
            'all_results': evaluated
        }


def fast_approximate_inference(
    model: Any,
    X: np.ndarray,
    n_samples: int = 100,
    method: str = 'sampling'
) -> np.ndarray:
    """
    Fast approximate inference using heuristics
    
    Args:
        model: Model with predict method
        X: Input data
        n_samples: Number of samples for approximation
        method: 'sampling', 'mean', 'median'
    
    Returns:
        Approximate predictions
    """
    if method == 'sampling':
        # Sample subset for fast prediction
        sample_indices = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
        X_sample = X[sample_indices]
        predictions = model.predict(X_sample)
        
        # Extend to full dataset (simplified: use nearest neighbor)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_sample)
        _, indices = nn.kneighbors(X)
        return predictions[indices.flatten()]
    
    elif method == 'mean':
        # Use mean prediction
        sample_indices = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
        X_sample = X[sample_indices]
        predictions = model.predict(X_sample)
        mean_pred = np.mean(predictions)
        return np.full(X.shape[0], mean_pred)
    
    elif method == 'median':
        # Use median prediction
        sample_indices = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
        X_sample = X[sample_indices]
        predictions = model.predict(X_sample)
        median_pred = np.median(predictions)
        return np.full(X.shape[0], median_pred)
    
    else:
        raise ValueError(f"Unknown method: {method}")
