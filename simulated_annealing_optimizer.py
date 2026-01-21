"""
Russell/Norvig Simulated Annealing for Hyperparameter Optimization
Local search algorithm that can escape local optima

Features:
- Simulated annealing for hyperparameter search
- Temperature schedule (exponential, linear, custom)
- Acceptance probability based on temperature
- Better exploration than grid/random search
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import numpy as np
from collections import defaultdict
import warnings
import math

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class SimulatedAnnealingOptimizer:
    """
    Simulated Annealing for hyperparameter optimization
    
    Escapes local optima by accepting worse solutions with decreasing probability
    """
    
    def __init__(
        self,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        n_iterations: int = 100,
        temperature_schedule: str = 'exponential',
        random_state: int = 42
    ):
        """
        Args:
            initial_temperature: Starting temperature
            cooling_rate: Temperature reduction rate
            min_temperature: Minimum temperature (stopping criterion)
            n_iterations: Maximum number of iterations
            temperature_schedule: 'exponential', 'linear', 'logarithmic'
            random_state: Random seed
        """
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.n_iterations = n_iterations
        self.temperature_schedule = temperature_schedule
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.optimization_history_ = []
    
    def _temperature(self, iteration: int) -> float:
        """Calculate temperature at given iteration"""
        if self.temperature_schedule == 'exponential':
            return self.initial_temperature * (self.cooling_rate ** iteration)
        elif self.temperature_schedule == 'linear':
            return self.initial_temperature * (1 - iteration / self.n_iterations)
        elif self.temperature_schedule == 'logarithmic':
            return self.initial_temperature / (1 + iteration)
        else:
            return self.initial_temperature * (self.cooling_rate ** iteration)
    
    def _acceptance_probability(self, current_score: float, new_score: float, temperature: float) -> float:
        """
        Calculate acceptance probability
        
        Accepts better solutions with probability 1
        Accepts worse solutions with probability exp(-delta / temperature)
        """
        if new_score > current_score:
            return 1.0
        
        delta = current_score - new_score
        if temperature == 0:
            return 0.0
        
        return math.exp(-delta / temperature)
    
    def _neighbor(self, current_params: Dict[str, Any], param_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate neighbor solution
        
        Randomly modifies one parameter
        """
        neighbor = current_params.copy()
        
        # Select random parameter to modify
        param_names = list(param_space.keys())
        if len(param_names) == 0:
            return neighbor
        
        param_to_modify = np.random.choice(param_names)
        param_values = param_space[param_to_modify]
        
        # Generate new value
        if isinstance(param_values, tuple) and len(param_values) == 2:
            # Continuous range
            min_val, max_val = param_values
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer range
                new_val = np.random.randint(min_val, max_val + 1)
            else:
                # Float range
                new_val = np.random.uniform(min_val, max_val)
        elif isinstance(param_values, list):
            # Categorical
            new_val = np.random.choice(param_values)
        else:
            # Default: keep current value
            new_val = neighbor.get(param_to_modify, param_values[0] if isinstance(param_values, list) else 0)
        
        neighbor[param_to_modify] = new_val
        return neighbor
    
    def _evaluate_params(
        self,
        model_class: type,
        params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        scoring: Optional[str] = None,
        cv: int = 5
    ) -> float:
        """Evaluate parameter set using cross-validation"""
        if not SKLEARN_AVAILABLE:
            return 0.0
        
        # Create model with parameters
        try:
            model = model_class(**params)
        except Exception as e:
            warnings.warn(f"Failed to create model with params {params}: {e}")
            return -np.inf
        
        # Determine scoring
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'r2'
        
        # CV splitter
        if len(np.unique(y)) < 10:  # Classification
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Cross-validation score
        try:
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
            return np.mean(scores)
        except Exception as e:
            warnings.warn(f"Failed to evaluate model: {e}")
            return -np.inf
    
    def optimize(
        self,
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        param_space: Dict[str, Any],
        initial_params: Optional[Dict[str, Any]] = None,
        scoring: Optional[str] = None,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using simulated annealing
        
        Args:
            model_class: Model class (e.g., RandomForestClassifier)
            X: Features
            y: Labels
            param_space: Parameter space dictionary
            initial_params: Starting parameters (None for random)
            scoring: Scoring metric
            cv: Cross-validation folds
            
        Returns:
            Dictionary with best parameters, best score, optimization history
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Initialize current solution
        if initial_params is None:
            # Random initialization
            current_params = {}
            for name, values in param_space.items():
                if isinstance(values, tuple) and len(values) == 2:
                    min_val, max_val = values
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        current_params[name] = np.random.randint(min_val, max_val + 1)
                    else:
                        current_params[name] = np.random.uniform(min_val, max_val)
                elif isinstance(values, list):
                    current_params[name] = np.random.choice(values)
                else:
                    current_params[name] = values
        else:
            current_params = initial_params.copy()
        
        # Evaluate initial solution
        current_score = self._evaluate_params(model_class, current_params, X, y, scoring, cv)
        
        # Initialize best
        self.best_params_ = current_params.copy()
        self.best_score_ = current_score
        
        # Simulated annealing loop
        for iteration in range(self.n_iterations):
            temperature = self._temperature(iteration)
            
            # Stop if temperature too low
            if temperature < self.min_temperature:
                break
            
            # Generate neighbor
            neighbor_params = self._neighbor(current_params, param_space)
            neighbor_score = self._evaluate_params(model_class, neighbor_params, X, y, scoring, cv)
            
            # Calculate acceptance probability
            accept_prob = self._acceptance_probability(current_score, neighbor_score, temperature)
            
            # Accept or reject
            if np.random.random() < accept_prob:
                current_params = neighbor_params
                current_score = neighbor_score
                
                # Update best if better
                if neighbor_score > self.best_score_:
                    self.best_params_ = neighbor_params.copy()
                    self.best_score_ = neighbor_score
            
            # Record history
            self.optimization_history_.append({
                'iteration': iteration,
                'temperature': temperature,
                'current_score': current_score,
                'best_score': self.best_score_,
                'accepted': accept_prob > np.random.random()
            })
        
        return {
            'best_params': self.best_params_,
            'best_score': float(self.best_score_),
            'n_iterations': len(self.optimization_history_),
            'optimization_history': self.optimization_history_,
            'method': 'simulated_annealing'
        }
