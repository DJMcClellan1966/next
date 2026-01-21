"""
Russell/Norvig Constraint Satisfaction for Hyperparameter Optimization
Handle complex constraints in hyperparameter spaces

Features:
- CSP-based hyperparameter optimization
- Constraint definition and validation
- Backtracking search for constrained spaces
- Constraint-based feature engineering
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Set, Callable
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class Constraint:
    """
    Constraint for CSP
    
    Represents a constraint on hyperparameters
    """
    
    def __init__(self, constraint_func: Callable, description: str = ""):
        """
        Args:
            constraint_func: Function that takes params dict and returns bool
            description: Human-readable description
        """
        self.constraint_func = constraint_func
        self.description = description
    
    def check(self, params: Dict[str, Any]) -> bool:
        """Check if constraint is satisfied"""
        try:
            return self.constraint_func(params)
        except Exception as e:
            warnings.warn(f"Constraint check failed: {e}")
            return False


class CSPOptimizer:
    """
    Constraint Satisfaction Problem Optimizer
    
    Finds hyperparameters that satisfy constraints while optimizing objective
    """
    
    def __init__(
        self,
        constraints: Optional[List[Constraint]] = None,
        random_state: int = 42
    ):
        """
        Args:
            constraints: List of Constraint objects
            random_state: Random seed
        """
        self.constraints = constraints or []
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.optimization_history_ = []
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint"""
        self.constraints.append(constraint)
    
    def _satisfies_constraints(self, params: Dict[str, Any]) -> bool:
        """Check if parameters satisfy all constraints"""
        for constraint in self.constraints:
            if not constraint.check(params):
                return False
        return True
    
    def _generate_candidate(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random candidate from parameter space"""
        candidate = {}
        for name, values in param_space.items():
            if isinstance(values, tuple) and len(values) == 2:
                min_val, max_val = values
                if isinstance(min_val, int) and isinstance(max_val, int):
                    candidate[name] = np.random.randint(min_val, max_val + 1)
                else:
                    candidate[name] = np.random.uniform(min_val, max_val)
            elif isinstance(values, list):
                candidate[name] = np.random.choice(values)
            else:
                candidate[name] = values
        return candidate
    
    def _backtrack_search(
        self,
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        param_space: Dict[str, Any],
        assignment: Dict[str, Any],
        unassigned: List[str],
        scoring: Optional[str],
        cv: int,
        max_iterations: int = 1000
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Backtracking search for CSP
        
        Recursively assigns values to unassigned variables
        """
        if len(unassigned) == 0:
            # All variables assigned, evaluate
            if self._satisfies_constraints(assignment):
                score = self._evaluate_params(model_class, assignment, X, y, scoring, cv)
                return assignment, score
            else:
                return None, -np.inf
        
        if len(self.optimization_history_) >= max_iterations:
            return None, -np.inf
        
        # Select variable (first unassigned)
        var = unassigned[0]
        remaining = unassigned[1:]
        
        # Try values for this variable
        values = param_space[var]
        if isinstance(values, tuple) and len(values) == 2:
            # Sample from range
            min_val, max_val = values
            if isinstance(min_val, int) and isinstance(max_val, int):
                value_list = list(range(min_val, max_val + 1))
            else:
                # Sample 5 values from continuous range
                value_list = np.linspace(min_val, max_val, 5).tolist()
        elif isinstance(values, list):
            value_list = values
        else:
            value_list = [values]
        
        best_assignment = None
        best_score = -np.inf
        
        for value in value_list:
            new_assignment = assignment.copy()
            new_assignment[var] = value
            
            # Check constraints (early pruning)
            if not self._satisfies_constraints(new_assignment):
                continue
            
            # Recursive call
            result_assignment, result_score = self._backtrack_search(
                model_class, X, y, param_space, new_assignment, remaining, scoring, cv, max_iterations
            )
            
            if result_score > best_score:
                best_score = result_score
                best_assignment = result_assignment
        
        return best_assignment, best_score
    
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
        
        try:
            model = model_class(**params)
        except Exception as e:
            return -np.inf
        
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'r2'
        
        if len(np.unique(y)) < 10:
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        try:
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
            return np.mean(scores)
        except:
            return -np.inf
    
    def optimize(
        self,
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        param_space: Dict[str, Any],
        scoring: Optional[str] = None,
        cv: int = 5,
        method: str = 'backtrack',
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters with constraints
        
        Args:
            model_class: Model class
            X: Features
            y: Labels
            param_space: Parameter space
            scoring: Scoring metric
            cv: Cross-validation folds
            method: 'backtrack' or 'random'
            n_samples: Number of random samples (for random method)
            
        Returns:
            Dictionary with best parameters, best score, optimization history
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if method == 'backtrack':
            # Backtracking search
            unassigned = list(param_space.keys())
            result_assignment, result_score = self._backtrack_search(
                model_class, X, y, param_space, {}, unassigned, scoring, cv
            )
            
            if result_assignment is not None:
                self.best_params_ = result_assignment
                self.best_score_ = result_score
        
        elif method == 'random':
            # Random sampling with constraint checking
            for i in range(n_samples):
                candidate = self._generate_candidate(param_space)
                
                if not self._satisfies_constraints(candidate):
                    continue
                
                score = self._evaluate_params(model_class, candidate, X, y, scoring, cv)
                
                if score > self.best_score_:
                    self.best_params_ = candidate.copy()
                    self.best_score_ = score
                
                self.optimization_history_.append({
                    'iteration': i,
                    'score': score,
                    'satisfies_constraints': True
                })
        
        return {
            'best_params': self.best_params_,
            'best_score': float(self.best_score_),
            'n_constraints': len(self.constraints),
            'optimization_history': self.optimization_history_,
            'method': 'csp_' + method
        }


def create_constraint(constraint_expr: str) -> Constraint:
    """
    Create constraint from expression string
    
    Examples:
        "n_estimators > max_depth"
        "learning_rate < 0.1"
        "max_depth * 2 < n_estimators"
    """
    def constraint_func(params):
        # Simple evaluation (in production, use safer eval)
        try:
            # Replace parameter names with values
            expr = constraint_expr
            for name, value in params.items():
                expr = expr.replace(name, str(value))
            return eval(expr)
        except:
            return False
    
    return Constraint(constraint_func, constraint_expr)
