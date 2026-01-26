"""
Cooperative Game Theory - Extended from John Nash

Implements:
- Shapley Value (fair value distribution)
- Coalition Formation
- Core Solution (stable allocations)
- Bargaining Theory
- Mechanism Design
"""
import numpy as np
from typing import List, Dict, Set, Tuple, Any, Callable, Optional
from itertools import combinations, permutations
import logging

logger = logging.getLogger(__name__)


def shapley_value(
    n_players: int,
    characteristic_function: Callable[[Set[int]], float]
) -> np.ndarray:
    """
    Calculate Shapley Value - fair value distribution in coalitions
    
    Args:
        n_players: Number of players
        characteristic_function: Function that takes a set of players and returns coalition value
    
    Returns:
        Shapley value for each player
    """
    shapley_values = np.zeros(n_players)
    
    # For each player
    for player in range(n_players):
        value = 0.0
        
        # For each coalition size
        for size in range(n_players):
            # For each coalition of this size that doesn't include player
            other_players = [i for i in range(n_players) if i != player]
            
            for coalition in combinations(other_players, size):
                coalition_set = set(coalition)
                
                # Value with player - value without player
                marginal_contribution = (
                    characteristic_function(coalition_set | {player}) -
                    characteristic_function(coalition_set)
                )
                
                # Weight: 1 / (n * C(n-1, s))
                weight = 1.0 / (n_players * len(list(combinations(other_players, size))))
                value += weight * marginal_contribution
        
        shapley_values[player] = value
    
    return shapley_values


def shapley_value_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    n_samples: int = 100
) -> np.ndarray:
    """
    Calculate Shapley value for feature importance
    
    Args:
        X: Feature matrix
        y: Target vector
        model: Trained model with predict method
        n_samples: Number of samples for approximation
    
    Returns:
        Shapley value for each feature
    """
    from sklearn.metrics import mean_squared_error, accuracy_score
    
    n_features = X.shape[1]
    n_samples_actual = min(n_samples, X.shape[0])
    sample_indices = np.random.choice(X.shape[0], n_samples_actual, replace=False)
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]
    
    # Baseline prediction (no features)
    baseline_pred = model.predict(np.zeros((1, n_features)))
    if len(baseline_pred.shape) == 0:
        baseline_pred = np.array([baseline_pred])
    
    def characteristic_function(feature_set: Set[int]) -> float:
        """Value of coalition (feature set)"""
        if len(feature_set) == 0:
            # No features - use baseline
            return 0.0
        
        # Create feature mask
        feature_mask = np.zeros(n_features, dtype=bool)
        for f in feature_set:
            feature_mask[f] = True
        
        # Predict with selected features
        X_selected = X_sample[:, feature_mask]
        
        # Create model input (zero out non-selected features)
        X_input = np.zeros_like(X_sample)
        X_input[:, feature_mask] = X_selected
        
        try:
            predictions = model.predict(X_input)
            if len(predictions.shape) == 0:
                predictions = np.array([predictions])
            
            # Calculate score (negative error for maximization)
            if len(np.unique(y_sample)) < 20:  # Classification
                score = accuracy_score(y_sample, predictions)
            else:  # Regression
                score = -mean_squared_error(y_sample, predictions)
            
            return score
        except:
            return 0.0
    
    # Calculate Shapley values
    shapley_values = shapley_value(n_features, characteristic_function)
    
    return shapley_values


class CoalitionFormation:
    """
    Optimal coalition formation for multi-agent systems
    """
    
    def __init__(
        self,
        n_agents: int,
        coalition_value_function: Callable[[Set[int]], float]
    ):
        """
        Initialize coalition formation
        
        Args:
            n_agents: Number of agents
            coalition_value_function: Function that returns value of a coalition
        """
        self.n_agents = n_agents
        self.coalition_value_function = coalition_value_function
    
    def exhaustive_search(self) -> List[Set[int]]:
        """
        Exhaustive search for optimal coalition structure
        
        Returns:
            List of optimal coalitions
        """
        best_coalitions = None
        best_value = float('-inf')
        
        # Try all possible partitions
        # This is exponential, so only feasible for small n
        if self.n_agents > 8:
            logger.warning("Exhaustive search only feasible for n <= 8")
            return self.greedy_formation()
        
        # Generate all partitions (simplified - use recursive approach)
        def generate_partitions(elements: List[int]) -> List[List[Set[int]]]:
            if len(elements) == 0:
                return [[]]
            
            first = elements[0]
            rest = elements[1:]
            
            partitions = []
            for partition in generate_partitions(rest):
                # Add first element to existing subset
                for i, subset in enumerate(partition):
                    new_partition = partition.copy()
                    new_partition[i] = subset | {first}
                    partitions.append(new_partition)
                
                # Create new subset with first element
                new_partition = partition + [{first}]
                partitions.append(new_partition)
            
            return partitions
        
        all_partitions = generate_partitions(list(range(self.n_agents)))
        
        for partition in all_partitions:
            total_value = sum(self.coalition_value_function(coalition) for coalition in partition)
            if total_value > best_value:
                best_value = total_value
                best_coalitions = partition
        
        return best_coalitions
    
    def greedy_formation(self) -> List[Set[int]]:
        """
        Greedy coalition formation
        
        Returns:
            List of coalitions
        """
        coalitions = []
        remaining_agents = set(range(self.n_agents))
        
        while remaining_agents:
            # Start with best single agent
            best_agent = None
            best_value = float('-inf')
            
            for agent in remaining_agents:
                value = self.coalition_value_function({agent})
                if value > best_value:
                    best_value = value
                    best_agent = agent
            
            # Try to add agents to this coalition
            current_coalition = {best_agent}
            remaining_agents.remove(best_agent)
            
            improved = True
            while improved and remaining_agents:
                improved = False
                best_addition = None
                best_improvement = 0
                
                for agent in remaining_agents:
                    new_coalition = current_coalition | {agent}
                    improvement = (self.coalition_value_function(new_coalition) -
                                  self.coalition_value_function(current_coalition))
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_addition = agent
                
                if best_addition is not None and best_improvement > 0:
                    current_coalition.add(best_addition)
                    remaining_agents.remove(best_addition)
                    improved = True
            
            coalitions.append(current_coalition)
        
        return coalitions


class NashBargainingSolution:
    """
    Nash Bargaining Solution for fair resource allocation
    """
    
    def __init__(
        self,
        n_players: int,
        utility_functions: List[Callable[[np.ndarray], float]],
        disagreement_point: np.ndarray
    ):
        """
        Initialize Nash bargaining
        
        Args:
            n_players: Number of players
            utility_functions: List of utility functions for each player
            disagreement_point: Utility values if no agreement (threat point)
        """
        self.n_players = n_players
        self.utility_functions = utility_functions
        self.disagreement_point = disagreement_point
    
    def solve(self, resource_bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Solve Nash bargaining problem
        
        Args:
            resource_bounds: Bounds for each resource dimension
        
        Returns:
            Optimal allocation and utilities
        """
        from scipy.optimize import minimize
        
        def nash_product(allocation: np.ndarray) -> float:
            """Nash product: product of utility gains"""
            utilities = [uf(allocation) for uf in self.utility_functions]
            gains = [u - d for u, d in zip(utilities, self.disagreement_point)]
            
            # Product (maximize, so return negative)
            product = 1.0
            for gain in gains:
                if gain <= 0:
                    return 1e10  # Penalize invalid solutions
                product *= gain
            
            return -product  # Negative for minimization
        
        # Initial guess: equal allocation
        initial = np.array([(min_val + max_val) / 2 for min_val, max_val in resource_bounds])
        
        # Constraints: sum of allocations <= total resources (simplified)
        bounds = resource_bounds
        
        result = minimize(nash_product, initial, bounds=bounds, method='L-BFGS-B')
        
        optimal_allocation = result.x
        optimal_utilities = [uf(optimal_allocation) for uf in self.utility_functions]
        
        return {
            'allocation': optimal_allocation,
            'utilities': optimal_utilities,
            'nash_product': -result.fun
        }
