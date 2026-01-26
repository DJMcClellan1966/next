"""
Systems Theory & Double Bind - Inspired by Gregory Bateson

Implements:
- Multi-Objective Optimization (handling contradictory constraints)
- Double Bind Resolution
- System Hierarchies
- Meta-Communication
"""
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple
import logging
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class MultiObjectiveOptimizer:
    """
    Multi-Objective Optimization
    
    Handles contradictory objectives (double bind resolution)
    """
    
    def __init__(
        self,
        objective_functions: List[Callable],
        bounds: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Initialize multi-objective optimizer
        
        Args:
            objective_functions: List of objective functions to minimize
            bounds: Optional bounds for each dimension
        """
        self.objective_functions = objective_functions
        self.n_objectives = len(objective_functions)
        self.bounds = bounds
    
    def pareto_front(
        self,
        initial_population: Optional[np.ndarray] = None,
        population_size: int = 50,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Find Pareto-optimal solutions
        
        Args:
            initial_population: Initial population
            population_size: Size of population
            max_iterations: Maximum iterations
        
        Returns:
            Pareto front solutions and objectives
        """
        from ml_toolbox.optimization.evolutionary_algorithms import GeneticAlgorithm
        
        # Create combined fitness function
        def combined_fitness(individual: np.ndarray) -> float:
            """Fitness: negative of dominated count (maximize non-dominated solutions)"""
            objectives = [f(individual) for f in self.objective_functions]
            
            # Count how many solutions this dominates
            # (simplified: use hypervolume approximation)
            dominated_count = 0
            
            # For each objective, check if better
            # Simple approximation: sum of normalized objectives
            normalized_objectives = [obj / (abs(obj) + 1) for obj in objectives]
            return -np.sum(normalized_objectives)  # Minimize sum
        
        # Use genetic algorithm
        if self.bounds:
            gene_ranges = self.bounds
        else:
            gene_ranges = [(-10.0, 10.0)] * len(initial_population[0]) if initial_population is not None else [(-10.0, 10.0)] * 10
        
        ga = GeneticAlgorithm(
            fitness_function=combined_fitness,
            gene_ranges=gene_ranges,
            population_size=population_size,
            max_generations=max_iterations
        )
        
        result = ga.evolve()
        
        # Evaluate all objectives for best solution
        best_individual = result['best_individual']
        objectives = [f(best_individual) for f in self.objective_functions]
        
        return {
            'solution': best_individual,
            'objectives': objectives,
            'fitness_history': result['fitness_history']
        }
    
    def weighted_sum(
        self,
        weights: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Weighted sum method for multi-objective optimization
        
        Args:
            weights: Weights for each objective (default: equal)
            initial_guess: Initial solution guess
        
        Returns:
            Optimal solution and objectives
        """
        if weights is None:
            weights = np.ones(self.n_objectives) / self.n_objectives
        
        def combined_objective(x: np.ndarray) -> float:
            """Weighted sum of objectives"""
            objectives = [f(x) for f in self.objective_functions]
            return np.sum(weights * np.array(objectives))
        
        if initial_guess is None:
            if self.bounds:
                initial_guess = np.array([(min_val + max_val) / 2 
                                        for min_val, max_val in self.bounds])
            else:
                initial_guess = np.random.random(10)
        
        result = minimize(combined_objective, initial_guess, bounds=self.bounds, method='L-BFGS-B')
        
        objectives = [f(result.x) for f in self.objective_functions]
        
        return {
            'solution': result.x,
            'objectives': objectives,
            'combined_value': result.fun
        }


class DoubleBindResolver:
    """
    Resolve double bind situations (contradictory constraints)
    """
    
    def __init__(
        self,
        constraints: List[Callable],
        objective_function: Callable
    ):
        """
        Initialize double bind resolver
        
        Args:
            constraints: List of constraint functions (should be <= 0)
            objective_function: Objective to minimize
        """
        self.constraints = constraints
        self.objective_function = objective_function
        self.n_constraints = len(constraints)
    
    def resolve(
        self,
        initial_guess: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        penalty_weight: float = 1000.0
    ) -> Dict[str, Any]:
        """
        Resolve double bind using penalty method
        
        Args:
            initial_guess: Initial solution
            bounds: Solution bounds
            penalty_weight: Penalty weight for constraint violations
        
        Returns:
            Resolved solution and constraint violations
        """
        def penalized_objective(x: np.ndarray) -> float:
            """Objective with penalty for constraint violations"""
            objective = self.objective_function(x)
            
            # Add penalty for violated constraints
            penalty = 0.0
            violations = []
            for constraint in self.constraints:
                violation = max(0, constraint(x))  # Positive if violated
                violations.append(violation)
                penalty += penalty_weight * violation ** 2
            
            return objective + penalty, violations
        
        if initial_guess is None:
            if bounds:
                initial_guess = np.array([(min_val + max_val) / 2 
                                        for min_val, max_val in bounds])
            else:
                initial_guess = np.random.random(10)
        
        # Optimize with increasing penalty
        best_solution = initial_guess
        best_objective = float('inf')
        best_violations = None
        
        for penalty in [10, 100, 1000, 10000]:
            def obj(x):
                obj_val, violations = penalized_objective(x)
                return obj_val
            
            result = minimize(obj, best_solution, bounds=bounds, method='L-BFGS-B')
            
            _, violations = penalized_objective(result.x)
            total_violation = np.sum(violations)
            
            if total_violation < 1e-6:  # Constraints satisfied
                best_solution = result.x
                best_objective = self.objective_function(result.x)
                best_violations = violations
                break
            else:
                if result.fun < best_objective:
                    best_solution = result.x
                    best_objective = result.fun
                    best_violations = violations
        
        return {
            'solution': best_solution,
            'objective': best_objective,
            'constraint_violations': best_violations,
            'satisfied': all(v < 1e-6 for v in best_violations) if best_violations else False
        }


class SystemHierarchy:
    """
    Hierarchical system structure
    """
    
    def __init__(self, levels: List[int]):
        """
        Initialize system hierarchy
        
        Args:
            levels: List of number of components at each level
        """
        self.levels = levels
        self.n_levels = len(levels)
        self.components = {}
        
        # Initialize components at each level
        for level, n_components in enumerate(levels):
            self.components[level] = [{'id': f'L{level}_C{i}', 'state': 0.0} 
                                    for i in range(n_components)]
    
    def update_level(self, level: int, update_function: Callable):
        """
        Update components at a specific level
        
        Args:
            level: Level to update
            update_function: Function that takes component and returns new state
        """
        for component in self.components[level]:
            component['state'] = update_function(component)
    
    def propagate(self, direction: str = 'down'):
        """
        Propagate information through hierarchy
        
        Args:
            direction: 'up' or 'down'
        """
        if direction == 'down':
            # Top-down propagation
            for level in range(self.n_levels - 1):
                parent_level = level
                child_level = level + 1
                
                # Aggregate parent states
                parent_states = [c['state'] for c in self.components[parent_level]]
                aggregate = np.mean(parent_states)
                
                # Update children
                for component in self.components[child_level]:
                    component['state'] = 0.7 * component['state'] + 0.3 * aggregate
        
        else:  # 'up'
            # Bottom-up propagation
            for level in range(self.n_levels - 1, 0, -1):
                child_level = level
                parent_level = level - 1
                
                # Aggregate child states
                child_states = [c['state'] for c in self.components[child_level]]
                aggregate = np.mean(child_states)
                
                # Update parents
                for component in self.components[parent_level]:
                    component['state'] = 0.7 * component['state'] + 0.3 * aggregate
    
    def get_hierarchy_state(self) -> Dict[int, List[float]]:
        """Get state of entire hierarchy"""
        return {level: [c['state'] for c in components] 
                for level, components in self.components.items()}


class MetaCommunication:
    """
    Meta-communication: communication about communication
    """
    
    def __init__(self):
        """Initialize meta-communication system"""
        self.communication_history = []
        self.meta_rules = {}
    
    def communicate(self, message: str, sender: str, receiver: str) -> Dict[str, Any]:
        """
        Communicate with meta-communication tracking
        
        Args:
            message: Message content
            sender: Sender identifier
            receiver: Receiver identifier
        
        Returns:
            Communication result with meta-information
        """
        communication = {
            'message': message,
            'sender': sender,
            'receiver': receiver,
            'timestamp': len(self.communication_history),
            'meta_level': 0
        }
        
        # Check for meta-communication (communication about communication)
        if message.startswith('META:'):
            communication['meta_level'] = 1
            communication['meta_message'] = message[5:]  # Remove 'META:' prefix
        
        self.communication_history.append(communication)
        
        return communication
    
    def analyze_communication_patterns(self) -> Dict[str, Any]:
        """Analyze communication patterns"""
        if len(self.communication_history) == 0:
            return {}
        
        # Count communications
        sender_counts = {}
        receiver_counts = {}
        meta_count = 0
        
        for comm in self.communication_history:
            sender_counts[comm['sender']] = sender_counts.get(comm['sender'], 0) + 1
            receiver_counts[comm['receiver']] = receiver_counts.get(comm['receiver'], 0) + 1
            if comm['meta_level'] > 0:
                meta_count += 1
        
        return {
            'total_communications': len(self.communication_history),
            'meta_communications': meta_count,
            'sender_distribution': sender_counts,
            'receiver_distribution': receiver_counts,
            'meta_ratio': meta_count / len(self.communication_history) if self.communication_history else 0
        }
