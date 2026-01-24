"""
Search and Planning Algorithms

Implements:
- A* Search
- Adversarial Search (Minimax, Alpha-Beta Pruning)
- Constraint Satisfaction Problems (CSP)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """Node for search algorithms"""
    state: Any
    parent: Optional['Node'] = None
    action: Optional[Any] = None
    cost: float = 0.0
    heuristic: float = 0.0
    depth: int = 0


class AStar:
    """
    A* Search Algorithm
    
    Optimal pathfinding using heuristic function
    """
    
    def __init__(self, heuristic_fn: Callable, cost_fn: Callable = None):
        """
        Initialize A* search
        
        Parameters
        ----------
        heuristic_fn : callable
            Heuristic function h(state) -> float
        cost_fn : callable, optional
            Cost function g(state1, state2) -> float
        """
        self.heuristic_fn = heuristic_fn
        self.cost_fn = cost_fn or (lambda s1, s2: 1.0)
    
    def search(self, start: Any, goal: Any, 
              successors_fn: Callable) -> Optional[List[Any]]:
        """
        Perform A* search
        
        Parameters
        ----------
        start : any
            Start state
        goal : any
            Goal state
        successors_fn : callable
            Function that returns (state, action, cost) tuples
            
        Returns
        -------
        path : list, optional
            Path from start to goal, or None if not found
        """
        # Priority queue: (f_score, state, node)
        open_set = []
        closed_set = set()
        
        # Start node
        start_node = Node(
            state=start,
            cost=0.0,
            heuristic=self.heuristic_fn(start, goal)
        )
        open_set.append((start_node.cost + start_node.heuristic, start, start_node))
        
        # Track best path
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: start_node.heuristic}
        
        while open_set:
            # Get node with lowest f_score
            open_set.sort(key=lambda x: x[0])
            current_f, current_state, current_node = open_set.pop(0)
            
            if current_state in closed_set:
                continue
            
            closed_set.add(current_state)
            
            # Check if goal reached
            if current_state == goal:
                # Reconstruct path
                path = []
                node = current_node
                while node:
                    path.insert(0, node.state)
                    node = node.parent
                return path
            
            # Explore successors
            for next_state, action, step_cost in successors_fn(current_state):
                if next_state in closed_set:
                    continue
                
                tentative_g = g_score[current_state] + step_cost
                
                if next_state not in g_score or tentative_g < g_score[next_state]:
                    # Better path found
                    came_from[next_state] = current_state
                    g_score[next_state] = tentative_g
                    h = self.heuristic_fn(next_state, goal)
                    f_score[next_state] = tentative_g + h
                    
                    # Create node
                    next_node = Node(
                        state=next_state,
                        parent=current_node,
                        action=action,
                        cost=tentative_g,
                        heuristic=h,
                        depth=current_node.depth + 1
                    )
                    
                    open_set.append((f_score[next_state], next_state, next_node))
        
        return None  # No path found


class AdversarialSearch:
    """
    Adversarial Search (Minimax with Alpha-Beta Pruning)
    
    For two-player zero-sum games
    """
    
    def __init__(self, evaluation_fn: Callable, max_depth: int = 5):
        """
        Initialize adversarial search
        
        Parameters
        ----------
        evaluation_fn : callable
            Evaluation function state -> float (positive for player 1, negative for player 2)
        max_depth : int
            Maximum search depth
        """
        self.evaluation_fn = evaluation_fn
        self.max_depth = max_depth
    
    def minimax(self, state: Any, depth: int, maximizing: bool,
               get_moves_fn: Callable, is_terminal_fn: Callable) -> Tuple[float, Optional[Any]]:
        """
        Minimax algorithm
        
        Parameters
        ----------
        state : any
            Current game state
        depth : int
            Current depth
        maximizing : bool
            Whether maximizing player's turn
        get_moves_fn : callable
            Function that returns list of (state, action) tuples
        is_terminal_fn : callable
            Function that checks if state is terminal
            
        Returns
        -------
        value : float
            Best value
        best_action : any, optional
            Best action
        """
        if depth == 0 or is_terminal_fn(state):
            return self.evaluation_fn(state), None
        
        moves = get_moves_fn(state)
        if not moves:
            return self.evaluation_fn(state), None
        
        if maximizing:
            max_value = float('-inf')
            best_action = None
            
            for next_state, action in moves:
                value, _ = self.minimax(next_state, depth - 1, False, 
                                      get_moves_fn, is_terminal_fn)
                if value > max_value:
                    max_value = value
                    best_action = action
            
            return max_value, best_action
        else:
            min_value = float('inf')
            best_action = None
            
            for next_state, action in moves:
                value, _ = self.minimax(next_state, depth - 1, True,
                                      get_moves_fn, is_terminal_fn)
                if value < min_value:
                    min_value = value
                    best_action = action
            
            return min_value, best_action
    
    def alpha_beta(self, state: Any, depth: int, alpha: float, beta: float,
                  maximizing: bool, get_moves_fn: Callable,
                  is_terminal_fn: Callable) -> Tuple[float, Optional[Any]]:
        """
        Alpha-Beta Pruning
        
        Parameters
        ----------
        state : any
            Current game state
        depth : int
            Current depth
        alpha : float
            Best value for maximizing player
        beta : float
            Best value for minimizing player
        maximizing : bool
            Whether maximizing player's turn
        get_moves_fn : callable
            Function that returns list of (state, action) tuples
        is_terminal_fn : callable
            Function that checks if state is terminal
            
        Returns
        -------
        value : float
            Best value
        best_action : any, optional
            Best action
        """
        if depth == 0 or is_terminal_fn(state):
            return self.evaluation_fn(state), None
        
        moves = get_moves_fn(state)
        if not moves:
            return self.evaluation_fn(state), None
        
        if maximizing:
            max_value = float('-inf')
            best_action = None
            
            for next_state, action in moves:
                value, _ = self.alpha_beta(next_state, depth - 1, alpha, beta,
                                         False, get_moves_fn, is_terminal_fn)
                if value > max_value:
                    max_value = value
                    best_action = action
                
                alpha = max(alpha, max_value)
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_value, best_action
        else:
            min_value = float('inf')
            best_action = None
            
            for next_state, action in moves:
                value, _ = self.alpha_beta(next_state, depth - 1, alpha, beta,
                                         True, get_moves_fn, is_terminal_fn)
                if value < min_value:
                    min_value = value
                    best_action = action
                
                beta = min(beta, min_value)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_value, best_action
    
    def get_best_move(self, state: Any, get_moves_fn: Callable,
                     is_terminal_fn: Callable, use_alpha_beta: bool = True) -> Optional[Any]:
        """
        Get best move using minimax or alpha-beta
        
        Parameters
        ----------
        state : any
            Current game state
        get_moves_fn : callable
            Function that returns list of (state, action) tuples
        is_terminal_fn : callable
            Function that checks if state is terminal
        use_alpha_beta : bool
            Whether to use alpha-beta pruning
            
        Returns
        -------
        best_action : any, optional
            Best action
        """
        if use_alpha_beta:
            _, best_action = self.alpha_beta(state, self.max_depth, 
                                            float('-inf'), float('inf'),
                                            True, get_moves_fn, is_terminal_fn)
        else:
            _, best_action = self.minimax(state, self.max_depth, True,
                                        get_moves_fn, is_terminal_fn)
        
        return best_action


class ConstraintSatisfaction:
    """
    Constraint Satisfaction Problem (CSP) Solver
    
    Uses backtracking with constraint propagation
    """
    
    def __init__(self, variables: List[Any], domains: Dict[Any, List[Any]],
                constraints: List[Callable]):
        """
        Initialize CSP
        
        Parameters
        ----------
        variables : list
            List of variables
        domains : dict
            Domain for each variable {variable: [values]}
        constraints : list of callable
            Constraint functions (variable, value, assignment) -> bool
        """
        self.variables = variables
        self.domains = domains.copy()
        self.constraints = constraints
        self.assignment = {}
    
    def is_consistent(self, variable: Any, value: Any, assignment: Dict) -> bool:
        """
        Check if assignment is consistent with constraints
        
        Parameters
        ----------
        variable : any
            Variable to check
        value : any
            Value to assign
        assignment : dict
            Current assignment
            
        Returns
        -------
        consistent : bool
            Whether assignment is consistent
        """
        test_assignment = assignment.copy()
        test_assignment[variable] = value
        
        for constraint in self.constraints:
            if not constraint(variable, value, test_assignment):
                return False
        
        return True
    
    def select_unassigned_variable(self, assignment: Dict) -> Optional[Any]:
        """Select next unassigned variable (MRV heuristic)"""
        unassigned = [v for v in self.variables if v not in assignment]
        if not unassigned:
            return None
        
        # Most Constrained Variable (MRV)
        return min(unassigned, key=lambda v: len(self.domains[v]))
    
    def order_domain_values(self, variable: Any, assignment: Dict) -> List[Any]:
        """Order domain values (Least Constraining Value heuristic)"""
        return self.domains[variable]
    
    def backtrack(self, assignment: Dict) -> Optional[Dict]:
        """
        Backtracking search
        
        Parameters
        ----------
        assignment : dict
            Current assignment
            
        Returns
        -------
        solution : dict, optional
            Complete assignment if found, None otherwise
        """
        if len(assignment) == len(self.variables):
            return assignment
        
        variable = self.select_unassigned_variable(assignment)
        if variable is None:
            return assignment
        
        for value in self.order_domain_values(variable, assignment):
            if self.is_consistent(variable, value, assignment):
                assignment[variable] = value
                
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                
                del assignment[variable]
        
        return None
    
    def solve(self) -> Optional[Dict]:
        """
        Solve CSP
        
        Returns
        -------
        solution : dict, optional
            Solution assignment if found, None otherwise
        """
        return self.backtrack({})
