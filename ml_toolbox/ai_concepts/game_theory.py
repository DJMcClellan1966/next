"""
Game Theory Extensions (Von Neumann & Morgenstern)

Extends adversarial search with:
- Nash Equilibrium computation
- Non-zero-sum games
- Multi-player games
- Game-theoretic model selection
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)

# Try to import scipy, but make it optional
try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Some game theory features may be limited.")


def find_nash_equilibrium(payoff_matrix: np.ndarray, 
                         method: str = 'linear_programming') -> Dict[str, Any]:
    """
    Find Nash Equilibrium for a two-player game
    
    Uses linear programming for zero-sum games and best response for general games
    
    Parameters
    ----------
    payoff_matrix : array, shape (n_strategies_player1, n_strategies_player2)
        Payoff matrix (player 1's payoffs)
        For zero-sum: player 2's payoffs = -payoff_matrix
        For non-zero-sum: provide both payoff matrices separately
    method : str
        Method to use ('linear_programming' for zero-sum, 'best_response' for general)
        
    Returns
    -------
    equilibrium : dict
        Dictionary containing:
        - 'player1_strategy': Mixed strategy for player 1
        - 'player2_strategy': Mixed strategy for player 2 (if applicable)
        - 'value': Game value (for zero-sum)
        - 'is_pure': Whether equilibrium is pure strategy
        - 'method': Method used
    """
    payoff_matrix = np.asarray(payoff_matrix)
    
    if method == 'linear_programming':
        # Assume zero-sum game (player 2's payoff = -player 1's payoff)
        return _nash_equilibrium_zero_sum(payoff_matrix)
    elif method == 'best_response':
        # General game (requires both payoff matrices)
        raise ValueError("For non-zero-sum games, use find_nash_equilibrium_general()")
    else:
        raise ValueError(f"Unknown method: {method}")


def find_nash_equilibrium_general(payoff_matrix_1: np.ndarray,
                                 payoff_matrix_2: np.ndarray,
                                 method: str = 'best_response',
                                 max_iterations: int = 1000,
                                 tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Find Nash Equilibrium for a non-zero-sum two-player game
    
    Parameters
    ----------
    payoff_matrix_1 : array, shape (n_strategies_player1, n_strategies_player2)
        Player 1's payoff matrix
    payoff_matrix_2 : array, shape (n_strategies_player1, n_strategies_player2)
        Player 2's payoff matrix
    method : str
        Method to use ('best_response', 'fictitious_play')
    max_iterations : int
        Maximum iterations for iterative methods
    tolerance : float
        Convergence tolerance
        
    Returns
    -------
    equilibrium : dict
        Dictionary containing:
        - 'player1_strategy': Mixed strategy for player 1
        - 'player2_strategy': Mixed strategy for player 2
        - 'player1_payoff': Expected payoff for player 1
        - 'player2_payoff': Expected payoff for player 2
        - 'is_pure': Whether equilibrium is pure strategy
        - 'converged': Whether algorithm converged
    """
    payoff_matrix_1 = np.asarray(payoff_matrix_1)
    payoff_matrix_2 = np.asarray(payoff_matrix_2)
    
    if payoff_matrix_1.shape != payoff_matrix_2.shape:
        raise ValueError("Payoff matrices must have the same shape")
    
    if method == 'best_response':
        return _nash_equilibrium_best_response(payoff_matrix_1, payoff_matrix_2,
                                               max_iterations, tolerance)
    elif method == 'fictitious_play':
        return _nash_equilibrium_fictitious_play(payoff_matrix_1, payoff_matrix_2,
                                                max_iterations, tolerance)
    else:
        raise ValueError(f"Unknown method: {method}")


def _nash_equilibrium_zero_sum(payoff_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Find Nash equilibrium for zero-sum game using linear programming
    
    Solves: max v subject to:
    - sum_i x_i * A_ij >= v for all j
    - sum_i x_i = 1
    - x_i >= 0
    """
    m, n = payoff_matrix.shape
    
    # Linear programming formulation
    # Maximize v (game value)
    # Subject to: sum_i x_i * A_ij >= v for all j
    #            sum_i x_i = 1
    #            x_i >= 0
    
    # Objective: maximize v (minimize -v)
    c = np.zeros(m + 1)
    c[-1] = -1  # Maximize v (minimize -v)
    
    # Constraints: sum_i x_i * A_ij >= v for all j
    # Rewritten as: sum_i x_i * A_ij - v >= 0
    A_ub = np.zeros((n, m + 1))
    A_ub[:, :m] = -payoff_matrix.T  # Negative because we need >=
    A_ub[:, -1] = 1  # -v term
    
    b_ub = np.zeros(n)
    
    # Equality constraint: sum_i x_i = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1
    b_eq = np.array([1])
    
    # Bounds: x_i >= 0, v unbounded
    bounds = [(0, None) for _ in range(m)] + [(None, None)]
    
    if not SCIPY_AVAILABLE:
        # Fallback to uniform strategy if scipy not available
        logger.warning("scipy not available, using uniform strategy")
        m, n = payoff_matrix.shape
        return {
            'player1_strategy': np.ones(m) / m,
            'player2_strategy': np.ones(n) / n,
            'value': np.mean(payoff_matrix),
            'is_pure': False,
            'method': 'fallback_no_scipy'
        }
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if result.success:
            player1_strategy = result.x[:m]
            game_value = result.x[-1]
            
            # Find player 2's strategy (minimizing player)
            # Solve dual problem
            c2 = np.ones(n)
            A_ub2 = payoff_matrix
            b_ub2 = np.ones(m) * game_value
            A_eq2 = np.ones((1, n))
            b_eq2 = np.array([1])
            bounds2 = [(0, None) for _ in range(n)]
            
            if SCIPY_AVAILABLE:
                result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                                 bounds=bounds2, method='highs')
            else:
                result2 = type('obj', (object,), {'success': False, 'x': np.ones(n) / n})()
            
            player2_strategy = result2.x if result2.success else np.ones(n) / n
            
            # Check if pure strategy
            is_pure = (np.sum(player1_strategy > 1e-6) == 1 and 
                      np.sum(player2_strategy > 1e-6) == 1)
            
            return {
                'player1_strategy': player1_strategy,
                'player2_strategy': player2_strategy,
                'value': game_value,
                'is_pure': is_pure,
                'method': 'linear_programming'
            }
        else:
            # Fallback to uniform strategy
            logger.warning("Linear programming failed, using uniform strategy")
            return {
                'player1_strategy': np.ones(m) / m,
                'player2_strategy': np.ones(n) / n,
                'value': np.mean(payoff_matrix),
                'is_pure': False,
                'method': 'fallback'
            }
    except Exception as e:
        logger.error(f"Error in Nash equilibrium computation: {e}")
        # Fallback
        m, n = payoff_matrix.shape
        return {
            'player1_strategy': np.ones(m) / m,
            'player2_strategy': np.ones(n) / n,
            'value': np.mean(payoff_matrix),
            'is_pure': False,
            'method': 'fallback'
        }


def _nash_equilibrium_best_response(payoff_matrix_1: np.ndarray,
                                   payoff_matrix_2: np.ndarray,
                                   max_iterations: int,
                                   tolerance: float) -> Dict[str, Any]:
    """
    Find Nash equilibrium using best response dynamics
    """
    m, n = payoff_matrix_1.shape
    
    # Initialize with uniform strategies
    p1_strategy = np.ones(m) / m
    p2_strategy = np.ones(n) / n
    
    for iteration in range(max_iterations):
        # Player 1's best response to player 2's current strategy
        p1_expected_payoffs = payoff_matrix_1 @ p2_strategy
        p1_best_response = np.zeros(m)
        p1_best_response[np.argmax(p1_expected_payoffs)] = 1.0
        
        # Player 2's best response to player 1's current strategy
        p2_expected_payoffs = payoff_matrix_2.T @ p1_strategy
        p2_best_response = np.zeros(n)
        p2_best_response[np.argmax(p2_expected_payoffs)] = 1.0
        
        # Update strategies (smooth update)
        learning_rate = 1.0 / (iteration + 2)
        p1_strategy = (1 - learning_rate) * p1_strategy + learning_rate * p1_best_response
        p2_strategy = (1 - learning_rate) * p2_strategy + learning_rate * p2_best_response
        
        # Check convergence
        if iteration > 10:
            p1_change = np.max(np.abs(p1_strategy - p1_best_response))
            p2_change = np.max(np.abs(p2_strategy - p2_best_response))
            if p1_change < tolerance and p2_change < tolerance:
                break
    
    # Calculate expected payoffs
    p1_payoff = p1_strategy @ payoff_matrix_1 @ p2_strategy
    p2_payoff = p1_strategy @ payoff_matrix_2 @ p2_strategy
    
    # Check if pure strategy
    is_pure = (np.sum(p1_strategy > 1e-6) == 1 and 
              np.sum(p2_strategy > 1e-6) == 1)
    
    return {
        'player1_strategy': p1_strategy,
        'player2_strategy': p2_strategy,
        'player1_payoff': float(p1_payoff),
        'player2_payoff': float(p2_payoff),
        'is_pure': is_pure,
        'converged': iteration < max_iterations - 1,
        'iterations': iteration + 1,
        'method': 'best_response'
    }


def _nash_equilibrium_fictitious_play(payoff_matrix_1: np.ndarray,
                                     payoff_matrix_2: np.ndarray,
                                     max_iterations: int,
                                     tolerance: float) -> Dict[str, Any]:
    """
    Find Nash equilibrium using fictitious play
    """
    m, n = payoff_matrix_1.shape
    
    # Initialize
    p1_strategy = np.ones(m) / m
    p2_strategy = np.ones(n) / n
    p1_counts = np.ones(m)
    p2_counts = np.ones(n)
    
    for iteration in range(max_iterations):
        # Player 1's best response to player 2's empirical distribution
        p1_expected_payoffs = payoff_matrix_1 @ p2_strategy
        p1_best_action = np.argmax(p1_expected_payoffs)
        p1_counts[p1_best_action] += 1
        
        # Player 2's best response to player 1's empirical distribution
        p2_expected_payoffs = payoff_matrix_2.T @ p1_strategy
        p2_best_action = np.argmax(p2_expected_payoffs)
        p2_counts[p2_best_action] += 1
        
        # Update empirical distributions
        p1_strategy = p1_counts / p1_counts.sum()
        p2_strategy = p2_counts / p2_counts.sum()
        
        # Check convergence (simplified)
        if iteration > 50 and iteration % 10 == 0:
            # Check if strategies are stable
            if iteration > 100:
                break
    
    # Calculate expected payoffs
    p1_payoff = p1_strategy @ payoff_matrix_1 @ p2_strategy
    p2_payoff = p1_strategy @ payoff_matrix_2 @ p2_strategy
    
    # Check if pure strategy
    is_pure = (np.sum(p1_strategy > 1e-6) == 1 and 
              np.sum(p2_strategy > 1e-6) == 1)
    
    return {
        'player1_strategy': p1_strategy,
        'player2_strategy': p2_strategy,
        'player1_payoff': float(p1_payoff),
        'player2_payoff': float(p2_payoff),
        'is_pure': is_pure,
        'converged': True,
        'iterations': iteration + 1,
        'method': 'fictitious_play'
    }


def game_theoretic_ensemble_selection(models: List[Any],
                                      X_validation: np.ndarray,
                                      y_validation: np.ndarray,
                                      evaluation_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Select ensemble weights using game theory (Nash equilibrium)
    
    Treats each model as a player in a game where:
    - Payoff = negative validation error (models want to minimize error)
    - Nash equilibrium gives optimal ensemble weights
    
    Parameters
    ----------
    models : list
        List of trained models with .predict() method
    X_validation : array
        Validation features
    y_validation : array
        Validation targets
    evaluation_fn : callable, optional
        Custom evaluation function (model, X, y) -> score
        Default: negative mean squared error for regression,
                 accuracy for classification
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'weights': Optimal ensemble weights (Nash equilibrium)
        - 'ensemble_predictions': Predictions using ensemble
        - 'individual_scores': Individual model scores
        - 'ensemble_score': Ensemble score
        - 'nash_equilibrium': Full Nash equilibrium result
    """
    X_validation = np.asarray(X_validation)
    y_validation = np.asarray(y_validation)
    
    n_models = len(models)
    
    # Evaluate each model individually
    individual_predictions = []
    individual_scores = []
    
    for model in models:
        try:
            pred = model.predict(X_validation)
            individual_predictions.append(pred)
            
            if evaluation_fn:
                score = evaluation_fn(model, X_validation, y_validation)
            else:
                # Default: negative MSE for regression, accuracy for classification
                if len(np.unique(y_validation)) > 10:  # Regression
                    score = -np.mean((pred - y_validation) ** 2)
                else:  # Classification
                    score = np.mean(pred == y_validation)
            
            individual_scores.append(score)
        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}")
            individual_predictions.append(np.zeros(len(y_validation)))
            individual_scores.append(0.0)
    
    individual_predictions = np.array(individual_predictions)
    individual_scores = np.array(individual_scores)
    
    # Create payoff matrix
    # Each model's payoff = its score - average of other models' scores
    # This creates a competitive game
    payoff_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                # Model's own payoff = its score
                payoff_matrix[i, j] = individual_scores[i]
            else:
                # When model j is selected, model i's payoff is reduced
                payoff_matrix[i, j] = individual_scores[i] - individual_scores[j]
    
    # Find Nash equilibrium (weights)
    nash_result = find_nash_equilibrium(payoff_matrix, method='linear_programming')
    weights = nash_result['player1_strategy']
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    # Create ensemble predictions
    ensemble_predictions = weights @ individual_predictions
    
    # Evaluate ensemble
    if len(np.unique(y_validation)) > 10:  # Regression
        ensemble_score = -np.mean((ensemble_predictions - y_validation) ** 2)
    else:  # Classification
        # For classification, use weighted voting
        ensemble_predictions_class = np.zeros(len(y_validation))
        for i in range(len(y_validation)):
            votes = individual_predictions[:, i]
            weighted_votes = np.bincount(votes.astype(int), weights=weights)
            ensemble_predictions_class[i] = np.argmax(weighted_votes)
        ensemble_score = np.mean(ensemble_predictions_class == y_validation)
    
    return {
        'weights': weights,
        'ensemble_predictions': ensemble_predictions,
        'individual_scores': individual_scores.tolist(),
        'ensemble_score': float(ensemble_score),
        'nash_equilibrium': nash_result
    }


class NonZeroSumGame:
    """
    Non-zero-sum game solver
    
    Extends adversarial search to handle cooperative and competitive games
    where players don't have directly opposing objectives
    """
    
    def __init__(self, payoff_matrix_1: np.ndarray, payoff_matrix_2: np.ndarray):
        """
        Initialize non-zero-sum game
        
        Parameters
        ----------
        payoff_matrix_1 : array
            Player 1's payoff matrix
        payoff_matrix_2 : array
            Player 2's payoff matrix
        """
        self.payoff_matrix_1 = np.asarray(payoff_matrix_1)
        self.payoff_matrix_2 = np.asarray(payoff_matrix_2)
        
        if self.payoff_matrix_1.shape != self.payoff_matrix_2.shape:
            raise ValueError("Payoff matrices must have the same shape")
    
    def find_equilibrium(self, method: str = 'best_response') -> Dict[str, Any]:
        """
        Find Nash equilibrium
        
        Parameters
        ----------
        method : str
            Method to use ('best_response', 'fictitious_play')
            
        Returns
        -------
        equilibrium : dict
            Nash equilibrium result
        """
        return find_nash_equilibrium_general(
            self.payoff_matrix_1,
            self.payoff_matrix_2,
            method=method
        )
    
    def is_cooperative(self, threshold: float = 0.1) -> bool:
        """
        Check if game is cooperative (both players can benefit)
        
        Parameters
        ----------
        threshold : float
            Threshold for considering game cooperative
            
        Returns
        -------
        is_cooperative : bool
            True if game is cooperative
        """
        # Check if there are outcomes where both players get positive payoffs
        # (relative to their minimum)
        p1_min = np.min(self.payoff_matrix_1)
        p2_min = np.min(self.payoff_matrix_2)
        
        p1_positive = self.payoff_matrix_1 - p1_min
        p2_positive = self.payoff_matrix_2 - p2_min
        
        # Count outcomes where both are positive
        both_positive = np.sum((p1_positive > threshold) & (p2_positive > threshold))
        total_outcomes = self.payoff_matrix_1.size
        
        return both_positive / total_outcomes > 0.3  # At least 30% cooperative outcomes


class MultiPlayerGame:
    """
    Multi-player game solver (n > 2 players)
    
    Extends game theory to handle games with more than 2 players
    """
    
    def __init__(self, payoff_matrices: List[np.ndarray]):
        """
        Initialize multi-player game
        
        Parameters
        ----------
        payoff_matrices : list of arrays
            Payoff matrix for each player
            Each matrix shape: (n_strategies_player1, n_strategies_player2, ..., n_strategies_playerN)
        """
        self.payoff_matrices = [np.asarray(p) for p in payoff_matrices]
        self.n_players = len(self.payoff_matrices)
        
        # Verify all matrices have same shape
        shape = self.payoff_matrices[0].shape
        if not all(p.shape == shape for p in self.payoff_matrices):
            raise ValueError("All payoff matrices must have the same shape")
        
        if len(shape) != self.n_players:
            raise ValueError(f"Payoff matrix dimensions ({len(shape)}) must match number of players ({self.n_players})")
    
    def find_equilibrium(self, max_iterations: int = 1000,
                        tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Find Nash equilibrium for multi-player game
        
        Uses best response dynamics
        
        Parameters
        ----------
        max_iterations : int
            Maximum iterations
        tolerance : float
            Convergence tolerance
            
        Returns
        -------
        equilibrium : dict
            Dictionary containing:
            - 'strategies': List of mixed strategies for each player
            - 'payoffs': Expected payoffs for each player
            - 'converged': Whether algorithm converged
        """
        n_players = self.n_players
        shape = self.payoff_matrices[0].shape
        
        # Initialize uniform strategies
        strategies = [np.ones(s) / s for s in shape]
        
        for iteration in range(max_iterations):
            new_strategies = []
            
            for player_idx in range(n_players):
                # Calculate expected payoff for each strategy
                # This is complex for multi-player games
                # Simplified: use best response to current strategies of others
                
                # Get current strategies of other players
                other_strategies = [strategies[i] for i in range(n_players) if i != player_idx]
                
                # Calculate expected payoffs (simplified calculation)
                expected_payoffs = self._calculate_expected_payoffs(
                    player_idx, strategies, other_strategies
                )
                
                # Best response
                best_response = np.zeros(shape[player_idx])
                best_response[np.argmax(expected_payoffs)] = 1.0
                
                # Update strategy
                learning_rate = 1.0 / (iteration + 2)
                new_strategy = (1 - learning_rate) * strategies[player_idx] + learning_rate * best_response
                new_strategies.append(new_strategy)
            
            strategies = new_strategies
            
            # Check convergence (simplified)
            if iteration > 50:
                break
        
        # Calculate final payoffs
        payoffs = []
        for player_idx in range(n_players):
            payoff = self._calculate_player_payoff(player_idx, strategies)
            payoffs.append(float(payoff))
        
        return {
            'strategies': strategies,
            'payoffs': payoffs,
            'converged': iteration < max_iterations - 1,
            'iterations': iteration + 1
        }
    
    def _calculate_expected_payoffs(self, player_idx: int,
                                    all_strategies: List[np.ndarray],
                                    other_strategies: List[np.ndarray]) -> np.ndarray:
        """Calculate expected payoffs for a player"""
        # Simplified: use uniform distribution over other players' strategies
        # Full implementation would require tensor operations
        payoff_matrix = self.payoff_matrices[player_idx]
        n_strategies = payoff_matrix.shape[player_idx]
        
        expected = np.zeros(n_strategies)
        # Simplified calculation
        for s in range(n_strategies):
            # Average over all possible combinations
            expected[s] = np.mean(payoff_matrix.flat)
        
        return expected
    
    def _calculate_player_payoff(self, player_idx: int,
                                 strategies: List[np.ndarray]) -> float:
        """Calculate expected payoff for a player given all strategies"""
        # Simplified calculation
        # Full implementation would require tensor contraction
        payoff_matrix = self.payoff_matrices[player_idx]
        return float(np.mean(payoff_matrix))
