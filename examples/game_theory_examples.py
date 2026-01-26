"""
Game Theory Extensions Examples (Von Neumann)

Demonstrates:
1. Nash Equilibrium for Zero-Sum Games
2. Nash Equilibrium for Non-Zero-Sum Games
3. Game-Theoretic Ensemble Selection
4. Multi-Player Games
5. Integration with Multi-Agent Systems
"""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_toolbox.ai_concepts.game_theory import (
    find_nash_equilibrium,
    find_nash_equilibrium_general,
    game_theoretic_ensemble_selection,
    NonZeroSumGame,
    MultiPlayerGame
)
from ml_toolbox.core_models import LinearRegression, LogisticRegression, DecisionTree

print("=" * 80)
print("Game Theory Extensions Examples (Von Neumann)")
print("=" * 80)

# ============================================================================
# Example 1: Nash Equilibrium for Zero-Sum Games
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Nash Equilibrium for Zero-Sum Games")
print("=" * 80)

# Classic example: Matching Pennies
# Player 1 wants to match, Player 2 wants to mismatch
# Payoff matrix (from Player 1's perspective):
#           Player 2: H    T
# Player 1: H        [1,  -1]
#           T        [-1,  1]
payoff_matrix = np.array([
    [1, -1],   # Player 1 plays H
    [-1, 1]    # Player 1 plays T
])

print("\nPayoff Matrix (Player 1's perspective):")
print(payoff_matrix)
print("\nNote: For zero-sum, Player 2's payoff = -Player 1's payoff")

# Find Nash equilibrium
nash_result = find_nash_equilibrium(payoff_matrix, method='linear_programming')

print("\nNash Equilibrium Result:")
print(f"  Player 1 Strategy: {nash_result['player1_strategy']}")
print(f"  Player 2 Strategy: {nash_result['player2_strategy']}")
print(f"  Game Value: {nash_result['value']:.4f}")
print(f"  Is Pure Strategy: {nash_result['is_pure']}")
print(f"  Method: {nash_result['method']}")

# Expected result: Mixed strategy (0.5, 0.5) for both players

# ============================================================================
# Example 2: Nash Equilibrium for Non-Zero-Sum Games
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Nash Equilibrium for Non-Zero-Sum Games")
print("=" * 80)

# Prisoner's Dilemma
# Both players can cooperate (C) or defect (D)
# Payoff matrix 1 (Player 1):     Payoff matrix 2 (Player 2):
#           C    D                        C    D
#       C  [3,   0]                  C  [3,   5]
#       D  [5,   1]                  D  [0,   1]

payoff_matrix_1 = np.array([
    [3, 0],   # Player 1 cooperates
    [5, 1]    # Player 1 defects
])

payoff_matrix_2 = np.array([
    [3, 5],   # Player 2 cooperates
    [0, 1]    # Player 2 defects
])

print("\nPrisoner's Dilemma:")
print("Player 1 Payoff Matrix:")
print(payoff_matrix_1)
print("\nPlayer 2 Payoff Matrix:")
print(payoff_matrix_2)

# Find Nash equilibrium
nash_result_general = find_nash_equilibrium_general(
    payoff_matrix_1,
    payoff_matrix_2,
    method='best_response'
)

print("\nNash Equilibrium Result:")
print(f"  Player 1 Strategy: {nash_result_general['player1_strategy']}")
print(f"  Player 2 Strategy: {nash_result_general['player2_strategy']}")
print(f"  Player 1 Expected Payoff: {nash_result_general['player1_payoff']:.4f}")
print(f"  Player 2 Expected Payoff: {nash_result_general['player2_payoff']:.4f}")
print(f"  Is Pure Strategy: {nash_result_general['is_pure']}")
print(f"  Converged: {nash_result_general['converged']}")

# Expected: Both players defect (D) - the Nash equilibrium

# Using NonZeroSumGame class
print("\n--- Using NonZeroSumGame Class ---")
game = NonZeroSumGame(payoff_matrix_1, payoff_matrix_2)
equilibrium = game.find_equilibrium(method='best_response')
print(f"  Is Cooperative Game: {game.is_cooperative()}")
print(f"  Equilibrium: {equilibrium['player1_strategy']} vs {equilibrium['player2_strategy']}")

# ============================================================================
# Example 3: Game-Theoretic Ensemble Selection
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Game-Theoretic Ensemble Selection")
print("=" * 80)

# Generate sample data
np.random.seed(42)
n_samples = 200
n_features = 5

X_train = np.random.randn(n_samples, n_features)
y_train = (X_train[:, 0] + X_train[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

X_val = np.random.randn(50, n_features)
y_val = (X_val[:, 0] + X_val[:, 1] + np.random.randn(50) * 0.1 > 0).astype(int)

# Train multiple models
print("\nTraining multiple models...")
models = []

# Model 1: Linear Regression (as classifier)
model1 = LinearRegression()
model1.fit(X_train, y_train)
models.append(model1)

# Model 2: Logistic Regression
model2 = LogisticRegression()
model2.fit(X_train, y_train)
models.append(model2)

# Model 3: Decision Tree
model3 = DecisionTree(max_depth=5)
model3.fit(X_train, y_train)
models.append(model3)

print(f"  Trained {len(models)} models")

# Use game-theoretic ensemble selection
print("\nFinding optimal ensemble weights using Nash equilibrium...")
ensemble_result = game_theoretic_ensemble_selection(
    models,
    X_val,
    y_val
)

print("\nEnsemble Selection Result:")
print(f"  Optimal Weights: {ensemble_result['weights']}")
print(f"  Individual Model Scores: {ensemble_result['individual_scores']}")
print(f"  Ensemble Score: {ensemble_result['ensemble_score']:.4f}")
print(f"  Nash Equilibrium Method: {ensemble_result['nash_equilibrium']['method']}")

# Compare with simple averaging
simple_avg_score = np.mean(ensemble_result['individual_scores'])
print(f"\n  Simple Average Score: {simple_avg_score:.4f}")
print(f"  Game-Theoretic Ensemble Score: {ensemble_result['ensemble_score']:.4f}")
print(f"  Improvement: {ensemble_result['ensemble_score'] - simple_avg_score:.4f}")

# ============================================================================
# Example 4: Multi-Player Games
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Multi-Player Games")
print("=" * 80)

# Simple 3-player game
# Each player has 2 strategies: A or B
# Payoff matrices (3D arrays)

# Player 1's payoffs
payoff_1 = np.array([
    [[2, 1], [1, 0]],  # Player 1 plays A
    [[1, 0], [0, -1]]  # Player 1 plays B
])

# Player 2's payoffs
payoff_2 = np.array([
    [[1, 2], [0, 1]],  # Player 2 plays A
    [[0, 1], [-1, 0]]  # Player 2 plays B
])

# Player 3's payoffs
payoff_3 = np.array([
    [[1, 1], [2, 0]],  # Player 3 plays A
    [[0, 2], [1, 1]]   # Player 3 plays B
])

print("\n3-Player Game:")
print(f"  Player 1 Payoff Matrix Shape: {payoff_1.shape}")
print(f"  Player 2 Payoff Matrix Shape: {payoff_2.shape}")
print(f"  Player 3 Payoff Matrix Shape: {payoff_3.shape}")

multi_game = MultiPlayerGame([payoff_1, payoff_2, payoff_3])
multi_equilibrium = multi_game.find_equilibrium()

print("\nMulti-Player Nash Equilibrium:")
for i, strategy in enumerate(multi_equilibrium['strategies']):
    print(f"  Player {i+1} Strategy: {strategy}")
print(f"  Player Payoffs: {multi_equilibrium['payoffs']}")
print(f"  Converged: {multi_equilibrium['converged']}")

# ============================================================================
# Example 5: Integration with Multi-Agent Systems
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Integration with Multi-Agent Systems")
print("=" * 80)

# Scenario: Multiple agents need to coordinate on resource allocation
# Each agent can choose: "cooperate" (share resources) or "compete" (hoard resources)

# Agent 1's payoff matrix
agent1_payoff = np.array([
    [5, 2],   # Agent 1 cooperates
    [8, 1]    # Agent 1 competes
])

# Agent 2's payoff matrix
agent2_payoff = np.array([
    [5, 8],   # Agent 2 cooperates
    [2, 1]    # Agent 2 competes
])

print("\nMulti-Agent Coordination Game:")
print("Agents can 'cooperate' or 'compete'")
print("\nAgent 1 Payoff Matrix:")
print(agent1_payoff)
print("\nAgent 2 Payoff Matrix:")
print(agent2_payoff)

# Find Nash equilibrium for agent coordination
agent_game = NonZeroSumGame(agent1_payoff, agent2_payoff)
agent_equilibrium = agent_game.find_equilibrium()

print("\nAgent Coordination Strategy (Nash Equilibrium):")
print(f"  Agent 1 Strategy: {agent_equilibrium['player1_strategy']}")
print(f"  Agent 2 Strategy: {agent_equilibrium['player2_strategy']}")
print(f"  Agent 1 Expected Payoff: {agent_equilibrium['player1_payoff']:.4f}")
print(f"  Agent 2 Expected Payoff: {agent_equilibrium['player2_payoff']:.4f}")
print(f"  Is Cooperative Game: {agent_game.is_cooperative()}")

# Interpret strategies
if agent_equilibrium['is_pure']:
    p1_action = "cooperate" if agent_equilibrium['player1_strategy'][0] > 0.5 else "compete"
    p2_action = "cooperate" if agent_equilibrium['player2_strategy'][0] > 0.5 else "compete"
    print(f"\n  Recommended Actions:")
    print(f"    Agent 1: {p1_action}")
    print(f"    Agent 2: {p2_action}")
else:
    print(f"\n  Mixed Strategy Recommended:")
    print(f"    Agent 1: {agent_equilibrium['player1_strategy'][0]:.2%} cooperate, {agent_equilibrium['player1_strategy'][1]:.2%} compete")
    print(f"    Agent 2: {agent_equilibrium['player2_strategy'][0]:.2%} cooperate, {agent_equilibrium['player2_strategy'][1]:.2%} compete")

print("\n" + "=" * 80)
print("[OK] All Game Theory Examples Completed!")
print("=" * 80)
