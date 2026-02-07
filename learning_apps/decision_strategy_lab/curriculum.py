"""
Curriculum: Advanced Decision Support & Strategy Platform.
Multi-objective optimization, game theory, scenario planning,
ethical reasoning, personality-based advisors, outcome forecasting.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "optimization", "name": "Multi-Objective Optimization", "short": "Optimize", "color": "#2563eb"},
    {"id": "gametheory", "name": "Game Theory & Strategic Analysis", "short": "Game", "color": "#059669"},
    {"id": "scenarios", "name": "Scenario Planning & Forecasting", "short": "Scenario", "color": "#7c3aed"},
    {"id": "ethics", "name": "Ethical Decision Support", "short": "Ethics", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    # --- Multi-Objective Optimization ---
    {"id": "ds_pareto", "book_id": "optimization", "level": "basics",
     "title": "Pareto Optimality",
     "learn": "A solution is Pareto-optimal if no objective can be improved without worsening another. Pareto front = set of all non-dominated solutions.",
     "try_code": "# Check Pareto dominance: x dominates y if x_i <= y_i for all i and x_j < y_j for some j",
     "try_demo": "ds_pareto_front"},
    {"id": "ds_moo_weights", "book_id": "optimization", "level": "intermediate",
     "title": "Weighted Sum & ε-Constraint Methods",
     "learn": "Scalarize multi-objective problems: minimize Σ wᵢfᵢ(x). ε-constraint: optimize one, constrain others. Trade-off exploration.",
     "try_code": "# Weighted sum: score = w1*cost + w2*quality + w3*speed",
     "try_demo": "ds_weighted_optimization"},
    {"id": "ds_nsga", "book_id": "optimization", "level": "advanced",
     "title": "NSGA-II & Evolutionary Multi-Objective",
     "learn": "Non-dominated Sorting Genetic Algorithm II. Fast non-dominated sorting, crowding distance, elitism. Population-based Pareto front discovery.",
     "try_code": "# NSGA-II: sort → crowding → select → crossover → mutate",
     "try_demo": "ds_nsga2"},
    {"id": "ds_mcdm", "book_id": "optimization", "level": "expert",
     "title": "TOPSIS & Multi-Criteria Decision Making",
     "learn": "TOPSIS: rank alternatives by distance to ideal/anti-ideal solutions. AHP, ELECTRE, PROMETHEE for complex decisions.",
     "try_code": "# TOPSIS: normalize matrix → weight → ideal points → distance → rank",
     "try_demo": None},

    # --- Game Theory & Strategic Analysis ---
    {"id": "ds_nash", "book_id": "gametheory", "level": "basics",
     "title": "Nash Equilibrium",
     "learn": "Nash equilibrium: no player can improve by unilaterally changing strategy. Pure vs mixed strategies. Prisoner's Dilemma.",
     "try_code": "# Prisoner's Dilemma payoff matrix:\n# payoff[player][action_a][action_b]",
     "try_demo": "ds_nash_equilibrium"},
    {"id": "ds_shapley", "book_id": "gametheory", "level": "intermediate",
     "title": "Shapley Values & Fair Attribution",
     "learn": "Shapley value: fair distribution of total payoff among players. Average marginal contribution across all coalitions.",
     "try_code": "# Shapley: φᵢ = Σ |S|!(n-|S|-1)!/n! × [v(S∪{i}) - v(S)]",
     "try_demo": "ds_shapley_values"},
    {"id": "ds_auction", "book_id": "gametheory", "level": "advanced",
     "title": "Auction Theory & Mechanism Design",
     "learn": "First-price, second-price (Vickrey), VCG auctions. Incentive compatibility, individual rationality. Revenue equivalence theorem.",
     "try_code": "# Second-price auction: winner pays second-highest bid",
     "try_demo": "ds_auction"},
    {"id": "ds_evolve", "book_id": "gametheory", "level": "expert",
     "title": "Evolutionary Game Theory",
     "learn": "Replicator dynamics: ẋᵢ = xᵢ(fᵢ - f̄). Evolutionarily stable strategies (ESS). Hawk-Dove game.",
     "try_code": "# Replicator: population share grows if fitness > average",
     "try_demo": None},

    # --- Scenario Planning & Forecasting ---
    {"id": "ds_scenarios", "book_id": "scenarios", "level": "basics",
     "title": "Scenario Planning Fundamentals",
     "learn": "Identify key uncertainties → create plausible futures → develop robust strategies. 2×2 matrix method (two key uncertainties).",
     "try_code": "# 2x2 scenario matrix: [tech_advance x regulation] = 4 scenarios",
     "try_demo": "ds_scenario_matrix"},
    {"id": "ds_montecarlo", "book_id": "scenarios", "level": "intermediate",
     "title": "Monte Carlo Simulation",
     "learn": "Sample from uncertainty distributions → simulate outcomes → estimate risk. Confidence intervals, VaR, expected value.",
     "try_code": "import numpy as np\noutcomes = [np.random.normal(100, 20) for _ in range(10000)]",
     "try_demo": "ds_monte_carlo"},
    {"id": "ds_decision_tree", "book_id": "scenarios", "level": "advanced",
     "title": "Decision Trees & Expected Value",
     "learn": "Decision nodes (choices) → chance nodes (probabilities) → outcomes. Expected value = Σ pᵢvᵢ. Value of information.",
     "try_code": "# Decision tree: EV = p_success * payoff + p_failure * loss",
     "try_demo": "ds_decision_tree"},
    {"id": "ds_causal", "book_id": "scenarios", "level": "expert",
     "title": "Causal Inference for Decision Making",
     "learn": "Do-calculus, intervention vs observation. Counterfactuals: 'What if we had chosen differently?' Structural causal models.",
     "try_code": "# P(Y | do(X)) ≠ P(Y | X) in general",
     "try_demo": None},

    # --- Ethical Decision Support ---
    {"id": "ds_frameworks", "book_id": "ethics", "level": "basics",
     "title": "Ethical Frameworks for Decisions",
     "learn": "Consequentialism (outcomes), deontology (duties), virtue ethics (character). Applying frameworks to real decisions.",
     "try_code": "# Evaluate decision under three ethical frameworks",
     "try_demo": "ds_ethics_eval"},
    {"id": "ds_fairness", "book_id": "ethics", "level": "intermediate",
     "title": "Fairness & Bias in Decision Systems",
     "learn": "Statistical parity, equalized odds, calibration. Impossibility theorem: can't satisfy all fairness criteria simultaneously.",
     "try_code": "# Check fairness: P(Y=1|A=0) vs P(Y=1|A=1)",
     "try_demo": "ds_fairness_check"},
    {"id": "ds_stakeholder", "book_id": "ethics", "level": "advanced",
     "title": "Stakeholder Analysis",
     "learn": "Identify all affected parties. Map power vs interest. Weighted stakeholder utility. Rawlsian maximin principle.",
     "try_code": "# Stakeholders: power × interest matrix",
     "try_demo": "ds_stakeholder"},
    {"id": "ds_moral_ai", "book_id": "ethics", "level": "expert",
     "title": "AI-Assisted Moral Reasoning",
     "learn": "Moral uncertainty: weight across ethical theories. Moral parliament model. Trolley problems as preference elicitation.",
     "try_code": "# Moral parliament: utilitarian_vote + deontological_vote + virtue_vote",
     "try_demo": None},
]


def get_curriculum(): return list(CURRICULUM)
def get_books(): return list(BOOKS)
def get_levels(): return list(LEVELS)
def get_by_book(book_id: str): return [c for c in CURRICULUM if c["book_id"] == book_id]
def get_by_level(level: str): return [c for c in CURRICULUM if c["level"] == level]
def get_item(item_id: str):
    for c in CURRICULUM:
        if c["id"] == item_id: return c
    return None
