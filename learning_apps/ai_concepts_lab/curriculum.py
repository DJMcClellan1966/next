"""
Curriculum: AI concepts — Russell & Norvig style (game theory, search, RL, probabilistic reasoning).
From ml_toolbox.ai_concepts.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "game_theory", "name": "Game Theory", "short": "Game Theory", "color": "#2563eb"},
    {"id": "search_planning", "name": "Search & Planning", "short": "Search & Planning", "color": "#059669"},
    {"id": "reinforcement", "name": "Reinforcement Learning", "short": "RL", "color": "#7c3aed"},
    {"id": "probabilistic", "name": "Probabilistic Reasoning", "short": "Prob. Reasoning", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    {"id": "gt_nash", "book_id": "game_theory", "level": "intermediate", "title": "Nash Equilibrium",
     "learn": "Strategy profile where no player gains by unilaterally deviating. Pure and mixed strategies.",
     "try_code": "from ml_toolbox.ai_concepts.game_theory import ...  # cooperative games, Nash",
     "try_demo": None},
    {"id": "gt_cooperative", "book_id": "game_theory", "level": "advanced", "title": "Cooperative Games",
     "learn": "Coalitions, Shapley value, core. From ml_toolbox.ai_concepts.cooperative_games.",
     "try_code": "from ml_toolbox.ai_concepts.cooperative_games import ...",
     "try_demo": None},
    {"id": "search_bfs", "book_id": "search_planning", "level": "basics", "title": "Search (BFS, DFS)",
     "learn": "State space search. BFS for shortest path; DFS for exploration. Heuristics for A*.",
     "try_code": "from ml_toolbox.ai_concepts.search_planning import ...",
     "try_demo": None},
    {"id": "rl_value", "book_id": "reinforcement", "level": "intermediate", "title": "Value Functions & Q-Learning",
     "learn": "V(s), Q(s,a). Bellman equation. Q-learning: off-policy TD control.",
     "try_code": "from ml_toolbox.ai_concepts.reinforcement_learning import ...",
     "try_demo": None},
    {"id": "prob_reasoning", "book_id": "probabilistic", "level": "intermediate", "title": "Probabilistic Reasoning",
     "learn": "Bayesian networks, inference. From ml_toolbox.ai_concepts.probabilistic_reasoning.",
     "try_code": "from ml_toolbox.ai_concepts.probabilistic_reasoning import ...",
     "try_demo": None},
    {"id": "clustering_ml", "book_id": "probabilistic", "level": "basics", "title": "Clustering (K-means, hierarchical)",
     "learn": "Unsupervised grouping. K-means, hierarchical clustering. From ai_concepts.clustering.",
     "try_code": "from ml_toolbox.ai_concepts.clustering import ...",
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
