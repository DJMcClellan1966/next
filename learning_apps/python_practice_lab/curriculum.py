"""
Curriculum: Python practice — Reed & Zelle style (problem decomposition, algorithms, data structures, code organization).
From reed_zelle_patterns.py at repo root.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced"]

BOOKS = [
    {"id": "decomposition", "name": "Problem Decomposition", "short": "Decomposition", "color": "#6366f1"},
    {"id": "algorithms", "name": "Algorithm Patterns", "short": "Algorithms", "color": "#14b8a6"},
    {"id": "data_structures", "name": "Data Structures & Code Org", "short": "Data & Code", "color": "#f97316"},
]

CURRICULUM: List[Dict[str, Any]] = [
    {"id": "decomp_ml", "book_id": "decomposition", "level": "basics", "title": "ML Problem Decomposition",
     "learn": "Break ML problems into: data collection, preprocessing, feature engineering, model selection, training, evaluation, deployment, monitoring.",
     "try_code": "from reed_zelle_patterns import ProblemDecomposition; ProblemDecomposition.decompose_ml_problem('classify images')",
     "try_demo": None},
    {"id": "decomp_algo", "book_id": "decomposition", "level": "intermediate", "title": "Algorithm Decomposition",
     "learn": "Decompose a function into steps (def, if, for, while, return). Reed & Zelle style.",
     "try_code": "from reed_zelle_patterns import ProblemDecomposition; ProblemDecomposition.decompose_algorithm(lambda x: x)",
     "try_demo": None},
    {"id": "div_conquer", "book_id": "algorithms", "level": "basics", "title": "Divide and Conquer",
     "learn": "Split data, recurse on halves, combine with an operation. Classic pattern from Reed & Zelle.",
     "try_code": "from reed_zelle_patterns import AlgorithmPatterns; AlgorithmPatterns.divide_and_conquer([1,2,3,4], lambda a,b: (a or 0)+(b or 0))",
     "try_demo": None},
    {"id": "greedy", "book_id": "algorithms", "level": "intermediate", "title": "Greedy Pattern",
     "learn": "Make locally optimal choices. Reed & Zelle algorithm patterns.",
     "try_code": "from reed_zelle_patterns import AlgorithmPatterns",
     "try_demo": None},
    {"id": "data_opt", "book_id": "data_structures", "level": "intermediate", "title": "Data Structure Optimizer",
     "learn": "Choose and optimize data structures for the problem. Reed & Zelle.",
     "try_code": "from reed_zelle_patterns import DataStructureOptimizer",
     "try_demo": None},
    {"id": "code_org", "book_id": "data_structures", "level": "basics", "title": "Code Organizer",
     "learn": "Organize code into logical modules and functions. Reed & Zelle code organization.",
     "try_code": "from reed_zelle_patterns import CodeOrganizer",
     "try_demo": None},
    {"id": "recursive", "book_id": "algorithms", "level": "intermediate", "title": "Recursive Solutions",
     "learn": "Base case and recursive case. Reed & Zelle recursive patterns.",
     "try_code": "from reed_zelle_patterns import RecursiveSolutions",
     "try_demo": None},
    {"id": "iterative", "book_id": "algorithms", "level": "basics", "title": "Iterative Refinement",
     "learn": "Refine solution step by step. Reed & Zelle iterative refinement.",
     "try_code": "from reed_zelle_patterns import IterativeRefinement",
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
