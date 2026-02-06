"""
Curriculum: Cross-domain "unusual" concepts — quantum, statistical mechanics, linguistics, precognition, self-organization.
From ml_toolbox.textbook_concepts.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "quantum", "name": "Quantum Mechanics", "short": "Quantum", "color": "#8b5cf6"},
    {"id": "stat_mech", "name": "Statistical Mechanics", "short": "Stat. Mech.", "color": "#0ea5e9"},
    {"id": "linguistics", "name": "Linguistics", "short": "Linguistics", "color": "#10b981"},
    {"id": "precognition", "name": "Precognition & Failure Modes", "short": "Precognition", "color": "#f59e0b"},
    {"id": "self_org", "name": "Self-Organization", "short": "Self-Org", "color": "#ec4899"},
]

CURRICULUM: List[Dict[str, Any]] = [
    {"id": "qm_basics", "book_id": "quantum", "level": "intermediate", "title": "Quantum-inspired ML",
     "learn": "Quantum mechanics concepts applied to ML: superposition, measurement. From textbook_concepts.quantum_mechanics.",
     "try_code": "from ml_toolbox.textbook_concepts.quantum_mechanics import ...",
     "try_demo": None},
    {"id": "sa", "book_id": "stat_mech", "level": "intermediate", "title": "Simulated Annealing",
     "learn": "Optimization inspired by cooling: temperature schedule, acceptance probability. From textbook_concepts.statistical_mechanics.",
     "try_code": "from ml_toolbox.textbook_concepts.statistical_mechanics import SimulatedAnnealing",
     "try_demo": None},
    {"id": "ling_parser", "book_id": "linguistics", "level": "basics", "title": "Syntactic Parsing & Grammar",
     "learn": "Simple syntactic parser and grammar-based feature extraction for text. From textbook_concepts.linguistics.",
     "try_code": "from ml_toolbox.textbook_concepts.linguistics import SimpleSyntacticParser, GrammarBasedFeatureExtractor",
     "try_demo": None},
    {"id": "precog", "book_id": "precognition", "level": "advanced", "title": "Precognitive Forecaster",
     "learn": "Failure-mode aware forecasting. From textbook_concepts.precognition.",
     "try_code": "from ml_toolbox.textbook_concepts.precognition import PrecognitiveForecaster",
     "try_demo": None},
    {"id": "som", "book_id": "self_org", "level": "intermediate", "title": "Self-Organizing Map",
     "learn": "Unsupervised topology-preserving maps. From textbook_concepts.self_organization.",
     "try_code": "from ml_toolbox.textbook_concepts.self_organization import SelfOrganizingMap",
     "try_demo": None},
    {"id": "dissipative", "book_id": "self_org", "level": "advanced", "title": "Dissipative Structures",
     "learn": "Far-from-equilibrium dynamics. From textbook_concepts.self_organization.",
     "try_code": "from ml_toolbox.textbook_concepts.self_organization import DissipativeStructure",
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
