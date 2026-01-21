"""
Complete AI System - Built Around Quantum Kernel

A comprehensive AI system providing:
- Semantic Understanding
- Knowledge Graphs
- Intelligent Search
- Reasoning
- Learning
- Conversation

Easy to use, modular, and reusable.
"""

from .core import CompleteAISystem
from .components import (
    SemanticUnderstandingEngine,
    KnowledgeGraphBuilder,
    IntelligentSearch,
    ReasoningEngine,
    LearningSystem,
    ConversationalAI
)

__version__ = "1.0.0"
__all__ = [
    "CompleteAISystem",
    "SemanticUnderstandingEngine",
    "KnowledgeGraphBuilder",
    "IntelligentSearch",
    "ReasoningEngine",
    "LearningSystem",
    "ConversationalAI"
]
