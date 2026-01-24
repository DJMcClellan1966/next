"""
Agent Compartment 2: Intelligence

LLM, RAG, knowledge graphs, and reasoning:
- LLM agents (LLM+RAG+KG)
- RAG systems
- Knowledge graphs
- Reasoning engines
- Prompt engineering
"""
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AgentIntelligenceCompartment:
    """
    Agent Compartment 2: Intelligence
    
    LLM, RAG, knowledge graphs, and reasoning:
    - LLM agents
    - RAG systems
    - Knowledge graphs
    - Reasoning
    - Prompt engineering
    """
    
    def __init__(self):
        self.components = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize intelligence components"""
        
        # LLM+RAG+KG Agents
        try:
            from ..ai_agents import (
                LLMRAGKGAgent, KnowledgeGraphAgent, AgentBuilder
            )
            self.components['LLMRAGKGAgent'] = LLMRAGKGAgent
            self.components['KnowledgeGraphAgent'] = KnowledgeGraphAgent
            self.components['AgentBuilder'] = AgentBuilder
        except ImportError as e:
            print(f"Warning: LLM+RAG+KG Agents not available: {e}")
        
        # LLM Engineering
        try:
            from ..llm_engineering import (
                PromptEngineer, PromptTemplate, RAGSystem,
                KnowledgeRetriever, ChainOfThoughtReasoner, FewShotLearner
            )
            self.components['PromptEngineer'] = PromptEngineer
            self.components['PromptTemplate'] = PromptTemplate
            self.components['RAGSystem'] = RAGSystem
            self.components['KnowledgeRetriever'] = KnowledgeRetriever
            self.components['ChainOfThoughtReasoner'] = ChainOfThoughtReasoner
            self.components['FewShotLearner'] = FewShotLearner
        except ImportError as e:
            print(f"Warning: LLM Engineering not available: {e}")
        
        # AI Components (from infrastructure)
        try:
            from ai.components import (
                SemanticUnderstandingEngine, KnowledgeGraphBuilder,
                IntelligentSearch, ReasoningEngine
            )
            self.components['SemanticUnderstandingEngine'] = SemanticUnderstandingEngine
            self.components['KnowledgeGraphBuilder'] = KnowledgeGraphBuilder
            self.components['IntelligentSearch'] = IntelligentSearch
            self.components['ReasoningEngine'] = ReasoningEngine
        except ImportError as e:
            print(f"Warning: AI Components not available: {e}")
    
    def create_llm_rag_agent(self, toolbox=None):
        """Create LLM+RAG+KG agent"""
        if 'LLMRAGKGAgent' in self.components:
            return self.components['LLMRAGKGAgent'](toolbox=toolbox)
        return None
    
    def create_rag_system(self):
        """Create RAG system"""
        if 'RAGSystem' in self.components:
            return self.components['RAGSystem']()
        return None
