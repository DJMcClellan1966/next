"""
Agent Compartment 1: Core

Basic agents, brain features, and fundamentals:
- Agent fundamentals (Microsoft's course)
- Brain-like features (working memory, episodic memory, etc.)
- Simple agents and loops
- Basic agent capabilities
"""
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AgentCoreCompartment:
    """
    Agent Compartment 1: Core
    
    Basic agent capabilities and brain-like features:
    - Agent fundamentals
    - Brain features (working memory, episodic memory, attention, metacognition)
    - Simple agents
    - Agent loops (ReAct, Plan-Act)
    """
    
    def __init__(self):
        self.components = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize core agent components"""
        
        # Agent Fundamentals
        try:
            from ..agent_fundamentals import (
                AgentBasics, SimpleAgent, AgentLoop, ReActLoop, PlanActLoop
            )
            self.components['AgentBasics'] = AgentBasics
            self.components['SimpleAgent'] = SimpleAgent
            self.components['AgentLoop'] = AgentLoop
            self.components['ReActLoop'] = ReActLoop
            self.components['PlanActLoop'] = PlanActLoop
        except ImportError as e:
            print(f"Warning: Agent Fundamentals not available: {e}")
        
        # Brain Features
        try:
            from ..agent_brain import (
                WorkingMemory, EpisodicMemory, SemanticMemory,
                AttentionMechanism, Metacognition, PatternAbstraction,
                CognitiveArchitecture, BrainSystem
            )
            self.components['WorkingMemory'] = WorkingMemory
            self.components['EpisodicMemory'] = EpisodicMemory
            self.components['SemanticMemory'] = SemanticMemory
            self.components['AttentionMechanism'] = AttentionMechanism
            self.components['Metacognition'] = Metacognition
            self.components['PatternAbstraction'] = PatternAbstraction
            self.components['CognitiveArchitecture'] = CognitiveArchitecture
            self.components['BrainSystem'] = BrainSystem
        except ImportError as e:
            print(f"Warning: Agent Brain features not available: {e}")
        
        # Agent Enhancements (Core features)
        try:
            from ..agent_enhancements import (
                AgentMemory, ShortTermMemory, LongTermMemory,
                AgentTool, ToolRegistry, ToolExecutor
            )
            self.components['AgentMemory'] = AgentMemory
            self.components['ShortTermMemory'] = ShortTermMemory
            self.components['LongTermMemory'] = LongTermMemory
            self.components['AgentTool'] = AgentTool
            self.components['ToolRegistry'] = ToolRegistry
            self.components['ToolExecutor'] = ToolExecutor
        except ImportError as e:
            print(f"Warning: Agent Enhancements (core) not available: {e}")
    
    def create_agent(self, name: str, system_prompt: str = "", tools: Optional[Dict] = None):
        """Create a simple agent"""
        if 'AgentBasics' in self.components:
            return self.components['AgentBasics'].create_agent(name, system_prompt, tools)
        return None
    
    def create_brain_system(self, capacity: int = 7):
        """Create brain system"""
        if 'BrainSystem' in self.components:
            return self.components['BrainSystem'](capacity)
        return None
