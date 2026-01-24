"""
Agent Compartment 3: Systems

Multi-agent systems, orchestration, and coordination:
- Multi-agent systems
- Agentic systems
- Orchestration
- Coordination patterns
- Specialist agents
"""
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AgentSystemsCompartment:
    """
    Agent Compartment 3: Systems
    
    Multi-agent systems, orchestration, and coordination:
    - Multi-agent systems
    - Agentic systems
    - Orchestration
    - Coordination patterns
    - Specialist agents
    """
    
    def __init__(self):
        self.components = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize systems components"""
        
        # Super Power Agent
        try:
            from ..ai_agent import SuperPowerAgent
            self.components['SuperPowerAgent'] = SuperPowerAgent
        except ImportError as e:
            print(f"Warning: Super Power Agent not available: {e}")
        
        # Agentic Systems
        try:
            from ..agentic_systems import (
                CompleteAgent, AgentCore, AgentPlanner, AgentExecutor,
                AgentToolRegistry, AgentCommunication, MultiAgentSystem, AgentEvaluator
            )
            self.components['CompleteAgent'] = CompleteAgent
            self.components['AgentCore'] = AgentCore
            self.components['AgentPlanner'] = AgentPlanner
            self.components['AgentExecutor'] = AgentExecutor
            self.components['AgentToolRegistry'] = AgentToolRegistry
            self.components['AgentCommunication'] = AgentCommunication
            self.components['MultiAgentSystem'] = MultiAgentSystem
            self.components['AgentEvaluator'] = AgentEvaluator
        except ImportError as e:
            print(f"Warning: Agentic Systems not available: {e}")
        
        # Multi-Agent Design
        try:
            from ..multi_agent_design import (
                AdvancedMultiAgentSystem, AgentHierarchy, CoordinatorPattern,
                BlackboardPattern, ContractNetPattern, SwarmPattern, PipelinePattern
            )
            self.components['AdvancedMultiAgentSystem'] = AdvancedMultiAgentSystem
            self.components['AgentHierarchy'] = AgentHierarchy
            self.components['CoordinatorPattern'] = CoordinatorPattern
            self.components['BlackboardPattern'] = BlackboardPattern
            self.components['ContractNetPattern'] = ContractNetPattern
            self.components['SwarmPattern'] = SwarmPattern
            self.components['PipelinePattern'] = PipelinePattern
        except ImportError as e:
            print(f"Warning: Multi-Agent Design not available: {e}")
        
        # Agent Orchestrator
        try:
            from ..ai_agent import AgentOrchestrator
            self.components['AgentOrchestrator'] = AgentOrchestrator
        except ImportError as e:
            print(f"Warning: Agent Orchestrator not available: {e}")
        
        # Specialist Agents
        try:
            from ..ai_agent.specialist_agents import (
                DataAgent, FeatureAgent, ModelAgent,
                TuningAgent, DeployAgent, InsightAgent
            )
            self.components['DataAgent'] = DataAgent
            self.components['FeatureAgent'] = FeatureAgent
            self.components['ModelAgent'] = ModelAgent
            self.components['TuningAgent'] = TuningAgent
            self.components['DeployAgent'] = DeployAgent
            self.components['InsightAgent'] = InsightAgent
        except ImportError as e:
            print(f"Warning: Specialist Agents not available: {e}")
    
    def create_super_power_agent(self, toolbox=None):
        """Create Super Power Agent"""
        if 'SuperPowerAgent' in self.components:
            return self.components['SuperPowerAgent'](toolbox=toolbox)
        return None
    
    def create_complete_agent(self):
        """Create Complete Agent"""
        if 'CompleteAgent' in self.components:
            return self.components['CompleteAgent']()
        return None
