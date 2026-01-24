"""
Agent Compartment 4: Operations

Monitoring, evaluation, persistence, and pipelines:
- Agent monitoring
- Cost tracking
- Evaluation
- Persistence/checkpointing
- Pipelines
- Framework integration
- Pattern catalog
"""
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AgentOperationsCompartment:
    """
    Agent Compartment 4: Operations
    
    Monitoring, evaluation, persistence, and pipelines:
    - Monitoring and cost tracking
    - Evaluation
    - Persistence
    - Pipelines
    - Framework integration
    - Pattern catalog
    """
    
    def __init__(self):
        self.components = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize operations components"""
        
        # Agent Enhancements (Operations)
        try:
            from ..agent_enhancements import (
                AgentPersistence, AgentCheckpoint,
                AgentMonitor, CostTracker, RateLimiter,
                AgentEvaluator, AgentMetrics
            )
            self.components['AgentPersistence'] = AgentPersistence
            self.components['AgentCheckpoint'] = AgentCheckpoint
            self.components['AgentMonitor'] = AgentMonitor
            self.components['CostTracker'] = CostTracker
            self.components['RateLimiter'] = RateLimiter
            self.components['AgentEvaluator'] = AgentEvaluator
            self.components['AgentMetrics'] = AgentMetrics
        except ImportError as e:
            print(f"Warning: Agent Enhancements (operations) not available: {e}")
        
        # Agent Pipelines
        try:
            from ..agent_pipelines import (
                PromptRAGDeployPipeline, EndToEndPipeline, PipelineStage
            )
            self.components['PromptRAGDeployPipeline'] = PromptRAGDeployPipeline
            self.components['EndToEndPipeline'] = EndToEndPipeline
            self.components['PipelineStage'] = PipelineStage
        except ImportError as e:
            print(f"Warning: Agent Pipelines not available: {e}")
        
        # Framework Integration
        try:
            from ..framework_integration import (
                LangGraphAgent, StateGraph, GraphNode,
                CrewAgent, Crew, Task, Agent
            )
            self.components['LangGraphAgent'] = LangGraphAgent
            self.components['StateGraph'] = StateGraph
            self.components['GraphNode'] = GraphNode
            self.components['CrewAgent'] = CrewAgent
            self.components['Crew'] = Crew
            self.components['Task'] = Task
            self.components['Agent'] = Agent
        except ImportError as e:
            print(f"Warning: Framework Integration not available: {e}")
        
        # Generative AI Patterns
        try:
            from ..generative_ai_patterns import (
                PatternCatalog, PatternLibrary,
                PatternCompositionStrategy, PatternOrchestrator, CompositionStrategy
            )
            self.components['PatternCatalog'] = PatternCatalog
            self.components['PatternLibrary'] = PatternLibrary
            self.components['PatternCompositionStrategy'] = PatternCompositionStrategy
            self.components['PatternOrchestrator'] = PatternOrchestrator
            self.components['CompositionStrategy'] = CompositionStrategy
        except ImportError as e:
            print(f"Warning: Generative AI Patterns not available: {e}")
    
    def create_monitor(self):
        """Create agent monitor"""
        if 'AgentMonitor' in self.components:
            return self.components['AgentMonitor']()
        return None
    
    def create_pipeline(self):
        """Create end-to-end pipeline"""
        if 'EndToEndPipeline' in self.components:
            return self.components['EndToEndPipeline']()
        return None
