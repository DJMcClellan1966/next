"""
Agent Compartments - Organized Agent System

Four compartments matching ML Toolbox structure:
1. Core: Basic agents, brain features, fundamentals
2. Intelligence: LLM, RAG, knowledge graphs, reasoning
3. Systems: Multi-agent, orchestration, coordination
4. Operations: Monitoring, evaluation, persistence, pipelines
"""
try:
    from .compartment1_core import AgentCoreCompartment
    from .compartment2_intelligence import AgentIntelligenceCompartment
    from .compartment3_systems import AgentSystemsCompartment
    from .compartment4_operations import AgentOperationsCompartment
    __all__ = [
        'AgentCoreCompartment',
        'AgentIntelligenceCompartment',
        'AgentSystemsCompartment',
        'AgentOperationsCompartment'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Agent Compartments not available: {e}")
    __all__ = []
