"""
Multi-Agent System Design - Advanced Patterns and Best Practices

Implements:
- Agent Hierarchies
- Coordination Patterns
- Task Decomposition
- Agent Negotiation
- Swarm Intelligence
- Distributed Execution
- Agent Monitoring
- Conflict Resolution
"""
try:
    from .agent_hierarchy import AgentHierarchy, HierarchyLevel
    from .coordination_patterns import (
        CoordinatorPattern, BlackboardPattern, ContractNetPattern,
        SwarmPattern, PipelinePattern
    )
    from .task_decomposition import TaskDecomposer, TaskDependencyGraph
    from .agent_negotiation import AgentNegotiation, NegotiationProtocol
    from .distributed_execution import DistributedExecutor, ExecutionStrategy
    from .agent_monitoring import AgentMonitor, HealthCheck, AgentHealth, HealthStatus
    from .advanced_multi_agent_system import AdvancedMultiAgentSystem
    from .divine_omniscience import (
        OmniscientKnowledgeBase, OmniscientCoordinator, DivineOversight
    )
    __all__ = [
        'AgentHierarchy',
        'HierarchyLevel',
        'CoordinatorPattern',
        'BlackboardPattern',
        'ContractNetPattern',
        'SwarmPattern',
        'PipelinePattern',
        'TaskDecomposer',
        'TaskDependencyGraph',
        'AgentNegotiation',
        'NegotiationProtocol',
        'DistributedExecutor',
        'ExecutionStrategy',
        'AgentMonitor',
        'HealthCheck',
        'AgentHealth',
        'HealthStatus',
        'AdvancedMultiAgentSystem',
        'OmniscientKnowledgeBase',
        'OmniscientCoordinator',
        'DivineOversight'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Some multi-agent design components not available: {e}")
    __all__ = []
