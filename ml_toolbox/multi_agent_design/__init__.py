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
from .agent_hierarchy import AgentHierarchy, HierarchyLevel
from .coordination_patterns import (
    CoordinatorPattern, BlackboardPattern, ContractNetPattern,
    SwarmPattern, PipelinePattern
)
from .task_decomposition import TaskDecomposer, TaskDependencyGraph
from .agent_negotiation import AgentNegotiation, NegotiationProtocol
from .distributed_execution import DistributedExecutor, ExecutionStrategy
from .agent_monitoring import AgentMonitor, HealthCheck, AgentHealth

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
    'AgentHealth'
]
