"""
Advanced Multi-Agent System - Complete Implementation

Combines all design patterns and best practices
"""
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Import all components
try:
    from .agent_hierarchy import AgentHierarchy, HierarchyLevel
    from .coordination_patterns import (
        CoordinatorPattern, BlackboardPattern, ContractNetPattern,
        SwarmPattern, PipelinePattern
    )
    from .task_decomposition import TaskDecomposer, TaskDependencyGraph
    from .agent_negotiation import AgentNegotiation, NegotiationProtocol
    from .distributed_execution import DistributedExecutor, ExecutionStrategy
    from .agent_monitoring import AgentMonitor, HealthStatus
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    logger.warning(f"Multi-agent design components not fully available: {e}")


class AdvancedMultiAgentSystem:
    """
    Advanced Multi-Agent System
    
    Combines:
    - Agent Hierarchies
    - Coordination Patterns
    - Task Decomposition
    - Agent Negotiation
    - Distributed Execution
    - Agent Monitoring
    """
    
    def __init__(self, coordination_pattern: str = 'coordinator'):
        """
        Initialize advanced multi-agent system
        
        Parameters
        ----------
        coordination_pattern : str
            Coordination pattern to use ('coordinator', 'blackboard', 'contract_net', 'swarm', 'pipeline')
        """
        if not COMPONENTS_AVAILABLE:
            raise ImportError("Multi-agent design components not available")
        
        # Initialize components
        self.hierarchy = AgentHierarchy()
        self.task_decomposer = TaskDecomposer()
        self.negotiation = AgentNegotiation()
        self.distributed_executor = DistributedExecutor()
        self.monitor = AgentMonitor()
        
        # Initialize coordination pattern
        self.coordination_pattern = None
        self._init_coordination_pattern(coordination_pattern)
        
        logger.info(f"[AdvancedMultiAgentSystem] Initialized with {coordination_pattern} pattern")
    
    def _init_coordination_pattern(self, pattern: str):
        """Initialize coordination pattern"""
        if pattern == 'coordinator':
            self.coordination_pattern = CoordinatorPattern("coordinator")
        elif pattern == 'blackboard':
            self.coordination_pattern = BlackboardPattern()
        elif pattern == 'contract_net':
            self.coordination_pattern = ContractNetPattern("manager")
        elif pattern == 'swarm':
            self.coordination_pattern = SwarmPattern()
        elif pattern == 'pipeline':
            self.coordination_pattern = PipelinePattern()
        else:
            self.coordination_pattern = CoordinatorPattern("coordinator")
    
    def add_agent(self, agent_id: str, agent: Any, role: str = 'worker',
                 level: HierarchyLevel = HierarchyLevel.WORKER,
                 capabilities: List[str] = None, parent_id: Optional[str] = None):
        """
        Add agent to system
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
        agent : any
            Agent instance
        role : str
            Agent role
        level : HierarchyLevel
            Hierarchy level
        capabilities : list of str
            Agent capabilities
        parent_id : str, optional
            Parent agent ID
        """
        # Add to hierarchy
        self.hierarchy.add_agent(agent_id, level, parent_id, capabilities)
        
        # Register with coordination pattern
        if isinstance(self.coordination_pattern, CoordinatorPattern):
            self.coordination_pattern.register_worker(agent_id, agent, capabilities or [])
        elif isinstance(self.coordination_pattern, BlackboardPattern):
            self.coordination_pattern.register_agent(agent_id, agent, capabilities or [])
        elif isinstance(self.coordination_pattern, ContractNetPattern):
            self.coordination_pattern.register_contractor(agent_id, agent, capabilities or [])
        elif isinstance(self.coordination_pattern, SwarmPattern):
            self.coordination_pattern.add_agent(agent)
        elif isinstance(self.coordination_pattern, PipelinePattern):
            self.coordination_pattern.add_stage(agent_id, agent)
        
        # Register with other components
        self.distributed_executor.register_agent(agent_id, agent)
        self.monitor.register_agent(agent_id, agent)
        
        logger.info(f"[AdvancedMultiAgentSystem] Added agent: {agent_id}")
    
    def execute_complex_task(self, task_description: str, 
                            context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute complex task using multi-agent system
        
        Parameters
        ----------
        task_description : str
            Task description
        context : dict, optional
            Additional context
            
        Returns
        -------
        result : dict
            Execution result
        """
        # Step 1: Decompose task
        task_graph = self.task_decomposer.decompose(task_description)
        
        # Step 2: Identify parallel tasks
        parallel_groups = self.task_decomposer.identify_parallel_tasks(task_graph)
        
        # Step 3: Allocate tasks to agents
        allocations = self._allocate_tasks(task_graph, parallel_groups)
        
        # Step 4: Execute using distributed executor
        execution_tasks = []
        for task_id, agent_id in allocations.items():
            task_node = task_graph.tasks[task_id]
            from .distributed_execution import ExecutionTask
            exec_task = ExecutionTask(
                task_id=task_id,
                agent_id=agent_id,
                action=task_node.description.split()[-1] if task_node.description else 'execute',
                parameters=context or {}
            )
            execution_tasks.append(exec_task)
        
        # Execute
        results = self.distributed_executor.execute_tasks(execution_tasks)
        
        # Step 5: Aggregate results
        aggregated = self._aggregate_results(results, task_graph)
        
        return {
            'task_description': task_description,
            'decomposition': {
                'total_tasks': len(task_graph.tasks),
                'parallel_groups': len(parallel_groups)
            },
            'execution_results': results,
            'aggregated_result': aggregated
        }
    
    def _allocate_tasks(self, task_graph: TaskDependencyGraph, 
                        parallel_groups: List[List[str]]) -> Dict[str, str]:
        """Allocate tasks to agents"""
        allocations = {}
        
        for group in parallel_groups:
            for task_id in group:
                task_node = task_graph.tasks[task_id]
                
                # Find suitable agent
                agent_id = self.hierarchy.find_agent_for_task({
                    'required_capabilities': task_node.required_capabilities
                })
                
                if agent_id:
                    allocations[task_id] = agent_id
                    task_node.assigned_agent = agent_id
        
        return allocations
    
    def _aggregate_results(self, results: Dict[str, Any], 
                          task_graph: TaskDependencyGraph) -> Dict[str, Any]:
        """Aggregate execution results"""
        successful = sum(1 for r in results.values() if 'error' not in r)
        failed = sum(1 for r in results.values() if 'error' in r)
        
        return {
            'total_tasks': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(results) if results else 0.0,
            'results': results
        }
    
    def monitor_system(self) -> Dict:
        """Monitor system health"""
        health_checks = self.monitor.check_all_agents()
        system_health = self.monitor.get_system_health()
        
        return {
            'system_health': system_health,
            'agent_health': {
                agent_id: check.status.value
                for agent_id, check in health_checks.items()
            },
            'unhealthy_agents': self.monitor.get_unhealthy_agents()
        }
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'hierarchy': self.hierarchy.get_hierarchy_stats(),
            'coordination_pattern': type(self.coordination_pattern).__name__,
            'execution_stats': self.distributed_executor.get_execution_stats(),
            'negotiation_stats': self.negotiation.get_negotiation_stats(),
            'monitoring': self.monitor.get_system_health()
        }
