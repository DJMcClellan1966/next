"""
Distributed Execution - Execute tasks across multiple agents

Implements:
- Distributed task execution
- Load balancing
- Fault tolerance
- Result aggregation
"""
from typing import Dict, List, Optional, Any
import logging
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategies"""
    SEQUENTIAL = "sequential"  # One after another
    PARALLEL = "parallel"  # All at once
    PIPELINE = "pipeline"  # Data flows through
    MAP_REDUCE = "map_reduce"  # Map then reduce
    WORK_STEALING = "work_stealing"  # Agents steal work


@dataclass
class ExecutionTask:
    """Task for distributed execution"""
    task_id: str
    agent_id: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class DistributedExecutor:
    """
    Distributed Executor - Execute tasks across agents
    
    Supports various execution strategies
    """
    
    def __init__(self, strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL):
        self.strategy = strategy
        self.agents: Dict[str, Any] = {}  # agent_id -> agent
        self.execution_history: List[Dict] = []
        self.max_workers: int = 4
    
    def register_agent(self, agent_id: str, agent: Any):
        """Register agent for execution"""
        self.agents[agent_id] = agent
        logger.info(f"[DistributedExecutor] Registered agent: {agent_id}")
    
    def execute_tasks(self, tasks: List[ExecutionTask]) -> Dict[str, Any]:
        """
        Execute tasks using selected strategy
        
        Parameters
        ----------
        tasks : list of ExecutionTask
            Tasks to execute
            
        Returns
        -------
        results : dict
            Execution results
        """
        if self.strategy == ExecutionStrategy.SEQUENTIAL:
            return self._execute_sequential(tasks)
        elif self.strategy == ExecutionStrategy.PARALLEL:
            return self._execute_parallel(tasks)
        elif self.strategy == ExecutionStrategy.PIPELINE:
            return self._execute_pipeline(tasks)
        elif self.strategy == ExecutionStrategy.MAP_REDUCE:
            return self._execute_map_reduce(tasks)
        else:
            return self._execute_sequential(tasks)
    
    def _execute_sequential(self, tasks: List[ExecutionTask]) -> Dict[str, Any]:
        """Execute tasks sequentially"""
        results = {}
        
        for task in tasks:
            if task.agent_id not in self.agents:
                results[task.task_id] = {'error': 'Agent not found'}
                continue
            
            agent = self.agents[task.agent_id]
            try:
                # Execute action
                if hasattr(agent, 'execute_action'):
                    result = agent.execute_action(task.action, **task.parameters)
                elif hasattr(agent, task.action):
                    handler = getattr(agent, task.action)
                    result = handler(**task.parameters)
                else:
                    result = {'error': f'Action not found: {task.action}'}
                
                results[task.task_id] = result
            except Exception as e:
                results[task.task_id] = {'error': str(e)}
        
        return results
    
    def _execute_parallel(self, tasks: List[ExecutionTask]) -> Dict[str, Any]:
        """Execute tasks in parallel"""
        results = {}
        
        def execute_task(task: ExecutionTask):
            if task.agent_id not in self.agents:
                return task.task_id, {'error': 'Agent not found'}
            
            agent = self.agents[task.agent_id]
            try:
                if hasattr(agent, 'execute_action'):
                    result = agent.execute_action(task.action, **task.parameters)
                elif hasattr(agent, task.action):
                    handler = getattr(agent, task.action)
                    result = handler(**task.parameters)
                else:
                    result = {'error': f'Action not found: {task.action}'}
                
                return task.task_id, result
            except Exception as e:
                return task.task_id, {'error': str(e)}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(execute_task, task): task for task in tasks}
            
            for future in as_completed(futures):
                task_id, result = future.result()
                results[task_id] = result
        
        return results
    
    def _execute_pipeline(self, tasks: List[ExecutionTask]) -> Dict[str, Any]:
        """Execute tasks as pipeline"""
        results = {}
        current_data = None
        
        for task in tasks:
            if task.agent_id not in self.agents:
                results[task.task_id] = {'error': 'Agent not found'}
                break
            
            agent = self.agents[task.agent_id]
            try:
                # Use current_data if available
                if current_data is not None:
                    task.parameters['data'] = current_data
                
                if hasattr(agent, 'execute_action'):
                    result = agent.execute_action(task.action, **task.parameters)
                elif hasattr(agent, task.action):
                    handler = getattr(agent, task.action)
                    result = handler(**task.parameters)
                else:
                    result = {'error': f'Action not found: {task.action}'}
                
                results[task.task_id] = result
                
                # Update current_data for next stage
                if 'result' in result:
                    current_data = result['result']
                elif 'data' in result:
                    current_data = result['data']
            except Exception as e:
                results[task.task_id] = {'error': str(e)}
                break
        
        return results
    
    def _execute_map_reduce(self, tasks: List[ExecutionTask]) -> Dict[str, Any]:
        """Execute using map-reduce pattern"""
        # Map phase - execute all tasks
        map_results = self._execute_parallel(tasks)
        
        # Reduce phase - aggregate results
        reduced = {
            'map_results': map_results,
            'aggregated': self._aggregate_results(map_results)
        }
        
        return reduced
    
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple tasks"""
        aggregated = {
            'total_tasks': len(results),
            'successful': sum(1 for r in results.values() if 'error' not in r),
            'failed': sum(1 for r in results.values() if 'error' in r),
            'results': results
        }
        return aggregated
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        return {
            'total_executions': len(self.execution_history),
            'registered_agents': len(self.agents),
            'strategy': self.strategy.value
        }
