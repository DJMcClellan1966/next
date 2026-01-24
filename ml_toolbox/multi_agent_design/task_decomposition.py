"""
Task Decomposition - Break down complex tasks for multi-agent execution

Implements:
- Hierarchical task decomposition
- Task dependency graphs
- Parallel task identification
- Task allocation strategies
"""
from typing import Dict, List, Optional, Any, Set
import logging
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TaskNode:
    """Node in task dependency graph"""
    task_id: str
    description: str
    required_capabilities: List[str]
    estimated_time: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, ready, executing, complete, failed
    assigned_agent: Optional[str] = None


class TaskDependencyGraph:
    """
    Task Dependency Graph
    
    Represents tasks and their dependencies
    """
    
    def __init__(self):
        self.tasks: Dict[str, TaskNode] = {}
        self.roots: List[str] = []  # Tasks with no dependencies
    
    def add_task(self, task: TaskNode):
        """Add task to graph"""
        self.tasks[task.task_id] = task
        
        # Update roots
        if not task.dependencies:
            if task.task_id not in self.roots:
                self.roots.append(task.task_id)
        else:
            # Remove from roots if it has dependencies
            if task.task_id in self.roots:
                self.roots.remove(task.task_id)
        
        # Update dependents
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                if task.task_id not in self.tasks[dep_id].dependents:
                    self.tasks[dep_id].dependents.append(task.task_id)
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks ready for execution (dependencies satisfied)"""
        ready = []
        
        for task_id, task in self.tasks.items():
            if task.status == "pending":
                # Check if all dependencies are complete
                deps_complete = all(
                    self.tasks[dep_id].status == "complete"
                    for dep_id in task.dependencies
                    if dep_id in self.tasks
                )
                
                if deps_complete:
                    ready.append(task_id)
        
        return ready
    
    def mark_complete(self, task_id: str):
        """Mark task as complete"""
        if task_id in self.tasks:
            self.tasks[task_id].status = "complete"
    
    def mark_failed(self, task_id: str):
        """Mark task as failed"""
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"
    
    def is_complete(self) -> bool:
        """Check if all tasks are complete"""
        return all(
            task.status in ["complete", "failed"]
            for task in self.tasks.values()
        )
    
    def get_critical_path(self) -> List[str]:
        """Get critical path (longest path)"""
        # Simple implementation - longest path by estimated time
        def longest_path(task_id: str, visited: Set[str]) -> tuple:
            if task_id in visited or task_id not in self.tasks:
                return ([], 0.0)
            
            visited.add(task_id)
            task = self.tasks[task_id]
            
            if not task.dependents:
                return ([task_id], task.estimated_time)
            
            max_path = []
            max_time = 0.0
            
            for dependent_id in task.dependents:
                path, time = longest_path(dependent_id, visited.copy())
                if time > max_time:
                    max_time = time
                    max_path = path
            
            return ([task_id] + max_path, task.estimated_time + max_time)
        
        # Find longest path from each root
        critical_path = []
        max_time = 0.0
        
        for root_id in self.roots:
            path, time = longest_path(root_id, set())
            if time > max_time:
                max_time = time
                critical_path = path
        
        return critical_path


class TaskDecomposer:
    """
    Task Decomposer - Break down complex tasks
    
    Implements various decomposition strategies
    """
    
    def __init__(self):
        self.decomposition_strategies = {}
        self._init_strategies()
    
    def _init_strategies(self):
        """Initialize decomposition strategies"""
        
        # ML Pipeline decomposition
        self.decomposition_strategies['ml_pipeline'] = [
            {'task_id': 'data_analysis', 'capabilities': ['analyze_data'], 'dependencies': []},
            {'task_id': 'preprocessing', 'capabilities': ['preprocess_data'], 'dependencies': ['data_analysis']},
            {'task_id': 'feature_engineering', 'capabilities': ['engineer_features'], 'dependencies': ['preprocessing']},
            {'task_id': 'model_selection', 'capabilities': ['select_model'], 'dependencies': ['feature_engineering']},
            {'task_id': 'training', 'capabilities': ['train_model'], 'dependencies': ['model_selection']},
            {'task_id': 'evaluation', 'capabilities': ['evaluate_model'], 'dependencies': ['training']}
        ]
        
        # Parallel decomposition
        self.decomposition_strategies['parallel'] = [
            {'task_id': 'task_1', 'capabilities': ['capability_1'], 'dependencies': []},
            {'task_id': 'task_2', 'capabilities': ['capability_2'], 'dependencies': []},
            {'task_id': 'task_3', 'capabilities': ['capability_3'], 'dependencies': []},
            {'task_id': 'merge', 'capabilities': ['merge_results'], 'dependencies': ['task_1', 'task_2', 'task_3']}
        ]
    
    def decompose(self, task_description: str, strategy: str = 'auto') -> TaskDependencyGraph:
        """
        Decompose task into subtasks
        
        Parameters
        ----------
        task_description : str
            Task description
        strategy : str
            Decomposition strategy
            
        Returns
        -------
        graph : TaskDependencyGraph
            Task dependency graph
        """
        graph = TaskDependencyGraph()
        
        # Auto-detect strategy
        if strategy == 'auto':
            strategy = self._detect_strategy(task_description)
        
        # Get strategy template
        if strategy in self.decomposition_strategies:
            template = self.decomposition_strategies[strategy]
        else:
            # Generic decomposition
            template = self._generic_decompose(task_description)
        
        # Create task nodes
        for task_info in template:
            task_node = TaskNode(
                task_id=task_info['task_id'],
                description=f"{task_description} - {task_info['task_id']}",
                required_capabilities=task_info['capabilities'],
                dependencies=task_info.get('dependencies', [])
            )
            graph.add_task(task_node)
        
        logger.info(f"[TaskDecomposer] Decomposed task into {len(graph.tasks)} subtasks using {strategy} strategy")
        return graph
    
    def _detect_strategy(self, task_description: str) -> str:
        """Auto-detect decomposition strategy"""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ['pipeline', 'workflow', 'end-to-end']):
            return 'ml_pipeline'
        elif any(word in task_lower for word in ['parallel', 'simultaneous', 'concurrent']):
            return 'parallel'
        else:
            return 'generic'
    
    def _generic_decompose(self, task_description: str) -> List[Dict]:
        """Generic task decomposition"""
        return [
            {'task_id': 'understand', 'capabilities': ['understand_task'], 'dependencies': []},
            {'task_id': 'plan', 'capabilities': ['plan_execution'], 'dependencies': ['understand']},
            {'task_id': 'execute', 'capabilities': ['execute_task'], 'dependencies': ['plan']},
            {'task_id': 'verify', 'capabilities': ['verify_result'], 'dependencies': ['execute']}
        ]
    
    def identify_parallel_tasks(self, graph: TaskDependencyGraph) -> List[List[str]]:
        """
        Identify tasks that can run in parallel
        
        Parameters
        ----------
        graph : TaskDependencyGraph
            Task dependency graph
            
        Returns
        -------
        parallel_groups : list of lists
            Groups of tasks that can run in parallel
        """
        parallel_groups = []
        completed = set()
        
        while not graph.is_complete():
            ready = graph.get_ready_tasks()
            
            if not ready:
                break
            
            # All ready tasks can run in parallel
            parallel_groups.append(ready.copy())
            
            # Mark as executing (simulated)
            for task_id in ready:
                if task_id in graph.tasks:
                    graph.tasks[task_id].status = "executing"
                    completed.add(task_id)
        
        return parallel_groups
