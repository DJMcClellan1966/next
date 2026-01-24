"""
Agent Evaluator - Evaluate Agent Performance

Implements:
- Performance metrics
- Success rate tracking
- Quality assessment
- Benchmarking
"""
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    agent_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    avg_quality_score: float = 0.0
    capabilities_used: Dict[str, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class AgentEvaluator:
    """
    Agent Evaluator
    
    Tracks and evaluates agent performance
    """
    
    def __init__(self):
        self.metrics: Dict[str, AgentMetrics] = {}  # agent_id -> metrics
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def record_task(self, agent_id: str, task_id: str, success: bool,
                   execution_time: float, quality_score: Optional[float] = None,
                   error: Optional[str] = None, capabilities_used: Optional[List[str]] = None):
        """
        Record task execution
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
        task_id : str
            Task identifier
        success : bool
            Whether task succeeded
        execution_time : float
            Execution time in seconds
        quality_score : float, optional
            Quality score (0-1)
        error : str, optional
            Error message if failed
        capabilities_used : list of str, optional
            Capabilities used
        """
        # Initialize metrics if needed
        if agent_id not in self.metrics:
            self.metrics[agent_id] = AgentMetrics(agent_id=agent_id)
        
        metrics = self.metrics[agent_id]
        
        # Update metrics
        metrics.total_tasks += 1
        if success:
            metrics.successful_tasks += 1
        else:
            metrics.failed_tasks += 1
            if error:
                error_type = error.split(':')[0] if ':' in error else error
                metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1
        
        # Update average execution time
        metrics.avg_execution_time = (
            (metrics.avg_execution_time * (metrics.total_tasks - 1) + execution_time) / 
            metrics.total_tasks
        )
        
        # Update average quality score
        if quality_score is not None:
            metrics.avg_quality_score = (
                (metrics.avg_quality_score * (metrics.successful_tasks - 1) + quality_score) /
                metrics.successful_tasks if metrics.successful_tasks > 0 else quality_score
            )
        
        # Track capabilities used
        if capabilities_used:
            for capability in capabilities_used:
                metrics.capabilities_used[capability] = (
                    metrics.capabilities_used.get(capability, 0) + 1
                )
        
        metrics.last_updated = datetime.now()
        
        # Store in history
        self.evaluation_history.append({
            'agent_id': agent_id,
            'task_id': task_id,
            'success': success,
            'execution_time': execution_time,
            'quality_score': quality_score,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get metrics for agent"""
        return self.metrics.get(agent_id)
    
    def get_success_rate(self, agent_id: str) -> float:
        """Get success rate for agent"""
        metrics = self.metrics.get(agent_id)
        if not metrics or metrics.total_tasks == 0:
            return 0.0
        return metrics.successful_tasks / metrics.total_tasks
    
    def compare_agents(self, agent_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple agents
        
        Parameters
        ----------
        agent_ids : list of str
            Agent identifiers to compare
            
        Returns
        -------
        comparison : dict
            Comparison results
        """
        comparison = {
            'agents': {},
            'best_performer': None,
            'most_reliable': None
        }
        
        best_success_rate = 0.0
        best_agent = None
        
        for agent_id in agent_ids:
            metrics = self.metrics.get(agent_id)
            if metrics:
                success_rate = self.get_success_rate(agent_id)
                comparison['agents'][agent_id] = {
                    'success_rate': success_rate,
                    'avg_execution_time': metrics.avg_execution_time,
                    'total_tasks': metrics.total_tasks
                }
                
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_agent = agent_id
        
        comparison['best_performer'] = best_agent
        comparison['most_reliable'] = best_agent  # Same for now
        
        return comparison
    
    def get_system_metrics(self) -> Dict:
        """Get overall system metrics"""
        if not self.metrics:
            return {'total_agents': 0}
        
        total_tasks = sum(m.total_tasks for m in self.metrics.values())
        total_successful = sum(m.successful_tasks for m in self.metrics.values())
        
        return {
            'total_agents': len(self.metrics),
            'total_tasks': total_tasks,
            'total_successful': total_successful,
            'overall_success_rate': total_successful / total_tasks if total_tasks > 0 else 0.0,
            'avg_tasks_per_agent': total_tasks / len(self.metrics) if self.metrics else 0.0
        }
