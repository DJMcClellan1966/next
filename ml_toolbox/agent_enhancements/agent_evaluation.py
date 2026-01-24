"""
Agent Evaluation - Metrics and Evaluation Framework

Essential for agent quality assurance
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    total_executions: int = 0
    error_rate: float = 0.0
    avg_cost: float = 0.0
    quality_score: float = 0.0


class AgentEvaluator:
    """
    Agent Evaluator
    
    Evaluate agent performance
    """
    
    def __init__(self):
        self.evaluations: List[Dict] = []
        self.metrics: Dict[str, AgentMetrics] = {}
    
    def evaluate(self, agent_name: str, task: str, result: Dict[str, Any],
                expected_result: Optional[Any] = None,
                evaluation_criteria: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """
        Evaluate agent execution
        
        Parameters
        ----------
        agent_name : str
            Agent name
        task : str
            Task description
        result : dict
            Agent result
        expected_result : any, optional
            Expected result for comparison
        evaluation_criteria : dict, optional
            Custom evaluation functions
            
        Returns
        -------
        evaluation : dict
            Evaluation results
        """
        evaluation = {
            'agent': agent_name,
            'task': task,
            'timestamp': time.time(),
            'success': result.get('success', False),
            'metrics': {}
        }
        
        # Execution time
        if 'execution_time' in result:
            evaluation['metrics']['execution_time'] = result['execution_time']
        
        # Accuracy (if expected result provided)
        if expected_result is not None:
            accuracy = self._calculate_accuracy(result.get('result'), expected_result)
            evaluation['metrics']['accuracy'] = accuracy
        
        # Custom criteria
        if evaluation_criteria:
            for criterion_name, criterion_func in evaluation_criteria.items():
                try:
                    score = criterion_func(result)
                    evaluation['metrics'][criterion_name] = score
                except Exception as e:
                    logger.warning(f"[AgentEvaluator] Criterion {criterion_name} failed: {e}")
        
        # Quality score
        evaluation['metrics']['quality_score'] = self._calculate_quality_score(evaluation)
        
        self.evaluations.append(evaluation)
        self._update_metrics(agent_name, evaluation)
        
        return evaluation
    
    def _calculate_accuracy(self, actual: Any, expected: Any) -> float:
        """Calculate accuracy"""
        if actual == expected:
            return 1.0
        
        # Simple string similarity
        if isinstance(actual, str) and isinstance(expected, str):
            # Simple word overlap
            actual_words = set(actual.lower().split())
            expected_words = set(expected.lower().split())
            if expected_words:
                overlap = len(actual_words & expected_words) / len(expected_words)
                return overlap
        
        return 0.0
    
    def _calculate_quality_score(self, evaluation: Dict) -> float:
        """Calculate overall quality score"""
        metrics = evaluation.get('metrics', {})
        
        score = 0.0
        factors = 0
        
        # Success factor
        if evaluation.get('success'):
            score += 1.0
        factors += 1
        
        # Accuracy factor
        if 'accuracy' in metrics:
            score += metrics['accuracy']
            factors += 1
        
        # Execution time factor (faster is better, normalized)
        if 'execution_time' in metrics:
            # Normalize: < 1s = 1.0, > 10s = 0.0
            exec_time = metrics['execution_time']
            time_score = max(0.0, 1.0 - (exec_time / 10.0))
            score += time_score
            factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def _update_metrics(self, agent_name: str, evaluation: Dict):
        """Update agent metrics"""
        if agent_name not in self.metrics:
            self.metrics[agent_name] = AgentMetrics()
        
        metrics = self.metrics[agent_name]
        metrics.total_executions += 1
        
        if evaluation.get('success'):
            metrics.success_rate = (
                (metrics.success_rate * (metrics.total_executions - 1) + 1.0) /
                metrics.total_executions
            )
        else:
            metrics.error_rate = (
                (metrics.error_rate * (metrics.total_executions - 1) + 1.0) /
                metrics.total_executions
            )
        
        if 'execution_time' in evaluation.get('metrics', {}):
            exec_time = evaluation['metrics']['execution_time']
            metrics.avg_execution_time = (
                (metrics.avg_execution_time * (metrics.total_executions - 1) + exec_time) /
                metrics.total_executions
            )
        
        if 'quality_score' in evaluation.get('metrics', {}):
            quality = evaluation['metrics']['quality_score']
            metrics.quality_score = (
                (metrics.quality_score * (metrics.total_executions - 1) + quality) /
                metrics.total_executions
            )
    
    def get_metrics(self, agent_name: Optional[str] = None) -> Dict[str, AgentMetrics]:
        """Get agent metrics"""
        if agent_name:
            return {agent_name: self.metrics.get(agent_name, AgentMetrics())}
        return dict(self.metrics)
    
    def get_report(self) -> Dict[str, Any]:
        """Get evaluation report"""
        return {
            'total_evaluations': len(self.evaluations),
            'agents': {
                name: {
                    'success_rate': m.success_rate,
                    'avg_execution_time': m.avg_execution_time,
                    'total_executions': m.total_executions,
                    'error_rate': m.error_rate,
                    'quality_score': m.quality_score
                }
                for name, m in self.metrics.items()
            }
        }
