"""
Agent Monitoring - Cost Tracking, Rate Limiting, Metrics

Production-ready monitoring
"""
from typing import Dict, Optional, List, Any
from collections import defaultdict
import time
import logging

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Cost Tracker
    
    Track LLM API costs
    """
    
    def __init__(self):
        self.costs: Dict[str, float] = defaultdict(float)
        self.token_counts: Dict[str, int] = defaultdict(int)
        self.call_counts: Dict[str, int] = defaultdict(int)
    
    def record_call(self, provider: str, tokens_in: int, tokens_out: int,
                   cost_per_1k_in: float = 0.001, cost_per_1k_out: float = 0.002):
        """
        Record API call cost
        
        Parameters
        ----------
        provider : str
            Provider name (e.g., 'openai', 'anthropic')
        tokens_in : int
            Input tokens
        tokens_out : int
            Output tokens
        cost_per_1k_in : float
            Cost per 1K input tokens
        cost_per_1k_out : float
            Cost per 1K output tokens
        """
        cost = (tokens_in / 1000 * cost_per_1k_in) + (tokens_out / 1000 * cost_per_1k_out)
        
        self.costs[provider] += cost
        self.token_counts[provider] += tokens_in + tokens_out
        self.call_counts[provider] += 1
        
        logger.debug(f"[CostTracker] {provider}: ${cost:.4f} ({tokens_in + tokens_out} tokens)")
    
    def get_total_cost(self) -> float:
        """Get total cost"""
        return sum(self.costs.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cost statistics"""
        return {
            'total_cost': self.get_total_cost(),
            'by_provider': dict(self.costs),
            'total_tokens': sum(self.token_counts.values()),
            'total_calls': sum(self.call_counts.values())
        }


class RateLimiter:
    """
    Rate Limiter
    
    Limit agent execution rate
    """
    
    def __init__(self, max_calls_per_minute: int = 60):
        """
        Initialize rate limiter
        
        Parameters
        ----------
        max_calls_per_minute : int
            Maximum calls per minute
        """
        self.max_calls = max_calls_per_minute
        self.calls: List[float] = []
    
    def can_proceed(self) -> bool:
        """Check if call can proceed"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        
        return len(self.calls) < self.max_calls
    
    def record_call(self):
        """Record a call"""
        self.calls.append(time.time())
    
    def wait_time(self) -> float:
        """Get wait time until next call allowed"""
        if self.can_proceed():
            return 0.0
        
        # Oldest call in current window
        oldest = min(self.calls)
        return 60 - (time.time() - oldest)


class AgentMonitor:
    """
    Agent Monitor
    
    Comprehensive monitoring for agents
    """
    
    def __init__(self):
        self.cost_tracker = CostTracker()
        self.rate_limiter = RateLimiter()
        self.metrics: Dict[str, List] = defaultdict(list)
        self.execution_times: List[float] = []
    
    def track_execution(self, agent_name: str, execution_time: float, success: bool):
        """Track agent execution"""
        self.execution_times.append(execution_time)
        self.metrics[agent_name].append({
            'execution_time': execution_time,
            'success': success,
            'timestamp': time.time()
        })
    
    def track_cost(self, provider: str, tokens_in: int, tokens_out: int,
                  cost_per_1k_in: float = 0.001, cost_per_1k_out: float = 0.002):
        """Track API cost"""
        self.cost_tracker.record_call(provider, tokens_in, tokens_out,
                                     cost_per_1k_in, cost_per_1k_out)
    
    def check_rate_limit(self) -> bool:
        """Check rate limit"""
        return self.rate_limiter.can_proceed()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        avg_execution_time = (
            sum(self.execution_times) / len(self.execution_times)
            if self.execution_times else 0.0
        )
        
        return {
            'cost_stats': self.cost_tracker.get_stats(),
            'avg_execution_time': avg_execution_time,
            'total_executions': len(self.execution_times),
            'agent_metrics': {
                name: {
                    'total_calls': len(metrics),
                    'success_rate': sum(1 for m in metrics if m['success']) / len(metrics) if metrics else 0.0,
                    'avg_time': sum(m['execution_time'] for m in metrics) / len(metrics) if metrics else 0.0
                }
                for name, metrics in self.metrics.items()
            }
        }
