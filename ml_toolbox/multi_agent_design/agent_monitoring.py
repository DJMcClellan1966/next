"""
Agent Monitoring - Monitor agent health and performance

Implements:
- Health checks
- Performance monitoring
- Failure detection
- Recovery mechanisms
"""
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result"""
    agent_id: str
    status: HealthStatus
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


@dataclass
class AgentHealth:
    """Agent health information"""
    agent_id: str
    status: HealthStatus
    last_check: datetime
    uptime: float = 0.0
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    error_count: int = 0
    total_tasks: int = 0


class AgentMonitor:
    """
    Agent Monitor - Monitor agent health and performance
    
    Tracks:
    - Agent availability
    - Performance metrics
    - Error rates
    - Response times
    """
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}  # agent_id -> agent
        self.health_records: Dict[str, List[HealthCheck]] = {}  # agent_id -> health checks
        self.health_status: Dict[str, AgentHealth] = {}  # agent_id -> current health
        self.monitoring_interval: float = 60.0  # seconds
    
    def register_agent(self, agent_id: str, agent: Any):
        """Register agent for monitoring"""
        self.agents[agent_id] = agent
        self.health_records[agent_id] = []
        self.health_status[agent_id] = AgentHealth(
            agent_id=agent_id,
            status=HealthStatus.UNKNOWN,
            last_check=datetime.now()
        )
        logger.info(f"[AgentMonitor] Registered agent for monitoring: {agent_id}")
    
    def check_health(self, agent_id: str) -> HealthCheck:
        """
        Perform health check on agent
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
            
        Returns
        -------
        health_check : HealthCheck
            Health check result
        """
        if agent_id not in self.agents:
            return HealthCheck(
                agent_id=agent_id,
                status=HealthStatus.UNKNOWN,
                issues=["Agent not registered"]
            )
        
        agent = self.agents[agent_id]
        issues = []
        metrics = {}
        
        # Check agent availability
        try:
            # Try to access agent
            if hasattr(agent, 'get_status'):
                status = agent.get_status()
                metrics['has_status'] = True
            else:
                metrics['has_status'] = False
                issues.append("Agent has no status method")
        except Exception as e:
            issues.append(f"Agent access failed: {e}")
        
        # Check agent state
        try:
            if hasattr(agent, 'core') and hasattr(agent.core, 'state'):
                state = agent.core.state.status.value
                metrics['state'] = state
                if state == 'error':
                    issues.append("Agent in error state")
        except Exception as e:
            issues.append(f"State check failed: {e}")
        
        # Determine health status
        if not issues:
            status = HealthStatus.HEALTHY
        elif len(issues) == 1:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        
        health_check = HealthCheck(
            agent_id=agent_id,
            status=status,
            metrics=metrics,
            issues=issues
        )
        
        # Store health check
        if agent_id not in self.health_records:
            self.health_records[agent_id] = []
        self.health_records[agent_id].append(health_check)
        
        # Update health status
        if agent_id in self.health_status:
            self.health_status[agent_id].status = status
            self.health_status[agent_id].last_check = datetime.now()
        
        logger.info(f"[AgentMonitor] Health check for {agent_id}: {status.value}")
        return health_check
    
    def check_all_agents(self) -> Dict[str, HealthCheck]:
        """Check health of all registered agents"""
        results = {}
        
        for agent_id in self.agents.keys():
            results[agent_id] = self.check_health(agent_id)
        
        return results
    
    def get_agent_health(self, agent_id: str) -> Optional[AgentHealth]:
        """Get current health status of agent"""
        return self.health_status.get(agent_id)
    
    def get_unhealthy_agents(self) -> List[str]:
        """Get list of unhealthy agents"""
        unhealthy = []
        
        for agent_id, health in self.health_status.items():
            if health.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
                unhealthy.append(agent_id)
        
        return unhealthy
    
    def get_system_health(self) -> Dict:
        """Get overall system health"""
        if not self.health_status:
            return {'status': 'unknown', 'total_agents': 0}
        
        total = len(self.health_status)
        healthy = sum(1 for h in self.health_status.values() if h.status == HealthStatus.HEALTHY)
        degraded = sum(1 for h in self.health_status.values() if h.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for h in self.health_status.values() if h.status == HealthStatus.UNHEALTHY)
        
        # Overall status
        if unhealthy > 0:
            overall_status = 'unhealthy'
        elif degraded > 0:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        return {
            'status': overall_status,
            'total_agents': total,
            'healthy': healthy,
            'degraded': degraded,
            'unhealthy': unhealthy,
            'health_rate': healthy / total if total > 0 else 0.0
        }
