"""
Multi-Agent System - Coordinate Multiple Agents

Implements:
- Agent roles
- Agent coordination
- Task distribution
- Agent collaboration
"""
from typing import Dict, List, Optional, Any
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles"""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    MONITOR = "monitor"


@dataclass
class AgentInfo:
    """Agent information"""
    agent_id: str
    name: str
    role: AgentRole
    capabilities: List[str]
    status: str = "idle"
    current_task: Optional[str] = None


class MultiAgentSystem:
    """
    Multi-Agent System
    
    Coordinates multiple agents to work together
    """
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}  # agent_id -> agent
        self.agent_info: Dict[str, AgentInfo] = {}
        self.task_queue: List[Dict[str, Any]] = []
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.coordinator: Optional[str] = None
        
        # Import communication system
        try:
            from .agent_communication import AgentCommunication
            self.communication = AgentCommunication()
        except ImportError:
            self.communication = None
            logger.warning("AgentCommunication not available")
    
    def register_agent(self, agent_id: str, agent: Any, name: str, 
                      role: AgentRole, capabilities: List[str]):
        """
        Register agent in system
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
        agent : any
            Agent instance
        name : str
            Agent name
        role : AgentRole
            Agent role
        capabilities : list of str
            Agent capabilities
        """
        self.agents[agent_id] = agent
        self.agent_info[agent_id] = AgentInfo(
            agent_id=agent_id,
            name=name,
            role=role,
            capabilities=capabilities
        )
        
        # Register for communication
        if self.communication:
            self.communication.register_agent(agent_id, agent)
        
        # Set first coordinator as coordinator
        if role == AgentRole.COORDINATOR and self.coordinator is None:
            self.coordinator = agent_id
        
        logger.info(f"[MultiAgentSystem] Registered agent: {name} ({agent_id}) as {role.value}")
    
    def assign_task(self, task: Dict[str, Any], agent_id: Optional[str] = None) -> str:
        """
        Assign task to agent
        
        Parameters
        ----------
        task : dict
            Task description
        agent_id : str, optional
            Specific agent ID (auto-assign if None)
            
        Returns
        -------
        task_id : str
            Task identifier
        """
        task_id = f"task_{len(self.task_queue)}"
        task['task_id'] = task_id
        
        # Auto-assign if not specified
        if agent_id is None:
            agent_id = self._select_agent_for_task(task)
        
        if agent_id:
            task['assigned_to'] = agent_id
            self.task_assignments[task_id] = agent_id
            self.agent_info[agent_id].current_task = task_id
            self.agent_info[agent_id].status = "busy"
            
            logger.info(f"[MultiAgentSystem] Assigned task {task_id} to {agent_id}")
        else:
            # Add to queue if no agent available
            self.task_queue.append(task)
            logger.info(f"[MultiAgentSystem] Task {task_id} queued (no available agent)")
        
        return task_id
    
    def _select_agent_for_task(self, task: Dict[str, Any]) -> Optional[str]:
        """Select best agent for task"""
        required_capabilities = task.get('required_capabilities', [])
        
        # Find agents with required capabilities
        candidates = []
        for agent_id, info in self.agent_info.items():
            if info.status == "idle":
                # Check if agent has required capabilities
                has_capabilities = all(cap in info.capabilities for cap in required_capabilities)
                if has_capabilities:
                    candidates.append((agent_id, info))
        
        if not candidates:
            return None
        
        # Select based on role (prefer specialists)
        specialists = [c for c in candidates if c[1].role == AgentRole.SPECIALIST]
        if specialists:
            return specialists[0][0]
        
        # Otherwise, select first available
        return candidates[0][0]
    
    def coordinate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate multi-agent task execution
        
        Parameters
        ----------
        task : dict
            Task description
            
        Returns
        -------
        result : dict
            Coordination result
        """
        # Break down task into subtasks
        subtasks = self._decompose_task(task)
        
        # Assign subtasks to agents
        assignments = {}
        for subtask in subtasks:
            agent_id = self._select_agent_for_task(subtask)
            if agent_id:
                assignments[subtask['subtask_id']] = agent_id
        
        # Execute subtasks (in production, would coordinate execution)
        results = {}
        for subtask_id, agent_id in assignments.items():
            results[subtask_id] = {
                'agent_id': agent_id,
                'status': 'assigned'
            }
        
        return {
            'task_id': task.get('task_id'),
            'subtasks': len(subtasks),
            'assignments': assignments,
            'results': results
        }
    
    def _decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose task into subtasks"""
        task_type = task.get('type', 'general')
        
        if task_type == 'ml_pipeline':
            return [
                {'subtask_id': 'data_analysis', 'required_capabilities': ['analyze_data']},
                {'subtask_id': 'preprocessing', 'required_capabilities': ['preprocess_data']},
                {'subtask_id': 'model_training', 'required_capabilities': ['train_model']},
                {'subtask_id': 'evaluation', 'required_capabilities': ['evaluate_model']}
            ]
        else:
            return [{'subtask_id': 'execute', 'required_capabilities': task.get('required_capabilities', [])}]
    
    def get_system_status(self) -> Dict:
        """Get system status"""
        return {
            'total_agents': len(self.agents),
            'idle_agents': sum(1 for info in self.agent_info.values() if info.status == "idle"),
            'busy_agents': sum(1 for info in self.agent_info.values() if info.status == "busy"),
            'queued_tasks': len(self.task_queue),
            'active_tasks': len(self.task_assignments)
        }
