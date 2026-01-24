"""
Coordination Patterns - Different Multi-Agent Coordination Strategies

Implements:
- Coordinator Pattern
- Blackboard Pattern
- Contract Net Pattern
- Swarm Pattern
- Pipeline Pattern
"""
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Task for coordination"""
    task_id: str
    description: str
    required_capabilities: List[str]
    status: str = "pending"
    assigned_to: Optional[str] = None
    result: Optional[Any] = None


class CoordinatorPattern:
    """
    Coordinator Pattern - Central coordinator manages all agents
    
    Architecture:
    - One coordinator agent
    - Multiple worker agents
    - Coordinator assigns tasks to workers
    """
    
    def __init__(self, coordinator_id: str):
        """
        Initialize coordinator pattern
        
        Parameters
        ----------
        coordinator_id : str
            Coordinator agent ID
        """
        self.coordinator_id = coordinator_id
        self.workers: Dict[str, Any] = {}  # worker_id -> agent
        self.task_queue: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}  # task_id -> task
    
    def register_worker(self, worker_id: str, agent: Any, capabilities: List[str]):
        """Register worker agent"""
        self.workers[worker_id] = {
            'agent': agent,
            'capabilities': capabilities,
            'status': 'idle',
            'current_task': None
        }
        logger.info(f"[CoordinatorPattern] Registered worker: {worker_id}")
    
    def submit_task(self, task: Task) -> str:
        """Submit task to coordinator"""
        self.task_queue.append(task)
        logger.info(f"[CoordinatorPattern] Task submitted: {task.task_id}")
        return task.task_id
    
    def assign_task(self, task: Task) -> Optional[str]:
        """Assign task to worker"""
        # Find suitable worker
        for worker_id, worker_info in self.workers.items():
            if worker_info['status'] == 'idle':
                has_capabilities = all(
                    cap in worker_info['capabilities'] 
                    for cap in task.required_capabilities
                )
                if has_capabilities:
                    worker_info['status'] = 'busy'
                    worker_info['current_task'] = task.task_id
                    task.assigned_to = worker_id
                    task.status = 'assigned'
                    self.active_tasks[task.task_id] = task
                    logger.info(f"[CoordinatorPattern] Assigned {task.task_id} to {worker_id}")
                    return worker_id
        
        return None
    
    def complete_task(self, task_id: str, result: Any):
        """Mark task as complete"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = 'complete'
            task.result = result
            
            # Free worker
            if task.assigned_to in self.workers:
                self.workers[task.assigned_to]['status'] = 'idle'
                self.workers[task.assigned_to]['current_task'] = None
            
            del self.active_tasks[task_id]
            logger.info(f"[CoordinatorPattern] Task completed: {task_id}")


class BlackboardPattern:
    """
    Blackboard Pattern - Shared knowledge space
    
    Architecture:
    - Blackboard (shared data structure)
    - Multiple specialist agents
    - Agents read/write to blackboard
    - Agents collaborate through blackboard
    """
    
    def __init__(self):
        self.blackboard: Dict[str, Any] = {}
        self.agents: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
    
    def register_agent(self, agent_id: str, agent: Any, expertise: List[str]):
        """Register agent with blackboard"""
        self.agents[agent_id] = {
            'agent': agent,
            'expertise': expertise
        }
        logger.info(f"[BlackboardPattern] Registered agent: {agent_id}")
    
    def write(self, key: str, value: Any, agent_id: str):
        """Write to blackboard"""
        self.blackboard[key] = value
        self.history.append({
            'action': 'write',
            'key': key,
            'agent_id': agent_id,
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"[BlackboardPattern] {agent_id} wrote to {key}")
    
    def read(self, key: str) -> Optional[Any]:
        """Read from blackboard"""
        return self.blackboard.get(key)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all blackboard contents"""
        return self.blackboard.copy()
    
    def clear(self):
        """Clear blackboard"""
        self.blackboard.clear()
        logger.info("[BlackboardPattern] Blackboard cleared")


class ContractNetPattern:
    """
    Contract Net Pattern - Agents bid on tasks
    
    Architecture:
    - Manager announces task
    - Agents submit bids
    - Manager selects best bid
    - Winner executes task
    """
    
    def __init__(self, manager_id: str):
        self.manager_id = manager_id
        self.contractors: Dict[str, Any] = {}  # contractor_id -> agent
        self.open_contracts: Dict[str, Dict] = {}  # contract_id -> contract
        self.bids: Dict[str, List[Dict]] = {}  # contract_id -> bids
    
    def register_contractor(self, contractor_id: str, agent: Any, capabilities: List[str]):
        """Register contractor"""
        self.contractors[contractor_id] = {
            'agent': agent,
            'capabilities': capabilities
        }
        logger.info(f"[ContractNetPattern] Registered contractor: {contractor_id}")
    
    def announce_task(self, contract_id: str, task: Dict[str, Any]) -> List[str]:
        """
        Announce task (call for proposals)
        
        Parameters
        ----------
        contract_id : str
            Contract identifier
        task : dict
            Task description
            
        Returns
        -------
        notified_contractors : list of str
            Contractor IDs notified
        """
        self.open_contracts[contract_id] = {
            'task': task,
            'status': 'open',
            'announced_at': datetime.now().isoformat()
        }
        self.bids[contract_id] = []
        
        # Notify all capable contractors
        notified = []
        required_capabilities = task.get('required_capabilities', [])
        
        for contractor_id, contractor_info in self.contractors.items():
            has_capabilities = all(
                cap in contractor_info['capabilities']
                for cap in required_capabilities
            )
            if has_capabilities:
                notified.append(contractor_id)
        
        logger.info(f"[ContractNetPattern] Announced contract {contract_id} to {len(notified)} contractors")
        return notified
    
    def submit_bid(self, contract_id: str, contractor_id: str, bid: Dict[str, Any]):
        """Submit bid for contract"""
        if contract_id not in self.open_contracts:
            logger.warning(f"[ContractNetPattern] Contract not found: {contract_id}")
            return
        
        bid['contractor_id'] = contractor_id
        bid['submitted_at'] = datetime.now().isoformat()
        self.bids[contract_id].append(bid)
        logger.info(f"[ContractNetPattern] Bid submitted by {contractor_id} for {contract_id}")
    
    def award_contract(self, contract_id: str, contractor_id: str):
        """Award contract to contractor"""
        if contract_id not in self.open_contracts:
            return False
        
        contract = self.open_contracts[contract_id]
        contract['status'] = 'awarded'
        contract['awarded_to'] = contractor_id
        contract['awarded_at'] = datetime.now().isoformat()
        
        logger.info(f"[ContractNetPattern] Contract {contract_id} awarded to {contractor_id}")
        return True
    
    def get_best_bid(self, contract_id: str, criteria: str = 'lowest_cost') -> Optional[Dict]:
        """Get best bid based on criteria"""
        if contract_id not in self.bids or not self.bids[contract_id]:
            return None
        
        bids = self.bids[contract_id]
        
        if criteria == 'lowest_cost':
            return min(bids, key=lambda b: b.get('cost', float('inf')))
        elif criteria == 'fastest':
            return min(bids, key=lambda b: b.get('estimated_time', float('inf')))
        elif criteria == 'highest_quality':
            return max(bids, key=lambda b: b.get('quality_score', 0))
        else:
            return bids[0] if bids else None


class SwarmPattern:
    """
    Swarm Pattern - Decentralized agent swarm
    
    Architecture:
    - Multiple identical agents
    - No central coordination
    - Agents communicate locally
    - Emergent behavior
    """
    
    def __init__(self):
        self.swarm: List[Any] = []  # List of agents
        self.communication_radius: float = 1.0  # Communication range
    
    def add_agent(self, agent: Any):
        """Add agent to swarm"""
        self.swarm.append(agent)
        logger.info(f"[SwarmPattern] Agent added to swarm (total: {len(self.swarm)})")
    
    def broadcast_task(self, task: Dict[str, Any]) -> List[str]:
        """
        Broadcast task to swarm
        
        Parameters
        ----------
        task : dict
            Task description
            
        Returns
        -------
        participating_agents : list of str
            Agent IDs that will participate
        """
        participating = []
        
        for agent in self.swarm:
            # Each agent decides independently
            if hasattr(agent, 'agent_id'):
                participating.append(agent.agent_id)
        
        logger.info(f"[SwarmPattern] Task broadcast to {len(participating)} agents")
        return participating
    
    def get_swarm_size(self) -> int:
        """Get swarm size"""
        return len(self.swarm)
    
    def get_swarm_stats(self) -> Dict:
        """Get swarm statistics"""
        return {
            'swarm_size': len(self.swarm),
            'communication_radius': self.communication_radius
        }


class PipelinePattern:
    """
    Pipeline Pattern - Sequential agent processing
    
    Architecture:
    - Agents form a pipeline
    - Data flows through pipeline
    - Each agent processes and passes to next
    """
    
    def __init__(self):
        self.pipeline: List[Any] = []  # Ordered list of agents
        self.stage_names: List[str] = []
    
    def add_stage(self, stage_name: str, agent: Any):
        """Add stage to pipeline"""
        self.pipeline.append(agent)
        self.stage_names.append(stage_name)
        logger.info(f"[PipelinePattern] Added stage: {stage_name} (total: {len(self.pipeline)})")
    
    def execute(self, data: Any) -> Any:
        """
        Execute pipeline on data
        
        Parameters
        ----------
        data : any
            Input data
            
        Returns
        -------
        result : any
            Pipeline output
        """
        current_data = data
        
        for i, agent in enumerate(self.pipeline):
            stage_name = self.stage_names[i] if i < len(self.stage_names) else f"stage_{i}"
            logger.info(f"[PipelinePattern] Processing stage {i+1}/{len(self.pipeline)}: {stage_name}")
            
            try:
                # Execute agent (assumes agent has process or execute method)
                if hasattr(agent, 'process'):
                    current_data = agent.process(current_data)
                elif hasattr(agent, 'execute'):
                    current_data = agent.execute(current_data)
                else:
                    logger.warning(f"[PipelinePattern] Agent at stage {i} has no process/execute method")
            except Exception as e:
                logger.error(f"[PipelinePattern] Stage {i} failed: {e}")
                raise
        
        return current_data
    
    def get_pipeline_info(self) -> Dict:
        """Get pipeline information"""
        return {
            'num_stages': len(self.pipeline),
            'stages': self.stage_names
        }
