"""
Agent Hierarchy - Hierarchical Multi-Agent Systems

Implements:
- Agent hierarchies (manager-worker, supervisor-agent)
- Hierarchical task decomposition
- Hierarchical communication
"""
from typing import Dict, List, Optional, Any
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class HierarchyLevel(Enum):
    """Hierarchy levels"""
    ROOT = "root"  # Top-level manager
    MANAGER = "manager"  # Middle management
    SUPERVISOR = "supervisor"  # Direct supervision
    WORKER = "worker"  # Execution level


@dataclass
class AgentNode:
    """Node in agent hierarchy"""
    agent_id: str
    level: HierarchyLevel
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.capabilities is None:
            self.capabilities = []


class AgentHierarchy:
    """
    Agent Hierarchy - Hierarchical multi-agent system
    
    Supports:
    - Manager-worker patterns
    - Supervisor-agent patterns
    - Hierarchical task delegation
    """
    
    def __init__(self):
        self.nodes: Dict[str, AgentNode] = {}
        self.root_agent: Optional[str] = None
    
    def add_agent(self, agent_id: str, level: HierarchyLevel, 
                 parent_id: Optional[str] = None, capabilities: List[str] = None):
        """
        Add agent to hierarchy
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
        level : HierarchyLevel
            Hierarchy level
        parent_id : str, optional
            Parent agent ID
        capabilities : list of str, optional
            Agent capabilities
        """
        node = AgentNode(
            agent_id=agent_id,
            level=level,
            parent_id=parent_id,
            capabilities=capabilities or []
        )
        
        self.nodes[agent_id] = node
        
        # Set as root if it's the first root-level agent
        if level == HierarchyLevel.ROOT and self.root_agent is None:
            self.root_agent = agent_id
        
        # Update parent's children
        if parent_id and parent_id in self.nodes:
            if agent_id not in self.nodes[parent_id].children_ids:
                self.nodes[parent_id].children_ids.append(agent_id)
        
        logger.info(f"[AgentHierarchy] Added agent {agent_id} at level {level.value}")
    
    def get_children(self, agent_id: str) -> List[str]:
        """Get children of agent"""
        node = self.nodes.get(agent_id)
        return node.children_ids if node else []
    
    def get_parent(self, agent_id: str) -> Optional[str]:
        """Get parent of agent"""
        node = self.nodes.get(agent_id)
        return node.parent_id if node else None
    
    def get_ancestors(self, agent_id: str) -> List[str]:
        """Get all ancestors of agent"""
        ancestors = []
        current = agent_id
        
        while current:
            parent = self.get_parent(current)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        
        return ancestors
    
    def get_descendants(self, agent_id: str) -> List[str]:
        """Get all descendants of agent"""
        descendants = []
        children = self.get_children(agent_id)
        
        for child in children:
            descendants.append(child)
            descendants.extend(self.get_descendants(child))
        
        return descendants
    
    def delegate_task(self, from_agent: str, to_agent: str, task: Dict[str, Any]) -> bool:
        """
        Delegate task from one agent to another
        
        Parameters
        ----------
        from_agent : str
            Delegating agent
        to_agent : str
            Receiving agent
        task : dict
            Task description
            
        Returns
        -------
        can_delegate : bool
            Whether delegation is allowed
        """
        # Check if to_agent is descendant or child
        descendants = self.get_descendants(from_agent)
        children = self.get_children(from_agent)
        
        if to_agent in children or to_agent in descendants:
            logger.info(f"[AgentHierarchy] Delegating task from {from_agent} to {to_agent}")
            return True
        
        logger.warning(f"[AgentHierarchy] Cannot delegate: {to_agent} is not a subordinate of {from_agent}")
        return False
    
    def find_agent_for_task(self, task: Dict[str, Any], start_agent: Optional[str] = None) -> Optional[str]:
        """
        Find appropriate agent for task in hierarchy
        
        Parameters
        ----------
        task : dict
            Task description
        start_agent : str, optional
            Starting agent (default: root)
            
        Returns
        -------
        agent_id : str, optional
            Suitable agent ID
        """
        required_capabilities = task.get('required_capabilities', [])
        start = start_agent or self.root_agent
        
        if not start:
            return None
        
        # Search hierarchy for suitable agent
        def search(node_id: str) -> Optional[str]:
            node = self.nodes.get(node_id)
            if not node:
                return None
            
            # Check if this agent has required capabilities
            has_capabilities = all(cap in node.capabilities for cap in required_capabilities)
            if has_capabilities:
                return node_id
            
            # Search children
            for child_id in node.children_ids:
                result = search(child_id)
                if result:
                    return result
            
            return None
        
        return search(start)
    
    def get_hierarchy_stats(self) -> Dict:
        """Get hierarchy statistics"""
        return {
            'total_agents': len(self.nodes),
            'root_agent': self.root_agent,
            'levels': {
                level.value: sum(1 for n in self.nodes.values() if n.level == level)
                for level in HierarchyLevel
            }
        }
