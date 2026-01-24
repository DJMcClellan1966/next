"""
Agent Core - Fundamental Agent Architecture

Implements:
- Agent state management
- Agent memory
- Agent lifecycle
- Agent capabilities
"""
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent status"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class AgentState:
    """Agent state"""
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    current_plan: Optional[Any] = None
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    last_action: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentMemory:
    """
    Agent Memory - Stores agent experiences and knowledge
    
    Types:
    - Episodic: Specific experiences
    - Semantic: General knowledge
    - Working: Current task context
    """
    episodic_memory: List[Dict[str, Any]] = field(default_factory=list)
    semantic_memory: Dict[str, Any] = field(default_factory=dict)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    
    def add_episode(self, episode: Dict[str, Any]):
        """Add episodic memory"""
        episode['timestamp'] = datetime.now().isoformat()
        self.episodic_memory.append(episode)
        # Keep last 1000 episodes
        if len(self.episodic_memory) > 1000:
            self.episodic_memory = self.episodic_memory[-1000:]
    
    def add_semantic(self, key: str, value: Any):
        """Add semantic memory"""
        self.semantic_memory[key] = value
    
    def get_semantic(self, key: str, default: Any = None) -> Any:
        """Get semantic memory"""
        return self.semantic_memory.get(key, default)
    
    def update_working(self, key: str, value: Any):
        """Update working memory"""
        self.working_memory[key] = value
    
    def clear_working(self):
        """Clear working memory"""
        self.working_memory.clear()
    
    def recall_similar(self, query: Dict[str, Any], top_k: int = 5) -> List[Dict]:
        """Recall similar episodes"""
        # Simple similarity (can be enhanced with embeddings)
        similar = []
        for episode in self.episodic_memory:
            similarity = self._calculate_similarity(query, episode)
            if similarity > 0.3:  # Threshold
                similar.append({'episode': episode, 'similarity': similarity})
        
        # Sort by similarity
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return similar[:top_k]
    
    def _calculate_similarity(self, query: Dict, episode: Dict) -> float:
        """Calculate similarity between query and episode"""
        # Simple keyword overlap
        query_keys = set(query.keys())
        episode_keys = set(episode.keys())
        
        if not query_keys or not episode_keys:
            return 0.0
        
        overlap = len(query_keys & episode_keys)
        return overlap / max(len(query_keys), len(episode_keys))


class AgentCore:
    """
    Agent Core - Fundamental agent architecture
    
    Implements:
    - State management
    - Memory system
    - Lifecycle management
    - Capability registration
    """
    
    def __init__(self, agent_id: str, name: str, description: str = ""):
        """
        Initialize agent core
        
        Parameters
        ----------
        agent_id : str
            Unique agent identifier
        name : str
            Agent name
        description : str
            Agent description
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        
        self.state = AgentState()
        self.memory = AgentMemory()
        self.capabilities = []
        self.tools = {}
        
        logger.info(f"[AgentCore] Initialized: {name} ({agent_id})")
    
    def register_capability(self, capability: str, handler: callable):
        """
        Register agent capability
        
        Parameters
        ----------
        capability : str
            Capability name
        handler : callable
            Handler function
        """
        self.capabilities.append(capability)
        self.tools[capability] = handler
        logger.info(f"[AgentCore] Registered capability: {capability}")
    
    def update_state(self, status: AgentStatus, **kwargs):
        """
        Update agent state
        
        Parameters
        ----------
        status : AgentStatus
            New status
        **kwargs
            Additional state updates
        """
        self.state.status = status
        self.state.timestamp = datetime.now()
        
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        logger.debug(f"[AgentCore] State updated: {status}")
    
    def remember(self, episode: Dict[str, Any]):
        """
        Remember an experience
        
        Parameters
        ----------
        episode : dict
            Episode to remember
        """
        self.memory.add_episode(episode)
    
    def recall(self, query: Dict[str, Any], top_k: int = 5) -> List[Dict]:
        """
        Recall similar experiences
        
        Parameters
        ----------
        query : dict
            Query for recall
        top_k : int
            Number of results
            
        Returns
        -------
        similar : list of dict
            Similar episodes
        """
        return self.memory.recall_similar(query, top_k)
    
    def learn(self, key: str, value: Any):
        """
        Learn semantic knowledge
        
        Parameters
        ----------
        key : str
            Knowledge key
        value : any
            Knowledge value
        """
        self.memory.add_semantic(key, value)
    
    def get_state(self) -> AgentState:
        """Get current state"""
        return self.state
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            'episodic_count': len(self.memory.episodic_memory),
            'semantic_count': len(self.memory.semantic_memory),
            'working_count': len(self.memory.working_memory)
        }
    
    def has_capability(self, capability: str) -> bool:
        """Check if agent has capability"""
        return capability in self.capabilities
    
    def execute_capability(self, capability: str, **kwargs) -> Any:
        """
        Execute a capability
        
        Parameters
        ----------
        capability : str
            Capability name
        **kwargs
            Arguments for capability
            
        Returns
        -------
        result : any
            Execution result
        """
        if capability not in self.tools:
            raise ValueError(f"Capability not found: {capability}")
        
        handler = self.tools[capability]
        return handler(**kwargs)
