"""
Agent Persistence - Checkpointing and Resume

Critical for long-running agents
"""
from typing import Dict, Optional, Any
import json
import os
import time
import logging

logger = logging.getLogger(__name__)


class AgentCheckpoint:
    """
    Agent Checkpoint
    
    Save and restore agent state
    """
    
    def __init__(self, checkpoint_dir: str = "agent_checkpoints"):
        """
        Initialize checkpoint system
        
        Parameters
        ----------
        checkpoint_dir : str
            Checkpoint directory
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, agent_id: str, state: Dict[str, Any]) -> str:
        """
        Save agent checkpoint
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
        state : dict
            Agent state
            
        Returns
        -------
        checkpoint_path : str
            Path to saved checkpoint
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{agent_id}_{int(time.time())}.json"
        )
        
        checkpoint_data = {
            'agent_id': agent_id,
            'timestamp': time.time(),
            'state': state
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"[AgentCheckpoint] Saved: {checkpoint_path}")
        return checkpoint_path
    
    def load(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load agent checkpoint
        
        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint
            
        Returns
        -------
        state : dict
            Agent state
        """
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        logger.info(f"[AgentCheckpoint] Loaded: {checkpoint_path}")
        return checkpoint_data.get('state', {})
    
    def get_latest(self, agent_id: str) -> Optional[str]:
        """Get latest checkpoint for agent"""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(agent_id) and f.endswith('.json')
        ]
        
        if not checkpoints:
            return None
        
        # Sort by timestamp (in filename)
        checkpoints.sort(reverse=True)
        return os.path.join(self.checkpoint_dir, checkpoints[0])


class AgentPersistence:
    """
    Agent Persistence Manager
    
    Complete persistence system for agents
    """
    
    def __init__(self, checkpoint_dir: str = "agent_checkpoints"):
        """
        Initialize persistence manager
        
        Parameters
        ----------
        checkpoint_dir : str
            Checkpoint directory
        """
        self.checkpoint = AgentCheckpoint(checkpoint_dir)
        self.agents: Dict[str, Dict] = {}
    
    def save_agent(self, agent_id: str, agent_state: Dict[str, Any]) -> str:
        """Save agent state"""
        checkpoint_path = self.checkpoint.save(agent_id, agent_state)
        self.agents[agent_id] = {
            'checkpoint_path': checkpoint_path,
            'last_saved': time.time()
        }
        return checkpoint_path
    
    def load_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent state"""
        checkpoint_path = self.checkpoint.get_latest(agent_id)
        if checkpoint_path:
            return self.checkpoint.load(checkpoint_path)
        return None
    
    def resume_agent(self, agent_id: str) -> Dict[str, Any]:
        """Resume agent from checkpoint"""
        state = self.load_agent(agent_id)
        if state:
            logger.info(f"[AgentPersistence] Resumed agent: {agent_id}")
            return state
        else:
            logger.warning(f"[AgentPersistence] No checkpoint found for: {agent_id}")
            return {}
