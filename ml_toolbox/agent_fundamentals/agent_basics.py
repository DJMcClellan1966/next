"""
Agent Basics - Lesson 1-3 from Microsoft's Course

Fundamentals: Simple agents, state management, basic tools
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentState:
    """Agent state container"""
    current_state: 'AgentStateEnum' = None
    
    def __post_init__(self):
        if self.current_state is None:
            self.current_state = AgentStateEnum.IDLE
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    
    def update(self, state: 'AgentStateEnum', data: Optional[Dict] = None):
        """Update agent state"""
        self.current_state = state
        if data:
            self.context.update(data)
        self.history.append({'state': state.value, 'data': data or {}})
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value"""
        return self.context.get(key, default)


class SimpleAgent:
    """
    Simple Agent - Lesson 1
    
    Basic agent with state and simple execution
    """
    
    def __init__(self, name: str = "Agent", system_prompt: str = ""):
        """
        Initialize simple agent
        
        Parameters
        ----------
        name : str
            Agent name
        system_prompt : str
            System prompt/instructions
        """
        self.name = name
        self.system_prompt = system_prompt
        self.state = AgentState()
        self.tools: Dict[str, Callable] = {}
    
    def add_tool(self, name: str, tool: Callable, description: str = ""):
        """Add tool to agent"""
        self.tools[name] = tool
        self.state.tools.append(name)
        logger.info(f"[{self.name}] Added tool: {name}")
    
    def execute(self, task: str) -> Dict[str, Any]:
        """
        Execute task
        
        Parameters
        ----------
        task : str
            Task description
            
        Returns
        -------
        result : dict
            Execution result
        """
        self.state.update(AgentState.THINKING, {'task': task})
        
        try:
            # Simple execution: parse task and use tools
            result = self._process_task(task)
            
            self.state.update(AgentState.DONE, {'result': result})
            return {
                'agent': self.name,
                'task': task,
                'result': result,
                'success': True
            }
        except Exception as e:
            self.state.update(AgentState.ERROR, {'error': str(e)})
            return {
                'agent': self.name,
                'task': task,
                'error': str(e),
                'success': False
            }
    
    def _process_task(self, task: str) -> Any:
        """Process task (simple implementation)"""
        # Check if task mentions a tool
        for tool_name, tool_func in self.tools.items():
            if tool_name.lower() in task.lower():
                return tool_func(task)
        
        # Default: return task acknowledgment
        return f"Processed: {task}"


class AgentBasics:
    """
    Agent Basics - Core concepts from Microsoft's course
    
    Provides helper methods for building agents
    """
    
    @staticmethod
    def create_agent(name: str, system_prompt: str = "", tools: Optional[Dict[str, Callable]] = None) -> SimpleAgent:
        """
        Create a simple agent
        
        Parameters
        ----------
        name : str
            Agent name
        system_prompt : str
            System prompt
        tools : dict, optional
            Tools {name: function}
            
        Returns
        -------
        agent : SimpleAgent
            Created agent
        """
        agent = SimpleAgent(name, system_prompt)
        
        if tools:
            for tool_name, tool_func in tools.items():
                agent.add_tool(tool_name, tool_func)
        
        return agent
    
    @staticmethod
    def create_agent_with_memory(name: str, system_prompt: str = "") -> 'SimpleAgent':
        """Create agent with memory (will be enhanced with AgentMemory)"""
        agent = SimpleAgent(name, system_prompt)
        # Memory will be added via AgentMemory integration
        return agent
    
    @staticmethod
    def validate_agent_state(agent: SimpleAgent) -> Dict[str, Any]:
        """Validate agent state"""
        return {
            'name': agent.name,
            'state': agent.state.current_state.value,
            'tools_count': len(agent.tools),
            'history_length': len(agent.state.history)
        }
