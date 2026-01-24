"""
Agent Tools - Tool Registry and Execution

Essential for agent capabilities
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentTool:
    """Agent tool definition"""
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = None
    returns: str = "any"
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class ToolRegistry:
    """
    Tool Registry
    
    Central registry for agent tools
    """
    
    def __init__(self):
        self.tools: Dict[str, AgentTool] = {}
    
    def register(self, tool: AgentTool):
        """Register tool"""
        self.tools[tool.name] = tool
        logger.info(f"[ToolRegistry] Registered: {tool.name}")
    
    def register_function(self, name: str, func: Callable, description: str = "",
                        parameters: Optional[Dict] = None):
        """Register function as tool"""
        tool = AgentTool(
            name=name,
            description=description or func.__doc__ or "",
            func=func,
            parameters=parameters
        )
        self.register(tool)
    
    def get_tool(self, name: str) -> Optional[AgentTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all tool names"""
        return list(self.tools.keys())
    
    def search_tools(self, query: str) -> List[AgentTool]:
        """Search tools by description"""
        query_lower = query.lower()
        results = []
        
        for tool in self.tools.values():
            if (query_lower in tool.name.lower() or
                query_lower in tool.description.lower()):
                results.append(tool)
        
        return results


class ToolExecutor:
    """
    Tool Executor
    
    Executes tools with error handling and validation
    """
    
    def __init__(self, registry: ToolRegistry):
        """
        Initialize tool executor
        
        Parameters
        ----------
        registry : ToolRegistry
            Tool registry
        """
        self.registry = registry
        self.execution_history: List[Dict] = []
    
    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute tool
        
        Parameters
        ----------
        tool_name : str
            Tool name
        **kwargs
            Tool arguments
            
        Returns
        -------
        result : dict
            Execution result
        """
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return {
                'success': False,
                'error': f'Tool not found: {tool_name}'
            }
        
        try:
            # Execute tool
            result = tool.func(**kwargs)
            
            execution_record = {
                'tool': tool_name,
                'arguments': kwargs,
                'result': result,
                'success': True
            }
            self.execution_history.append(execution_record)
            
            return {
                'success': True,
                'result': result,
                'tool': tool_name
            }
        except Exception as e:
            execution_record = {
                'tool': tool_name,
                'arguments': kwargs,
                'error': str(e),
                'success': False
            }
            self.execution_history.append(execution_record)
            
            logger.error(f"[ToolExecutor] Error executing {tool_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'tool': tool_name
            }
    
    def get_history(self, tool_name: Optional[str] = None) -> List[Dict]:
        """Get execution history"""
        if tool_name:
            return [h for h in self.execution_history if h.get('tool') == tool_name]
        return self.execution_history
