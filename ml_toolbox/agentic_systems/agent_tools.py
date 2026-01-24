"""
Agent Tools - Tool Registry and Management

Implements:
- Tool registration
- Tool discovery
- Tool execution
- Tool validation
"""
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass
from inspect import signature

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Agent tool"""
    tool_id: str
    name: str
    description: str
    handler: Callable
    parameters: Dict[str, Any] = None
    category: str = "general"
    
    def execute(self, **kwargs) -> Any:
        """Execute tool"""
        # Validate parameters
        sig = signature(self.handler)
        valid_params = {}
        
        for param_name, param_value in kwargs.items():
            if param_name in sig.parameters:
                valid_params[param_name] = param_value
        
        return self.handler(**valid_params)


class AgentToolRegistry:
    """
    Agent Tool Registry
    
    Manages tools available to agents
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[str, List[str]] = {}
    
    def register_tool(self, tool: Tool):
        """
        Register a tool
        
        Parameters
        ----------
        tool : Tool
            Tool to register
        """
        self.tools[tool.tool_id] = tool
        
        if tool.category not in self.categories:
            self.categories[tool.category] = []
        self.categories[tool.category].append(tool.tool_id)
        
        logger.info(f"[ToolRegistry] Registered tool: {tool.name} ({tool.tool_id})")
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get tool by ID"""
        return self.tools.get(tool_id)
    
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get tools by category"""
        tool_ids = self.categories.get(category, [])
        return [self.tools[tid] for tid in tool_ids if tid in self.tools]
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all tools"""
        return [
            {
                'tool_id': tool.tool_id,
                'name': tool.name,
                'description': tool.description,
                'category': tool.category
            }
            for tool in self.tools.values()
        ]
    
    def search_tools(self, query: str) -> List[Tool]:
        """
        Search tools by name or description
        
        Parameters
        ----------
        query : str
            Search query
            
        Returns
        -------
        tools : list of Tool
            Matching tools
        """
        query_lower = query.lower()
        matches = []
        
        for tool in self.tools.values():
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                matches.append(tool)
        
        return matches
    
    def execute_tool(self, tool_id: str, **kwargs) -> Any:
        """
        Execute a tool
        
        Parameters
        ----------
        tool_id : str
            Tool ID
        **kwargs
            Tool parameters
            
        Returns
        -------
        result : any
            Tool execution result
        """
        tool = self.get_tool(tool_id)
        if not tool:
            raise ValueError(f"Tool not found: {tool_id}")
        
        return tool.execute(**kwargs)
