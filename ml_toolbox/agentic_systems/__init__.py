"""
Agentic AI Systems - Best Practices Implementation

Based on:
- Building Agentic AI Systems: Hands-On Agent Development
- Build an AI Agent (From Scratch)

Implements:
- Agent Architecture
- Agent Memory & State
- Agent Planning & Execution
- Agent Tools & Actions
- Agent Communication
- Multi-Agent Systems
- Agent Evaluation
"""
try:
    from .agent_core import AgentCore, AgentState, AgentMemory, AgentStatus
    from .agent_planner import AgentPlanner, Plan, PlanStep, PlanStatus
    from .agent_executor import AgentExecutor, Action, ActionResult, ActionStatus
    from .agent_tools import AgentToolRegistry, Tool
    from .agent_communication import AgentCommunication, Message, MessageType
    from .multi_agent_system import MultiAgentSystem, AgentRole, AgentInfo
    from .agent_evaluator import AgentEvaluator, AgentMetrics
    from .complete_agent import CompleteAgent
    __all__ = [
        'AgentCore',
        'AgentState',
        'AgentMemory',
        'AgentStatus',
        'AgentPlanner',
        'Plan',
        'PlanStep',
        'PlanStatus',
        'AgentExecutor',
        'Action',
        'ActionResult',
        'ActionStatus',
        'AgentToolRegistry',
        'Tool',
        'AgentCommunication',
        'Message',
        'MessageType',
        'MultiAgentSystem',
        'AgentRole',
        'AgentInfo',
        'AgentEvaluator',
        'AgentMetrics',
        'CompleteAgent'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Some agentic systems components not available: {e}")
    __all__ = []
