"""
Agent Fundamentals - From Microsoft's AI Agents for Beginners

12-lesson modular course covering:
- Fundamentals → Advanced patterns
- Quick wins for core concepts
- Production-ready patterns
"""
try:
    from .agent_basics import AgentBasics, SimpleAgent, AgentStateEnum
    from .agent_loops import AgentLoop, ReActLoop, PlanActLoop
    __all__ = [
        'AgentBasics',
        'SimpleAgent',
        'AgentStateEnum',
        'AgentLoop',
        'ReActLoop',
        'PlanActLoop'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Agent Fundamentals not available: {e}")
    __all__ = []
