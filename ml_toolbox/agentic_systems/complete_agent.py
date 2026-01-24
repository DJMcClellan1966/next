"""
Complete Agent - Full Agent Implementation

Combines all agent components into a complete, production-ready agent
"""
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Import all components
try:
    from .agent_core import AgentCore, AgentState, AgentMemory, AgentStatus
    from .agent_planner import AgentPlanner, Plan, PlanStatus
    from .agent_executor import AgentExecutor, Action, ActionResult
    from .agent_tools import AgentToolRegistry, Tool
    from .agent_communication import AgentCommunication, Message, MessageType
    from .agent_evaluator import AgentEvaluator, AgentMetrics
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    logger.warning(f"Agent components not fully available: {e}")


class CompleteAgent:
    """
    Complete Agent - Full implementation
    
    Combines:
    - Agent Core (state, memory)
    - Planner (goal decomposition, planning)
    - Executor (action execution)
    - Tools (capabilities)
    - Communication (inter-agent)
    - Evaluation (performance tracking)
    """
    
    def __init__(self, agent_id: str, name: str, description: str = "", toolbox=None):
        """
        Initialize complete agent
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
        name : str
            Agent name
        description : str
            Agent description
        toolbox : MLToolbox, optional
            ML Toolbox instance
        """
        if not COMPONENTS_AVAILABLE:
            raise ImportError("Agent components not available")
        
        self.toolbox = toolbox
        
        # Initialize core components
        self.core = AgentCore(agent_id, name, description)
        self.planner = AgentPlanner()
        self.executor = AgentExecutor()
        self.tool_registry = AgentToolRegistry()
        self.evaluator = AgentEvaluator()
        
        # Communication (optional - for multi-agent)
        self.communication = None
        
        # Register default ML tools if toolbox available
        if toolbox:
            self._register_ml_tools()
        
        logger.info(f"[CompleteAgent] Initialized: {name} ({agent_id})")
    
    def _register_ml_tools(self):
        """Register ML-specific tools"""
        if not self.toolbox:
            return
        
        # Data analysis tool
        def analyze_data(data=None, **kwargs):
            if data is None:
                return {'error': 'No data provided'}
            import numpy as np
            data = np.asarray(data)
            return {
                'shape': data.shape,
                'mean': np.mean(data, axis=0).tolist() if len(data.shape) > 1 else float(np.mean(data)),
                'std': np.std(data, axis=0).tolist() if len(data.shape) > 1 else float(np.std(data))
            }
        
        # Preprocessing tool
        def preprocess_data(data=None, method='standardize', **kwargs):
            if data is None:
                return {'error': 'No data provided'}
            if self.toolbox.feature_kernel:
                import numpy as np
                data = np.asarray(data)
                return self.toolbox.feature_kernel.transform(data, operations=[method])
            return {'error': 'Feature kernel not available'}
        
        # Model training tool
        def train_model(X=None, y=None, task_type='classification', **kwargs):
            if X is None or y is None:
                return {'error': 'Missing data'}
            result = self.toolbox.fit(X, y, task_type=task_type)
            return {'model': result, 'status': 'trained'}
        
        # Register tools
        tools = [
            Tool('analyze_data', 'Analyze Data', 'Analyze dataset characteristics', analyze_data, category='data'),
            Tool('preprocess_data', 'Preprocess Data', 'Preprocess data for ML', preprocess_data, category='data'),
            Tool('train_model', 'Train Model', 'Train ML model', train_model, category='ml')
        ]
        
        for tool in tools:
            self.tool_registry.register_tool(tool)
            self.core.register_capability(tool.name.lower().replace(' ', '_'), tool.handler)
    
    def execute_goal(self, goal: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a goal
        
        Parameters
        ----------
        goal : str
            Goal description
        context : dict, optional
            Additional context
            
        Returns
        -------
        result : dict
            Execution result
        """
        import time
        start_time = time.time()
        
        # Update state
        self.core.update_state(AgentStatus.PLANNING, current_task=goal)
        
        # Create plan
        capabilities = self.core.capabilities
        plan = self.planner.create_plan(goal, capabilities, context)
        
        # Validate plan
        validation = self.planner.validate_plan(plan, capabilities)
        if not validation['is_valid']:
            execution_time = time.time() - start_time
            self.core.update_state(AgentStatus.ERROR, error=str(validation['issues']))
            return {
                'success': False,
                'error': 'Plan validation failed',
                'issues': validation['issues'],
                'execution_time': execution_time
            }
        
        # Optimize plan
        plan = self.planner.optimize_plan(plan)
        
        # Update state
        self.core.update_state(AgentStatus.EXECUTING, current_plan=plan)
        
        # Execute plan
        try:
            execution_result = self.executor.execute_plan(plan, self.core)
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[CompleteAgent] Plan execution failed: {e}")
            self.core.update_state(AgentStatus.ERROR, error=str(e))
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }
        
        execution_time = time.time() - start_time
        
        # Record in memory
        self.core.remember({
            'goal': goal,
            'plan': plan.plan_id,
            'success': execution_result.get('steps_failed', 0) == 0,
            'execution_time': execution_time
        })
        
        # Evaluate performance
        success = execution_result.get('steps_failed', 0) == 0
        capabilities_used = [step.action for step in plan.steps] if plan.steps else []
        
        self.evaluator.record_task(
            agent_id=self.core.agent_id,
            task_id=plan.plan_id,
            success=success,
            execution_time=execution_time,
            capabilities_used=capabilities_used
        )
        
        # Update state
        if success:
            self.core.update_state(AgentStatus.COMPLETE)
        else:
            self.core.update_state(AgentStatus.ERROR, error='Plan execution failed')
        
        # Get metrics (may be None if no tasks recorded yet)
        metrics = self.evaluator.get_metrics(self.core.agent_id)
        
        return {
            'success': success,
            'plan_id': plan.plan_id,
            'execution_result': execution_result,
            'execution_time': execution_time,
            'metrics': metrics
        }
    
    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            'agent_id': self.core.agent_id,
            'name': self.core.name,
            'state': {
                'status': self.core.state.status.value,
                'current_task': self.core.state.current_task
            },
            'capabilities': self.core.capabilities,
            'memory_stats': self.core.get_memory_stats(),
            'metrics': self.evaluator.get_metrics(self.core.agent_id)
        }
