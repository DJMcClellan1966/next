"""
Agent Executor - Execute plans and actions

Implements:
- Action execution
- Result handling
- Error recovery
- Retry logic
"""
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Action status"""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Action:
    """Agent action"""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    handler: Optional[Callable] = None
    max_retries: int = 3
    timeout: Optional[float] = None


@dataclass
class ActionResult:
    """Action execution result"""
    action_id: str
    status: ActionStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0


class AgentExecutor:
    """
    Agent Executor - Execute agent actions and plans
    
    Implements:
    - Action execution
    - Error handling
    - Retry logic
    - Timeout management
    """
    
    def __init__(self):
        self.action_handlers = {}
        self.execution_history = []
        self.max_retries = 3
        self.default_timeout = 30.0
    
    def register_handler(self, action_type: str, handler: Callable):
        """
        Register action handler
        
        Parameters
        ----------
        action_type : str
            Action type
        handler : callable
            Handler function
        """
        self.action_handlers[action_type] = handler
        logger.info(f"[AgentExecutor] Registered handler: {action_type}")
    
    def execute_action(self, action: Action) -> ActionResult:
        """
        Execute an action
        
        Parameters
        ----------
        action : Action
            Action to execute
            
        Returns
        -------
        result : ActionResult
            Execution result
        """
        import time
        
        start_time = time.time()
        result = ActionResult(action_id=action.action_id, status=ActionStatus.EXECUTING)
        
        # Get handler
        handler = action.handler
        if not handler and action.action_type in self.action_handlers:
            handler = self.action_handlers[action.action_type]
        
        if not handler:
            result.status = ActionStatus.FAILED
            result.error = f"No handler for action type: {action.action_type}"
            result.execution_time = time.time() - start_time
            return result
        
        # Execute with retries
        for attempt in range(action.max_retries):
            try:
                result.retry_count = attempt
                
                if attempt > 0:
                    result.status = ActionStatus.RETRYING
                    logger.info(f"[AgentExecutor] Retrying action {action.action_id} (attempt {attempt + 1})")
                
                # Execute handler
                action_result = handler(**action.parameters)
                
                result.status = ActionStatus.SUCCESS
                result.result = action_result
                result.execution_time = time.time() - start_time
                
                logger.info(f"[AgentExecutor] Action {action.action_id} succeeded")
                break
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"[AgentExecutor] Action {action.action_id} failed: {error_msg}")
                
                if attempt == action.max_retries - 1:
                    result.status = ActionStatus.FAILED
                    result.error = error_msg
                    result.execution_time = time.time() - start_time
                else:
                    # Wait before retry
                    time.sleep(0.5 * (attempt + 1))
        
        # Store in history
        self.execution_history.append({
            'action_id': action.action_id,
            'action_type': action.action_type,
            'status': result.status.value,
            'execution_time': result.execution_time,
            'retry_count': result.retry_count
        })
        
        return result
    
    def execute_plan(self, plan, agent_core) -> Dict[str, Any]:
        """
        Execute a plan
        
        Parameters
        ----------
        plan : Plan
            Plan to execute
        agent_core : AgentCore
            Agent core instance
            
        Returns
        -------
        results : dict
            Execution results
        """
        from .agent_planner import PlanStatus
        
        plan.status = PlanStatus.EXECUTING
        results = {}
        
        while not plan.is_complete():
            # Get next step
            next_step = plan.get_next_step()
            
            if not next_step:
                # No more steps or waiting for dependencies
                break
            
            # Create action
            action = Action(
                action_id=next_step.step_id,
                action_type=next_step.action,
                parameters=next_step.parameters,
                handler=agent_core.tools.get(next_step.action),
                max_retries=3
            )
            
            # Execute action
            result = self.execute_action(action)
            
            # Update step
            if result.status == ActionStatus.SUCCESS:
                plan.mark_step_complete(next_step.step_id, result.result)
                results[next_step.step_id] = result.result
            else:
                plan.mark_step_failed(next_step.step_id, result.error or "Unknown error")
                results[next_step.step_id] = {'error': result.error}
                
                # Decide whether to continue or stop
                if plan.has_failed():
                    plan.status = PlanStatus.FAILED
                    break
        
        if plan.is_complete() and not plan.has_failed():
            plan.status = PlanStatus.COMPLETE
        
        return {
            'plan_id': plan.plan_id,
            'status': plan.status.value,
            'results': results,
            'steps_completed': sum(1 for s in plan.steps if s.status == PlanStatus.COMPLETE),
            'steps_failed': sum(1 for s in plan.steps if s.status == PlanStatus.FAILED)
        }
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        if not self.execution_history:
            return {'total_actions': 0}
        
        total = len(self.execution_history)
        successful = sum(1 for h in self.execution_history if h['status'] == 'success')
        failed = sum(1 for h in self.execution_history if h['status'] == 'failed')
        avg_time = sum(h['execution_time'] for h in self.execution_history) / total
        
        return {
            'total_actions': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0.0,
            'avg_execution_time': avg_time
        }
