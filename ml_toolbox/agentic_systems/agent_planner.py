"""
Agent Planner - Planning and Goal Decomposition

Implements:
- Goal decomposition
- Plan generation
- Plan validation
- Plan optimization
"""
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    """Plan status"""
    CREATED = "created"
    VALIDATED = "validated"
    EXECUTING = "executing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PlanStep:
    """Plan step"""
    step_id: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    expected_result: Optional[str] = None
    status: PlanStatus = PlanStatus.CREATED
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class Plan:
    """Agent plan"""
    plan_id: str
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.CREATED
    created_at: str = ""
    completed_at: Optional[str] = None
    
    def add_step(self, step: PlanStep):
        """Add step to plan"""
        self.steps.append(step)
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get next step to execute"""
        for step in self.steps:
            if step.status == PlanStatus.CREATED:
                # Check dependencies
                if all(self._is_step_complete(dep) for dep in step.dependencies):
                    return step
        return None
    
    def _is_step_complete(self, step_id: str) -> bool:
        """Check if step is complete"""
        for step in self.steps:
            if step.step_id == step_id:
                return step.status == PlanStatus.COMPLETE
        return True  # Assume complete if not found
    
    def mark_step_complete(self, step_id: str, result: Any = None):
        """Mark step as complete"""
        for step in self.steps:
            if step.step_id == step_id:
                step.status = PlanStatus.COMPLETE
                step.result = result
                break
    
    def mark_step_failed(self, step_id: str, error: str):
        """Mark step as failed"""
        for step in self.steps:
            if step.step_id == step_id:
                step.status = PlanStatus.FAILED
                step.error = error
                break
    
    def is_complete(self) -> bool:
        """Check if plan is complete"""
        return all(step.status in [PlanStatus.COMPLETE, PlanStatus.FAILED] 
                  for step in self.steps)
    
    def has_failed(self) -> bool:
        """Check if plan has failed steps"""
        return any(step.status == PlanStatus.FAILED for step in self.steps)


class AgentPlanner:
    """
    Agent Planner - Generate and manage plans
    
    Implements:
    - Goal decomposition
    - Plan generation
    - Plan validation
    - Plan optimization
    """
    
    def __init__(self):
        self.plan_templates = {}
        self.decomposition_rules = {}
        self._init_templates()
    
    def _init_templates(self):
        """Initialize plan templates"""
        
        # ML task templates
        self.plan_templates['classification'] = [
            {'action': 'analyze_data', 'parameters': {}},
            {'action': 'preprocess_data', 'parameters': {}},
            {'action': 'select_model', 'parameters': {'task_type': 'classification'}},
            {'action': 'train_model', 'parameters': {}},
            {'action': 'evaluate_model', 'parameters': {}}
        ]
        
        self.plan_templates['regression'] = [
            {'action': 'analyze_data', 'parameters': {}},
            {'action': 'preprocess_data', 'parameters': {}},
            {'action': 'select_model', 'parameters': {'task_type': 'regression'}},
            {'action': 'train_model', 'parameters': {}},
            {'action': 'evaluate_model', 'parameters': {}}
        ]
    
    def create_plan(self, goal: str, agent_capabilities: List[str], 
                   context: Optional[Dict] = None) -> Plan:
        """
        Create plan for goal
        
        Parameters
        ----------
        goal : str
            Goal description
        agent_capabilities : list of str
            Available agent capabilities
        context : dict, optional
            Additional context
            
        Returns
        -------
        plan : Plan
            Generated plan
        """
        plan_id = f"plan_{len(self.plan_templates)}"
        plan = Plan(plan_id=plan_id, goal=goal)
        
        # Detect task type
        task_type = self._detect_task_type(goal)
        
        # Use template if available
        if task_type in self.plan_templates:
            template = self.plan_templates[task_type]
            for i, step_template in enumerate(template):
                step = PlanStep(
                    step_id=f"{plan_id}_step_{i}",
                    action=step_template['action'],
                    parameters=step_template.get('parameters', {}),
                    dependencies=[f"{plan_id}_step_{j}" for j in range(i)]  # Sequential
                )
                plan.add_step(step)
        else:
            # Generic plan
            steps = self._decompose_goal(goal, agent_capabilities)
            for i, step_info in enumerate(steps):
                step = PlanStep(
                    step_id=f"{plan_id}_step_{i}",
                    action=step_info['action'],
                    parameters=step_info.get('parameters', {}),
                    dependencies=step_info.get('dependencies', [])
                )
                plan.add_step(step)
        
        logger.info(f"[AgentPlanner] Created plan: {plan_id} with {len(plan.steps)} steps")
        return plan
    
    def _detect_task_type(self, goal: str) -> str:
        """Detect task type from goal"""
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ['classify', 'classification', 'predict category']):
            return 'classification'
        elif any(word in goal_lower for word in ['predict', 'regression', 'forecast']):
            return 'regression'
        elif any(word in goal_lower for word in ['analyze', 'explore', 'understand']):
            return 'analysis'
        elif any(word in goal_lower for word in ['deploy', 'serve', 'production']):
            return 'deployment'
        else:
            return 'general'
    
    def _decompose_goal(self, goal: str, capabilities: List[str]) -> List[Dict]:
        """Decompose goal into steps"""
        steps = []
        
        # Generic decomposition
        if 'analyze' in goal.lower() or 'explore' in goal.lower():
            steps.append({'action': 'analyze_data'})
        
        if 'preprocess' in goal.lower() or 'clean' in goal.lower():
            steps.append({'action': 'preprocess_data'})
        
        if 'train' in goal.lower() or 'build' in goal.lower():
            steps.append({'action': 'select_model'})
            steps.append({'action': 'train_model'})
        
        if 'evaluate' in goal.lower() or 'test' in goal.lower():
            steps.append({'action': 'evaluate_model'})
        
        # Default steps if none found
        if not steps:
            steps = [
                {'action': 'understand_task'},
                {'action': 'execute_task'},
                {'action': 'verify_result'}
            ]
        
        return steps
    
    def validate_plan(self, plan: Plan, agent_capabilities: List[str]) -> Dict[str, Any]:
        """
        Validate plan
        
        Parameters
        ----------
        plan : Plan
            Plan to validate
        agent_capabilities : list of str
            Available capabilities
            
        Returns
        -------
        validation : dict
            Validation results
        """
        issues = []
        warnings = []
        
        # Check if all actions are supported
        for step in plan.steps:
            if step.action not in agent_capabilities:
                issues.append(f"Step {step.step_id}: Action '{step.action}' not supported")
        
        # Check dependencies
        step_ids = {step.step_id for step in plan.steps}
        for step in plan.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    warnings.append(f"Step {step.step_id}: Dependency '{dep}' not found")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            plan.status = PlanStatus.VALIDATED
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'warnings': warnings
        }
    
    def optimize_plan(self, plan: Plan) -> Plan:
        """
        Optimize plan (reorder, parallelize, etc.)
        
        Parameters
        ----------
        plan : Plan
            Plan to optimize
            
        Returns
        -------
        optimized_plan : Plan
            Optimized plan
        """
        # Simple optimization: identify parallelizable steps
        # In production, use more sophisticated algorithms
        
        optimized = Plan(plan_id=plan.plan_id, goal=plan.goal)
        
        # Group steps by dependencies
        independent_steps = [s for s in plan.steps if not s.dependencies]
        dependent_steps = [s for s in plan.steps if s.dependencies]
        
        # Add independent steps first (can be parallelized)
        for step in independent_steps:
            optimized.add_step(step)
        
        # Add dependent steps
        for step in dependent_steps:
            optimized.add_step(step)
        
        logger.info(f"[AgentPlanner] Optimized plan: {plan.plan_id}")
        return optimized
