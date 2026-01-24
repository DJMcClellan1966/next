"""
Test Agentic AI Systems
"""
import sys
from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("AGENTIC AI SYSTEMS TEST")
print("="*80)

try:
    from ml_toolbox import MLToolbox
    from ml_toolbox.agentic_systems import (
        CompleteAgent, AgentCore, AgentPlanner, AgentExecutor,
        MultiAgentSystem, AgentRole, AgentEvaluator
    )
    print("\n[OK] All imports successful")
except ImportError as e:
    print(f"\n[ERROR] Import failed: {e}")
    sys.exit(1)

# Generate test data
np.random.seed(42)
X = np.random.randn(100, 10).astype(np.float64)
y = np.random.randint(0, 2, 100)

print(f"\nTest data: {X.shape}, {len(y)} samples")
print("="*80)

# Initialize toolbox
toolbox = MLToolbox()

# Test 1: Complete Agent
print("\n1. COMPLETE AGENT")
print("-"*80)
try:
    agent = CompleteAgent("test_agent", "Test Agent", "Test agent for ML tasks", toolbox=toolbox)
    print(f"Agent created: {agent.core.name}")
    print(f"Capabilities: {agent.core.capabilities}")
    
    # Execute goal
    result = agent.execute_goal(
        "Build a classification model",
        context={"X": X, "y": y}
    )
    print(f"Goal execution: Success={result['success']}")
    print(f"Execution time: {result['execution_time']:.2f}s")
    
    # Get status
    status = agent.get_status()
    print(f"Agent status: {status['state']['status']}")
    print(f"Memory stats: {status['memory_stats']}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Agent Planner
print("\n2. AGENT PLANNER")
print("-"*80)
try:
    planner = AgentPlanner()
    plan = planner.create_plan("Build classification model", ["analyze_data", "train_model", "evaluate_model"])
    print(f"Plan created: {plan.plan_id}")
    print(f"Steps: {len(plan.steps)}")
    for step in plan.steps:
        print(f"  - {step.action}")
    
    validation = planner.validate_plan(plan, ["analyze_data", "train_model", "evaluate_model"])
    print(f"Plan valid: {validation['is_valid']}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Multi-Agent System
print("\n3. MULTI-AGENT SYSTEM")
print("-"*80)
try:
    system = MultiAgentSystem()
    
    # Create agents
    data_agent = CompleteAgent("data1", "Data Agent", toolbox=toolbox)
    ml_agent = CompleteAgent("ml1", "ML Agent", toolbox=toolbox)
    
    # Register
    system.register_agent("data1", data_agent, "Data Agent", 
                         AgentRole.SPECIALIST, ["analyze_data", "preprocess_data"])
    system.register_agent("ml1", ml_agent, "ML Agent",
                         AgentRole.SPECIALIST, ["train_model", "evaluate_model"])
    
    print(f"Registered agents: {len(system.agents)}")
    
    # Assign task
    task_id = system.assign_task({
        "type": "ml_pipeline",
        "data": X,
        "target": y,
        "required_capabilities": ["analyze_data", "train_model"]
    })
    print(f"Task assigned: {task_id}")
    
    # System status
    status = system.get_system_status()
    print(f"System status: {status}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Agent Evaluator
print("\n4. AGENT EVALUATOR")
print("-"*80)
try:
    evaluator = AgentEvaluator()
    evaluator.record_task("agent1", "task1", success=True, execution_time=1.5, quality_score=0.9)
    evaluator.record_task("agent1", "task2", success=True, execution_time=2.0, quality_score=0.85)
    evaluator.record_task("agent1", "task3", success=False, execution_time=0.5, error="Timeout")
    
    metrics = evaluator.get_metrics("agent1")
    if metrics:
        print(f"Total tasks: {metrics.total_tasks}")
        print(f"Success rate: {evaluator.get_success_rate('agent1'):.2%}")
        print(f"Avg execution time: {metrics.avg_execution_time:.2f}s")
    
    system_metrics = evaluator.get_system_metrics()
    print(f"System metrics: {system_metrics}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*80)
print("AGENTIC AI SYSTEMS TEST COMPLETE")
print("="*80)
