"""
Test Multi-Agent Design Patterns
"""
import sys
from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("MULTI-AGENT DESIGN PATTERNS TEST")
print("="*80)

try:
    from ml_toolbox import MLToolbox
    from ml_toolbox.multi_agent_design import (
        AdvancedMultiAgentSystem, AgentHierarchy, HierarchyLevel,
        CoordinatorPattern, BlackboardPattern, ContractNetPattern,
        SwarmPattern, PipelinePattern, TaskDecomposer, AgentNegotiation,
        DistributedExecutor, AgentMonitor
    )
    from ml_toolbox.agentic_systems import CompleteAgent
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

# Test 1: Agent Hierarchy
print("\n1. AGENT HIERARCHY")
print("-"*80)
try:
    hierarchy = AgentHierarchy()
    hierarchy.add_agent("root", HierarchyLevel.ROOT, capabilities=["manage"])
    hierarchy.add_agent("manager1", HierarchyLevel.MANAGER, parent_id="root", capabilities=["coordinate"])
    hierarchy.add_agent("worker1", HierarchyLevel.WORKER, parent_id="manager1", capabilities=["execute"])
    hierarchy.add_agent("worker2", HierarchyLevel.WORKER, parent_id="manager1", capabilities=["execute"])
    
    stats = hierarchy.get_hierarchy_stats()
    print(f"Hierarchy stats: {stats}")
    print(f"Children of manager1: {hierarchy.get_children('manager1')}")
    print(f"Descendants of root: {hierarchy.get_descendants('root')}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Coordination Patterns
print("\n2. COORDINATION PATTERNS")
print("-"*80)
try:
    # Coordinator Pattern
    coordinator = CoordinatorPattern("coord1")
    agent1 = CompleteAgent("agent1", "Worker 1", toolbox=toolbox)
    agent2 = CompleteAgent("agent2", "Worker 2", toolbox=toolbox)
    coordinator.register_worker("agent1", agent1, ["analyze_data"])
    coordinator.register_worker("agent2", agent2, ["train_model"])
    
    from ml_toolbox.multi_agent_design.coordination_patterns import Task
    task = Task("task1", "Analyze data", ["analyze_data"])
    coordinator.submit_task(task)
    assigned = coordinator.assign_task(task)
    print(f"Coordinator assigned task to: {assigned}")
    
    # Blackboard Pattern
    blackboard = BlackboardPattern()
    blackboard.register_agent("agent1", agent1, ["data_analysis"])
    blackboard.write("data_analysis", {"shape": X.shape}, "agent1")
    data = blackboard.read("data_analysis")
    print(f"Blackboard read: {data}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Task Decomposition
print("\n3. TASK DECOMPOSITION")
print("-"*80)
try:
    decomposer = TaskDecomposer()
    graph = decomposer.decompose("Build ML pipeline", strategy="ml_pipeline")
    print(f"Decomposed into {len(graph.tasks)} tasks")
    for task_id, task in graph.tasks.items():
        print(f"  - {task_id}: {task.description}")
    
    parallel_groups = decomposer.identify_parallel_tasks(graph)
    print(f"Parallel groups: {len(parallel_groups)}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Agent Negotiation
print("\n4. AGENT NEGOTIATION")
print("-"*80)
try:
    negotiation = AgentNegotiation()
    negotiation.register_agent("agent1", agent1)
    negotiation.register_agent("agent2", agent2)
    
    neg_id = negotiation.initiate_negotiation("neg1", {"task": "analyze"}, "manager", ["agent1", "agent2"])
    proposal = negotiation.submit_proposal(neg_id, "agent1", {"cost": 10, "quality": 0.9, "estimated_time": 5.0})
    print(f"Proposal submitted: {proposal.proposal_id}")
    
    best = negotiation.evaluate_proposals(neg_id, criteria='best_value')
    if best:
        agreement = negotiation.form_agreement(neg_id, best)
        print(f"Agreement formed: {agreement.agreement_id}")
except Exception as e:
    print(f"Error: {e}")

# Test 5: Distributed Execution
print("\n5. DISTRIBUTED EXECUTION")
print("-"*80)
try:
    executor = DistributedExecutor(strategy=ExecutionStrategy.PARALLEL)
    executor.register_agent("agent1", agent1)
    executor.register_agent("agent2", agent2)
    
    from ml_toolbox.multi_agent_design.distributed_execution import ExecutionTask
    tasks = [
        ExecutionTask("task1", "agent1", "analyze_data", {"data": X}),
        ExecutionTask("task2", "agent2", "train_model", {"X": X, "y": y})
    ]
    
    results = executor.execute_tasks(tasks)
    print(f"Executed {len(tasks)} tasks")
    print(f"Results: {list(results.keys())}")
except Exception as e:
    print(f"Error: {e}")

# Test 6: Agent Monitoring
print("\n6. AGENT MONITORING")
print("-"*80)
try:
    monitor = AgentMonitor()
    monitor.register_agent("agent1", agent1)
    monitor.register_agent("agent2", agent2)
    
    health = monitor.check_all_agents()
    print(f"Health checks: {len(health)} agents")
    for agent_id, check in health.items():
        print(f"  - {agent_id}: {check.status.value}")
    
    system_health = monitor.get_system_health()
    print(f"System health: {system_health['status']}")
except Exception as e:
    print(f"Error: {e}")

# Test 7: Advanced Multi-Agent System
print("\n7. ADVANCED MULTI-AGENT SYSTEM")
print("-"*80)
try:
    system = AdvancedMultiAgentSystem(coordination_pattern='coordinator')
    system.add_agent("agent1", agent1, role="worker", level=HierarchyLevel.WORKER, 
                    capabilities=["analyze_data", "preprocess_data"])
    system.add_agent("agent2", agent2, role="worker", level=HierarchyLevel.WORKER,
                    capabilities=["train_model", "evaluate_model"])
    
    stats = system.get_system_stats()
    print(f"System stats: {stats}")
    
    # Monitor system
    monitoring = system.monitor_system()
    print(f"System monitoring: {monitoring['system_health']}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*80)
print("MULTI-AGENT DESIGN PATTERNS TEST COMPLETE")
print("="*80)
