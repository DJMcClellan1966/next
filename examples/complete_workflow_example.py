"""
Complete ML Workflow Example

Demonstrates end-to-end ML pipeline using ML Toolbox
This is what users actually want to do.

Installation:
    pip install -e .  # From project root

Usage:
    python examples/complete_workflow_example.py
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Import ML Toolbox
try:
    from ml_toolbox import MLToolbox
    print("✅ ML Toolbox imported successfully")
except ImportError as e:
    print(f"❌ Error importing ML Toolbox: {e}")
    print("\n💡 Try installing the package first:")
    print("   pip install -e .")
    raise

def example_1_simple_classification():
    """Example 1: Simple Classification - What users actually want"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Classification")
    print("="*80)
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                               n_redundant=10, n_classes=2, random_state=42)
    
    # Use toolbox
    toolbox = MLToolbox()
    
    # Simple API - auto-detects and trains
    result = toolbox.fit(X, y)
    
    print(f"\n✅ Model trained!")
    print(f"   Model type: {result.get('model_type', 'auto-detected')}")
    print(f"   Accuracy: {result.get('accuracy', 'N/A')}")
    
    # Make predictions
    predictions = toolbox.predict(result['model'], X[:10])
    print(f"\n✅ Predictions made: {predictions[:5]}")
    
    return result


def example_2_natural_language_ml():
    """Example 2: Natural Language ML - Super Power Agent"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Natural Language ML")
    print("="*80)
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=15, n_classes=2, random_state=42)
    
    toolbox = MLToolbox()
    
    # Natural language interface
    response = toolbox.chat("Classify this data and tell me the accuracy", X, y)
    
    print(f"\n✅ Agent response:")
    print(f"   {response.get('message', 'No message')}")
    print(f"   Success: {response.get('success', False)}")
    
    return response


def example_3_agent_workflow():
    """Example 3: Agent Workflow - Multi-agent system"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Agent Workflow")
    print("="*80)
    
    toolbox = MLToolbox()
    
    if toolbox.agents and hasattr(toolbox.agents, 'systems'):
        # Use agent systems
        agent = toolbox.agents.systems.create_super_power_agent()
        if agent:
            print("\n✅ Super Power Agent created")
            # Use agent for task
            print("   Agent ready for natural language ML tasks")
        else:
            print("\n⚠️  Agent not available")
    else:
        print("\n⚠️  Agent systems not available")
    
    return None


def example_4_brain_features():
    """Example 4: Brain Features - Cognitive architecture"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Brain Features")
    print("="*80)
    
    toolbox = MLToolbox()
    
    if toolbox.agents and hasattr(toolbox.agents, 'core'):
        # Create brain system
        brain = toolbox.agents.core.create_brain_system()
        if brain:
            # Use brain
            thinking = brain.think("How to improve model accuracy?")
            brain.remember("User prefers fast models", importance=0.8)
            recalled = brain.recall("models")
            state = brain.get_state()
            
            print("\n✅ Brain system working:")
            print(f"   Working memory chunks: {state.get('working_memory', {}).get('chunks', 0)}")
            print(f"   Episodic events: {state.get('episodic_memory', {}).get('events', 0)}")
        else:
            print("\n⚠️  Brain system not available")
    else:
        print("\n⚠️  Agent core not available")
    
    return None


def example_5_production_workflow():
    """Example 5: Production Workflow - End-to-end"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Production Workflow")
    print("="*80)
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    
    toolbox = MLToolbox()
    
    # Train
    result = toolbox.fit(X, y)
    
    # Register model
    if toolbox.model_registry:
        model_id = toolbox.register_model(
            result['model'],
            model_name='house_price_predictor',
            metadata={'r2_score': result.get('r2_score', 0.0)}
        )
        print(f"\n✅ Model registered: {model_id}")
        
        # Retrieve model
        model, metadata = toolbox.get_registered_model(model_id)
        print(f"   Retrieved model: {model_id}")
    else:
        print("\n⚠️  Model registry not available")
    
    return result


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*80)
    print("COMPLETE ML TOOLBOX WORKFLOW EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate what the toolbox can actually do.")
    print("Focus on making these work well rather than adding more features.\n")
    
    examples = [
        ("Simple Classification", example_1_simple_classification),
        ("Natural Language ML", example_2_natural_language_ml),
        ("Agent Workflow", example_3_agent_workflow),
        ("Brain Features", example_4_brain_features),
        ("Production Workflow", example_5_production_workflow),
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            result = example_func()
            results[name] = {'success': True, 'result': result}
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    successful = sum(1 for r in results.values() if r.get('success'))
    print(f"\n✅ Successful: {successful}/{len(examples)}")
    print(f"❌ Failed: {len(examples) - successful}/{len(examples)}")
    
    return results


if __name__ == "__main__":
    run_all_examples()
