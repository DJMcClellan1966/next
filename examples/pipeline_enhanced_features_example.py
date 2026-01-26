"""
Enhanced Pipeline Features Example

Demonstrates:
- Pipeline Monitoring
- Pipeline Persistence
- Pipeline Retry Logic
- Pipeline Debugging

Installation:
    pip install -e .  # From project root

Usage:
    python examples/pipeline_enhanced_features_example.py
"""
import numpy as np
from sklearn.datasets import make_classification

# Import ML Toolbox
try:
    from ml_toolbox import MLToolbox, UnifiedMLPipeline
    from ml_toolbox.pipelines import (
        PipelineMonitor, PipelinePersistence, RetryHandler, RetryConfig,
        RetryStrategy, PipelineDebugger
    )
    print("[OK] ML Toolbox and enhanced features imported successfully")
except ImportError as e:
    print(f"[ERROR] Error importing: {e}")
    print("\n[TIP] Try installing the package first:")
    print("   pip install -e .")
    raise


def example_1_monitoring():
    """Example 1: Pipeline Monitoring"""
    print("\n" + "="*80)
    print("EXAMPLE 1: PIPELINE MONITORING")
    print("="*80)
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=15, random_state=42)
    
    # Initialize toolbox and pipeline with monitoring
    toolbox = MLToolbox()
    pipeline = UnifiedMLPipeline(toolbox, enable_feature_store=True)
    
    # Enable monitoring on pipelines
    pipeline.feature_pipeline.monitor = PipelineMonitor(enable_tracking=True)
    pipeline.training_pipeline.monitor = PipelineMonitor(enable_tracking=True)
    
    print("\n1. Executing pipeline with monitoring...")
    result = pipeline.execute(X, y, mode='train', feature_name='monitored_features',
                              model_name='monitored_model')
    
    print(f"\n[OK] Pipeline completed!")
    print(f"   Model metrics: {result.get('metrics', {})}")
    
    # Get monitoring statistics
    print("\n2. Monitoring Statistics:")
    feature_stats = pipeline.feature_pipeline.monitor.get_pipeline_statistics('feature_pipeline')
    training_stats = pipeline.training_pipeline.monitor.get_pipeline_statistics('training_pipeline')
    
    print(f"   Feature Pipeline:")
    print(f"      Executions: {feature_stats.get('execution_count', 0)}")
    print(f"      Avg Duration: {feature_stats.get('average_duration', 0):.4f}s")
    
    print(f"   Training Pipeline:")
    print(f"      Executions: {training_stats.get('execution_count', 0)}")
    print(f"      Avg Duration: {training_stats.get('average_duration', 0):.4f}s")
    
    return result


def example_2_persistence():
    """Example 2: Pipeline Persistence"""
    print("\n" + "="*80)
    print("EXAMPLE 2: PIPELINE PERSISTENCE")
    print("="*80)
    
    # Generate sample data
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    
    # Initialize
    toolbox = MLToolbox()
    pipeline = UnifiedMLPipeline(toolbox)
    persistence = PipelinePersistence(storage_dir="example_pipeline_storage")
    
    print("\n1. Executing and saving pipeline...")
    result = pipeline.execute(X, y, mode='train', feature_name='persisted_features',
                              model_name='persisted_model')
    
    # Save pipeline state
    state_id = persistence.save_pipeline_state(
        'unified_pipeline',
        pipeline.state.get_state_summary(),
        metadata={'metrics': result.get('metrics', {})}
    )
    print(f"   [OK] Pipeline state saved: {state_id}")
    
    # Save model
    model_id = persistence.save_model(
        result['model'],
        'persisted_model',
        metadata={'metrics': result.get('metrics', {})}
    )
    print(f"   [OK] Model saved: {model_id}")
    
    # Load pipeline state
    print("\n2. Loading pipeline state...")
    loaded_state = persistence.load_pipeline_state('unified_pipeline')
    if loaded_state:
        print(f"   [OK] Pipeline state loaded")
        print(f"      Features: {loaded_state.get('features_count', 0)}")
        print(f"      Models: {loaded_state.get('models_count', 0)}")
    
    # Load model
    print("\n3. Loading model...")
    loaded_model = persistence.load_model('persisted_model')
    if loaded_model:
        print(f"   [OK] Model loaded: {type(loaded_model).__name__}")
    
    return result


def example_3_retry_logic():
    """Example 3: Retry Logic"""
    print("\n" + "="*80)
    print("EXAMPLE 3: RETRY LOGIC")
    print("="*80)
    
    # Generate sample data
    X, y = make_classification(n_samples=400, n_features=12, random_state=42)
    
    # Initialize with retry
    toolbox = MLToolbox()
    pipeline = UnifiedMLPipeline(toolbox)
    
    # Configure retry
    retry_config = RetryConfig(
        max_retries=3,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay=0.5
    )
    retry_handler = RetryHandler(retry_config)
    
    # Enable retry on pipelines
    pipeline.feature_pipeline.retry_handler = retry_handler
    pipeline.training_pipeline.retry_handler = retry_handler
    pipeline.feature_pipeline.enable_retry = True
    pipeline.training_pipeline.enable_retry = True
    
    print("\n1. Executing pipeline with retry logic...")
    result = pipeline.execute(X, y, mode='train', feature_name='retry_features',
                              model_name='retry_model')
    
    print(f"\n[OK] Pipeline completed with retry support!")
    
    # Get retry statistics
    print("\n2. Retry Statistics:")
    feature_retry_stats = retry_handler.get_retry_statistics('feature_pipeline')
    if feature_retry_stats:
        print(f"   Feature Pipeline:")
        print(f"      Total Attempts: {feature_retry_stats.get('total_attempts', 0)}")
        print(f"      Successful: {feature_retry_stats.get('successful_attempts', 0)}")
    
    return result


def example_4_debugging():
    """Example 4: Pipeline Debugging"""
    print("\n" + "="*80)
    print("EXAMPLE 4: PIPELINE DEBUGGING")
    print("="*80)
    
    # Generate sample data
    X, y = make_classification(n_samples=200, n_features=8, random_state=42)
    
    # Initialize with debugging
    toolbox = MLToolbox()
    pipeline = UnifiedMLPipeline(toolbox)
    
    # Enable debugging
    debugger = PipelineDebugger(enable_debugging=True)
    pipeline.feature_pipeline.debugger = debugger
    pipeline.training_pipeline.debugger = debugger
    pipeline.feature_pipeline.enable_debugging = True
    pipeline.training_pipeline.enable_debugging = True
    
    # Add breakpoint (optional)
    # debugger.add_breakpoint('preprocessing')
    
    print("\n1. Executing pipeline with debugging...")
    result = pipeline.execute(X, y, mode='train', feature_name='debug_features',
                              model_name='debug_model')
    
    print(f"\n[OK] Pipeline completed with debugging!")
    
    # Get execution trace
    print("\n2. Execution Trace:")
    trace_summary = debugger.get_trace_summary()
    print(f"   Total Stages: {trace_summary.get('total_stages', 0)}")
    print(f"   Total Duration: {trace_summary.get('total_duration', 0):.4f}s")
    print(f"   Failed Stages: {trace_summary.get('failed_stages', 0)}")
    
    # Visualize trace
    print("\n3. Trace Visualization:")
    visualization = debugger.visualize_trace()
    print(visualization[:500] + "..." if len(visualization) > 500 else visualization)
    
    return result


def main():
    """Run all examples"""
    print("="*80)
    print("ENHANCED PIPELINE FEATURES EXAMPLES")
    print("="*80)
    print("\nDemonstrates monitoring, persistence, retry, and debugging features.")
    
    examples = {
        'Monitoring': example_1_monitoring,
        'Persistence': example_2_persistence,
        'Retry Logic': example_3_retry_logic,
        'Debugging': example_4_debugging
    }
    
    results = {}
    for name, func in examples.items():
        try:
            results[name] = {'success': True, 'result': func()}
        except Exception as e:
            print(f"\n[ERROR] Error in {name}: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    successful = sum(1 for r in results.values() if r.get('success'))
    print(f"\n[OK] Successful: {successful}/{len(examples)}")
    print(f"[ERROR] Failed: {len(examples) - successful}/{len(examples)}")
    
    return results


if __name__ == "__main__":
    main()
