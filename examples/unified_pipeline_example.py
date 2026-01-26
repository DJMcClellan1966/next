"""
Unified ML Pipeline Example

Demonstrates the new unified pipeline architecture (Option 3):
- Feature Pipeline → Training Pipeline → Inference Pipeline

Installation:
    pip install -e .  # From project root

Usage:
    python examples/unified_pipeline_example.py
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Import ML Toolbox
try:
    from ml_toolbox import MLToolbox, UnifiedMLPipeline
    print("[OK] ML Toolbox and UnifiedMLPipeline imported successfully")
except ImportError as e:
    print(f"[ERROR] Error importing: {e}")
    print("\n[TIP] Try installing the package first:")
    print("   pip install -e .")
    raise


def example_1_training_pipeline():
    """Example 1: Complete training pipeline"""
    print("\n" + "="*80)
    print("EXAMPLE 1: TRAINING PIPELINE")
    print("="*80)
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                               n_redundant=10, random_state=42)
    
    # Initialize toolbox and pipeline
    toolbox = MLToolbox()
    pipeline = UnifiedMLPipeline(toolbox, enable_feature_store=True, enable_registry=True)
    
    print("\n1. Executing training pipeline...")
    result = pipeline.execute(
        X, y,
        mode='train',
        feature_name='classification_features',
        model_name='classifier'
    )
    
    print(f"\n[OK] Training completed!")
    print(f"   Model ID: {result.get('model_id', 'N/A')}")
    print(f"   Metrics: {result.get('metrics', {})}")
    print(f"   Features shape: {result['features'].shape}")
    
    return result


def example_2_inference_pipeline():
    """Example 2: Inference pipeline with feature reuse"""
    print("\n" + "="*80)
    print("EXAMPLE 2: INFERENCE PIPELINE (WITH FEATURE REUSE)")
    print("="*80)
    
    # Generate sample data
    X_train, y_train = make_classification(n_samples=1000, n_features=20, 
                                          n_informative=10, n_redundant=10, random_state=42)
    X_test, _ = make_classification(n_samples=200, n_features=20, 
                                   n_informative=10, n_redundant=10, random_state=123)
    
    # Initialize toolbox and pipeline
    toolbox = MLToolbox()
    pipeline = UnifiedMLPipeline(toolbox, enable_feature_store=True)
    
    # Step 1: Train
    print("\n1. Training model...")
    train_result = pipeline.execute(
        X_train, y_train,
        mode='train',
        feature_name='test_features',
        model_name='test_model'
    )
    
    # Step 2: Inference (reuses features from training)
    print("\n2. Running inference (reusing features)...")
    inference_result = pipeline.execute(
        X_test,
        mode='inference',
        feature_name='test_features',
        model_name='test_model',
        reuse_features=True
    )
    
    print(f"\n[OK] Inference completed!")
    print(f"   Predictions shape: {inference_result['predictions'].shape}")
    print(f"   Sample predictions: {inference_result['predictions'][:5]}")
    print(f"   Monitoring metrics: {inference_result.get('monitoring_metrics', {})}")
    
    return inference_result


def example_3_pipeline_status():
    """Example 3: Check pipeline status"""
    print("\n" + "="*80)
    print("EXAMPLE 3: PIPELINE STATUS")
    print("="*80)
    
    # Generate sample data
    X, y = make_regression(n_samples=500, n_features=15, noise=0.1, random_state=42)
    
    # Initialize toolbox and pipeline
    toolbox = MLToolbox()
    pipeline = UnifiedMLPipeline(toolbox)
    
    # Execute training
    print("\n1. Executing training...")
    result = pipeline.execute(X, y, mode='train', feature_name='regression_features', 
                             model_name='regressor')
    
    # Get status
    print("\n2. Pipeline status:")
    status = pipeline.get_status()
    
    print(f"\n   Feature Pipeline Stages: {len(status['feature_pipeline']['stages'])}")
    print(f"   Training Pipeline Stages: {len(status['training_pipeline']['stages'])}")
    print(f"   Inference Pipeline Stages: {len(status['inference_pipeline']['stages'])}")
    print(f"\n   State Summary:")
    state_summary = status['state']
    print(f"      Version: {state_summary['version']}")
    print(f"      Features: {state_summary['features_count']}")
    print(f"      Models: {state_summary['models_count']}")
    
    return status


def example_4_individual_pipelines():
    """Example 4: Use individual pipelines"""
    print("\n" + "="*80)
    print("EXAMPLE 4: INDIVIDUAL PIPELINES")
    print("="*80)
    
    from ml_toolbox.pipelines import FeaturePipeline, TrainingPipeline, InferencePipeline
    
    # Generate sample data
    X, y = make_classification(n_samples=800, n_features=15, random_state=42)
    
    # Initialize toolbox
    toolbox = MLToolbox()
    
    # Step 1: Feature Pipeline
    print("\n1. Feature Pipeline...")
    feature_pipeline = FeaturePipeline(toolbox)
    X_features = feature_pipeline.execute(X, feature_name='individual_features')
    print(f"   [OK] Features shape: {X_features.shape}")
    
    # Step 2: Training Pipeline
    print("\n2. Training Pipeline...")
    training_pipeline = TrainingPipeline(toolbox)
    train_result = training_pipeline.execute(X_features, y, model_name='individual_model')
    print(f"   [OK] Model trained: {type(train_result['model']).__name__}")
    print(f"   Metrics: {train_result.get('metrics', {})}")
    
    # Step 3: Inference Pipeline
    print("\n3. Inference Pipeline...")
    X_test = X[:100]
    inference_pipeline = InferencePipeline(toolbox)
    inference_result = inference_pipeline.execute(X_test, train_result['model'])
    print(f"   [OK] Predictions: {inference_result['predictions'].shape}")
    
    return {
        'features': X_features,
        'model': train_result['model'],
        'predictions': inference_result['predictions']
    }


def main():
    """Run all examples"""
    print("="*80)
    print("UNIFIED ML PIPELINE EXAMPLES")
    print("="*80)
    print("\nThis demonstrates the new unified pipeline architecture (Option 3)")
    print("with explicit Feature -> Training -> Inference pipelines.")
    
    examples = {
        'Training Pipeline': example_1_training_pipeline,
        'Inference Pipeline': example_2_inference_pipeline,
        'Pipeline Status': example_3_pipeline_status,
        'Individual Pipelines': example_4_individual_pipelines
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
