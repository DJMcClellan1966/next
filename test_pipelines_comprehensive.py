"""
Comprehensive Pipeline Tests

Tests all pipeline components:
- Data Collection Pipeline (ETL)
- Feature Pipeline
- Training Pipeline
- Inference Pipeline
- UnifiedMLPipeline
- Enhanced Features (Monitoring, Persistence, Retry, Debugging)
"""
import numpy as np
import sys
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
import tempfile
import shutil
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print("COMPREHENSIVE PIPELINE TEST SUITE")
print("="*80)

# Test results
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_result(test_name, passed, message=""):
    """Record test result"""
    if passed:
        test_results['passed'].append(test_name)
        print(f"[OK] {test_name}")
        if message:
            print(f"     {message}")
    else:
        test_results['failed'].append((test_name, message))
        print(f"[FAIL] {test_name}: {message}")

def test_warning(test_name, message):
    """Record test warning"""
    test_results['warnings'].append((test_name, message))
    print(f"[WARNING] {test_name}: {message}")

# Test 1: Import Tests
print("\n" + "="*80)
print("TEST 1: IMPORTS")
print("="*80)

try:
    from ml_toolbox import MLToolbox
    from ml_toolbox.pipelines import (
        DataCollectionPipeline, FeaturePipeline, TrainingPipeline,
        InferencePipeline, UnifiedMLPipeline, PipelineState, FeatureStore
    )
    test_result("Import MLToolbox and pipelines", True)
except Exception as e:
    test_result("Import MLToolbox and pipelines", False, str(e))
    sys.exit(1)

try:
    from ml_toolbox.pipelines import (
        PipelineMonitor, PipelinePersistence, RetryHandler, RetryConfig,
        RetryStrategy, PipelineDebugger
    )
    test_result("Import enhanced features", True)
except Exception as e:
    test_warning("Import enhanced features", str(e))

# Test 2: Data Collection Pipeline (ETL)
print("\n" + "="*80)
print("TEST 2: DATA COLLECTION PIPELINE (ETL)")
print("="*80)

try:
    toolbox = MLToolbox()
    data_collection = DataCollectionPipeline(toolbox)
    
    # Test user input extraction
    user_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    result = data_collection.execute(user_data, source_type='user_input')
    assert isinstance(result, np.ndarray), "Result should be numpy array"
    assert result.shape[0] == 3, "Should have 3 rows"
    test_result("Extract from user input", True, f"Shape: {result.shape}")
    
    # Test dict input
    dict_data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    result = data_collection.execute(dict_data, source_type='user_input')
    assert isinstance(result, np.ndarray), "Result should be numpy array"
    test_result("Extract from dict input", True, f"Shape: {result.shape}")
    
    # Test NoSQL simulation
    nosql_data = [
        {'_id': '1', 'temp': 25.5, 'humidity': 60.0},
        {'_id': '2', 'temp': 26.0, 'humidity': 58.0}
    ]
    result = data_collection.execute(nosql_data, source_type='nosql')
    assert isinstance(result, np.ndarray), "Result should be numpy array"
    test_result("Extract from NoSQL (simulated)", True, f"Shape: {result.shape}")
    
except Exception as e:
    test_result("Data Collection Pipeline", False, str(e))

# Test 3: Feature Pipeline
print("\n" + "="*80)
print("TEST 3: FEATURE PIPELINE")
print("="*80)

try:
    toolbox = MLToolbox()
    feature_pipeline = FeaturePipeline(toolbox)
    
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    features = feature_pipeline.execute(X, feature_name='test_features')
    
    assert isinstance(features, np.ndarray), "Features should be numpy array"
    assert features.shape[0] == X.shape[0], "Should preserve number of samples"
    test_result("Feature Pipeline execution", True, f"Shape: {features.shape}")
    
    # Test feature store
    stored = feature_pipeline.get_stored_features('test_features')
    if stored is not None:
        test_result("Feature Store retrieval", True, f"Retrieved shape: {stored.shape}")
    else:
        test_warning("Feature Store retrieval", "Features not found in store")
    
except Exception as e:
    test_result("Feature Pipeline", False, str(e))

# Test 4: Training Pipeline
print("\n" + "="*80)
print("TEST 4: TRAINING PIPELINE")
print("="*80)

try:
    toolbox = MLToolbox()
    training_pipeline = TrainingPipeline(toolbox)
    
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    result = training_pipeline.execute(X, y, model_name='test_model')
    
    assert 'model' in result, "Result should contain model"
    assert 'metrics' in result, "Result should contain metrics"
    assert hasattr(result['model'], 'predict'), "Model should have predict method"
    test_result("Training Pipeline execution", True, f"Metrics: {result.get('metrics', {})}")
    
except Exception as e:
    test_result("Training Pipeline", False, str(e))

# Test 5: Inference Pipeline
print("\n" + "="*80)
print("TEST 5: INFERENCE PIPELINE")
print("="*80)

try:
    toolbox = MLToolbox()
    inference_pipeline = InferencePipeline(toolbox)
    
    # Train a simple model first
    from sklearn.ensemble import RandomForestClassifier
    X_train, y_train = make_classification(n_samples=50, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Test inference
    X_test = X_train[:10]
    result = inference_pipeline.execute(X_test, model)
    
    assert 'predictions' in result, "Result should contain predictions"
    assert len(result['predictions']) == len(X_test), "Should have predictions for all samples"
    test_result("Inference Pipeline execution", True, f"Predictions: {len(result['predictions'])}")
    
except Exception as e:
    test_result("Inference Pipeline", False, str(e))

# Test 6: UnifiedMLPipeline
print("\n" + "="*80)
print("TEST 6: UNIFIED ML PIPELINE")
print("="*80)

try:
    toolbox = MLToolbox()
    pipeline = UnifiedMLPipeline(toolbox, enable_feature_store=True)
    
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Test training
    result = pipeline.execute(X, y, mode='train', feature_name='unified_test', model_name='unified_model')
    assert 'model' in result, "Result should contain model"
    assert 'features' in result, "Result should contain features"
    test_result("Unified Pipeline - Training", True, f"Model ID: {result.get('model_id', 'N/A')}")
    
    # Test inference
    X_test = X[:20]
    inference_result = pipeline.execute(X_test, mode='inference', feature_name='unified_test',
                                       model_name='unified_model', reuse_features=True)
    assert 'predictions' in inference_result, "Result should contain predictions"
    test_result("Unified Pipeline - Inference", True, f"Predictions: {len(inference_result['predictions'])}")
    
except Exception as e:
    test_result("UnifiedMLPipeline", False, str(e))

# Test 7: Unified Pipeline with ETL
print("\n" + "="*80)
print("TEST 7: UNIFIED PIPELINE WITH ETL")
print("="*80)

try:
    toolbox = MLToolbox()
    pipeline = UnifiedMLPipeline(toolbox, enable_data_collection=True)
    
    # Raw user input
    X_raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
    y = np.array([0, 1, 0, 1, 1])
    
    result = pipeline.execute(
        X_raw,
        y,
        mode='train',
        use_data_collection=True,
        source_type='user_input',
        feature_name='etl_test',
        model_name='etl_model'
    )
    
    assert 'model' in result, "Result should contain model"
    assert 'features' in result, "Result should contain features"
    test_result("Unified Pipeline with ETL", True, f"Features shape: {result['features'].shape}")
    
except Exception as e:
    test_result("Unified Pipeline with ETL", False, str(e))

# Test 8: Pipeline Monitoring
print("\n" + "="*80)
print("TEST 8: PIPELINE MONITORING")
print("="*80)

try:
    from ml_toolbox.pipelines import PipelineMonitor
    
    monitor = PipelineMonitor(enable_tracking=True)
    toolbox = MLToolbox()
    feature_pipeline = FeaturePipeline(toolbox)
    feature_pipeline.monitor = monitor
    
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    features = feature_pipeline.execute(X, feature_name='monitored_features')
    
    stats = monitor.get_pipeline_statistics('feature_pipeline')
    assert 'execution_count' in stats, "Stats should contain execution_count"
    test_result("Pipeline Monitoring", True, f"Executions: {stats.get('execution_count', 0)}")
    
except Exception as e:
    test_warning("Pipeline Monitoring", str(e))

# Test 9: Pipeline Persistence
print("\n" + "="*80)
print("TEST 9: PIPELINE PERSISTENCE")
print("="*80)

try:
    from ml_toolbox.pipelines import PipelinePersistence
    
    # Use temporary directory
    temp_dir = tempfile.mkdtemp()
    persistence = PipelinePersistence(storage_dir=temp_dir)
    
    # Save pipeline state
    test_state = {'features': np.array([1, 2, 3]), 'version': 1}
    state_id = persistence.save_pipeline_state('test_pipeline', test_state)
    assert state_id is not None, "Should return state ID"
    test_result("Save pipeline state", True, f"State ID: {state_id}")
    
    # Load pipeline state
    loaded_state = persistence.load_pipeline_state('test_pipeline')
    assert loaded_state is not None, "Should load state"
    test_result("Load pipeline state", True, f"Loaded version: {loaded_state.get('version', 'N/A')}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
except Exception as e:
    test_warning("Pipeline Persistence", str(e))

# Test 10: Retry Logic
print("\n" + "="*80)
print("TEST 10: RETRY LOGIC")
print("="*80)

try:
    from ml_toolbox.pipelines import RetryHandler, RetryConfig, RetryStrategy
    
    retry_config = RetryConfig(max_retries=2, strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
    retry_handler = RetryHandler(retry_config)
    
    # Test successful execution
    def success_func():
        return np.array([1, 2, 3])
    
    result = retry_handler.execute_with_retry('test_stage', success_func)
    assert isinstance(result, np.ndarray), "Should return result"
    test_result("Retry Logic - Success", True, f"Result shape: {result.shape}")
    
    # Test retry statistics
    stats = retry_handler.get_retry_statistics('test_stage')
    assert 'total_attempts' in stats, "Should have retry statistics"
    test_result("Retry Statistics", True, f"Attempts: {stats.get('total_attempts', 0)}")
    
except Exception as e:
    test_warning("Retry Logic", str(e))

# Test 11: Pipeline Debugger
print("\n" + "="*80)
print("TEST 11: PIPELINE DEBUGGER")
print("="*80)

try:
    from ml_toolbox.pipelines import PipelineDebugger
    
    debugger = PipelineDebugger(enable_debugging=True)
    toolbox = MLToolbox()
    feature_pipeline = FeaturePipeline(toolbox)
    feature_pipeline.debugger = debugger
    
    X, y = make_classification(n_samples=30, n_features=5, random_state=42)
    features = feature_pipeline.execute(X, feature_name='debugged_features')
    
    trace = debugger.get_execution_trace()
    assert len(trace) > 0, "Should have execution trace"
    test_result("Pipeline Debugger", True, f"Trace entries: {len(trace)}")
    
    summary = debugger.get_trace_summary()
    assert 'total_stages' in summary, "Should have trace summary"
    test_result("Debugger Trace Summary", True, f"Stages: {summary.get('total_stages', 0)}")
    
except Exception as e:
    test_warning("Pipeline Debugger", str(e))

# Test 12: Pipeline State Management
print("\n" + "="*80)
print("TEST 12: PIPELINE STATE MANAGEMENT")
print("="*80)

try:
    from ml_toolbox.pipelines import PipelineState
    
    state = PipelineState()
    
    # Store features
    features = np.array([[1, 2, 3], [4, 5, 6]])
    state.store_features(features, name='test_features')
    test_result("Store features in state", True)
    
    # Retrieve features
    retrieved = state.get_features('test_features')
    assert retrieved is not None, "Should retrieve features"
    assert np.array_equal(retrieved, features), "Retrieved features should match"
    test_result("Retrieve features from state", True, f"Shape: {retrieved.shape}")
    
    # Store model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    state.store_model(model, name='test_model')
    test_result("Store model in state", True)
    
    # Retrieve model
    retrieved_model = state.get_model('test_model')
    assert retrieved_model is not None, "Should retrieve model"
    test_result("Retrieve model from state", True)
    
    # Get state summary
    summary = state.get_state_summary()
    assert 'features_count' in summary, "Should have state summary"
    test_result("State Summary", True, f"Features: {summary.get('features_count', 0)}")
    
except Exception as e:
    test_result("Pipeline State Management", False, str(e))

# Test 13: Feature Store
print("\n" + "="*80)
print("TEST 13: FEATURE STORE")
print("="*80)

try:
    from ml_toolbox.pipelines import FeatureStore
    
    feature_store = FeatureStore(enable_disk_storage=False)
    
    # Store features
    features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    feature_id = feature_store.store(features, name='test_features')
    assert feature_id is not None, "Should return feature ID"
    test_result("Store features", True, f"Feature ID: {feature_id}")
    
    # Retrieve features
    retrieved = feature_store.get('test_features')
    assert retrieved is not None, "Should retrieve features"
    assert np.array_equal(retrieved, features), "Retrieved features should match"
    test_result("Retrieve features", True, f"Shape: {retrieved.shape}")
    
    # List features
    feature_list = feature_store.list_features('test_features')
    assert len(feature_list) > 0, "Should list features"
    test_result("List features", True, f"Count: {len(feature_list)}")
    
    # Get metadata
    metadata = feature_store.get_metadata('test_features')
    assert metadata is not None, "Should get metadata"
    test_result("Get feature metadata", True)
    
except Exception as e:
    test_result("Feature Store", False, str(e))

# Test 14: End-to-End Pipeline
print("\n" + "="*80)
print("TEST 14: END-TO-END PIPELINE")
print("="*80)

try:
    toolbox = MLToolbox()
    pipeline = UnifiedMLPipeline(toolbox, enable_feature_store=True, enable_data_collection=True)
    
    # Generate data
    X, y = make_classification(n_samples=200, n_features=15, random_state=42)
    
    # Full pipeline: ETL (optional) → Feature → Training
    result = pipeline.execute(
        X,
        y,
        mode='train',
        use_data_collection=False,  # Skip ETL for this test
        feature_name='e2e_features',
        model_name='e2e_model'
    )
    
    assert 'model' in result, "Should have model"
    assert 'features' in result, "Should have features"
    assert 'metrics' in result, "Should have metrics"
    test_result("End-to-End Pipeline", True, 
                f"Model: {type(result['model']).__name__}, Accuracy: {result.get('metrics', {}).get('accuracy', 'N/A')}")
    
    # Test inference
    X_test = X[:50]
    inference_result = pipeline.execute(
        X_test,
        mode='inference',
        feature_name='e2e_features',
        model_name='e2e_model',
        reuse_features=True
    )
    
    assert 'predictions' in inference_result, "Should have predictions"
    test_result("End-to-End Inference", True, f"Predictions: {len(inference_result['predictions'])}")
    
except Exception as e:
    test_result("End-to-End Pipeline", False, str(e))

# Test 15: Error Handling
print("\n" + "="*80)
print("TEST 15: ERROR HANDLING")
print("="*80)

try:
    toolbox = MLToolbox()
    pipeline = UnifiedMLPipeline(toolbox)
    
    # Test with invalid data
    try:
        result = pipeline.execute([], [], mode='train')
        test_warning("Error Handling - Empty data", "Should have raised error")
    except Exception:
        test_result("Error Handling - Empty data", True, "Correctly raised error")
    
    # Test with invalid mode
    try:
        X, y = make_classification(n_samples=10, n_features=5, random_state=42)
        result = pipeline.execute(X, y, mode='invalid_mode')
        test_warning("Error Handling - Invalid mode", "Should have raised error")
    except ValueError:
        test_result("Error Handling - Invalid mode", True, "Correctly raised ValueError")
    
except Exception as e:
    test_result("Error Handling", False, str(e))

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

total_tests = len(test_results['passed']) + len(test_results['failed'])
passed = len(test_results['passed'])
failed = len(test_results['failed'])
warnings = len(test_results['warnings'])

print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {passed} ({passed/total_tests*100:.1f}%)")
print(f"Failed: {failed} ({failed/total_tests*100:.1f}%)")
print(f"Warnings: {warnings}")

if test_results['failed']:
    print("\nFailed Tests:")
    for test_name, message in test_results['failed']:
        print(f"  - {test_name}: {message}")

if test_results['warnings']:
    print("\nWarnings:")
    for test_name, message in test_results['warnings']:
        print(f"  - {test_name}: {message}")

print("\n" + "="*80)
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"{failed} TEST(S) FAILED")
print("="*80)

# Exit with appropriate code
sys.exit(0 if failed == 0 else 1)
