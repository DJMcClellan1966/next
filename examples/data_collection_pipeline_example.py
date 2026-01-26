"""
Data Collection Pipeline Example

Demonstrates ETL pattern:
- Extract: From user inputs and NoSQL databases
- Transform: Clean, validate, structure data
- Load: Output to Feature Pipeline

Installation:
    pip install -e .  # From project root

Usage:
    python examples/data_collection_pipeline_example.py
"""
import numpy as np
from sklearn.datasets import make_classification

# Import ML Toolbox
try:
    from ml_toolbox import MLToolbox, UnifiedMLPipeline
    from ml_toolbox.pipelines import DataCollectionPipeline
    print("[OK] ML Toolbox and DataCollectionPipeline imported successfully")
except ImportError as e:
    print(f"[ERROR] Error importing: {e}")
    print("\n[TIP] Try installing the package first:")
    print("   pip install -e .")
    raise


def example_1_user_input():
    """Example 1: Extract from user input"""
    print("\n" + "="*80)
    print("EXAMPLE 1: EXTRACT FROM USER INPUT")
    print("="*80)
    
    # Simulate user input (various formats)
    user_inputs = [
        [1.0, 2.0, 3.0, 4.0, 5.0],  # List
        {'feature1': 1.0, 'feature2': 2.0, 'feature3': 3.0},  # Dict
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # Numpy array
    ]
    
    toolbox = MLToolbox()
    data_collection = DataCollectionPipeline(toolbox)
    
    for i, user_input in enumerate(user_inputs, 1):
        print(f"\n{i}. Processing user input: {type(user_input).__name__}")
        try:
            result = data_collection.execute(
                user_input,
                source_type='user_input',
                feature_name=f'user_input_{i}'
            )
            print(f"   [OK] Extracted shape: {result.shape}")
            print(f"   Sample data: {result[:3] if len(result) > 3 else result}")
        except Exception as e:
            print(f"   [ERROR] {e}")
    
    return result


def example_2_nosql_simulation():
    """Example 2: Extract from NoSQL database (simulated)"""
    print("\n" + "="*80)
    print("EXAMPLE 2: EXTRACT FROM NOSQL DATABASE (SIMULATED)")
    print("="*80)
    
    # Simulate NoSQL query results (MongoDB-like documents)
    nosql_results = [
        {'_id': '1', 'temperature': 25.5, 'humidity': 60.0, 'pressure': 1013.25, 'timestamp': '2024-01-01'},
        {'_id': '2', 'temperature': 26.0, 'humidity': 58.0, 'pressure': 1012.50, 'timestamp': '2024-01-02'},
        {'_id': '3', 'temperature': 24.5, 'humidity': 62.0, 'pressure': 1014.00, 'timestamp': '2024-01-03'},
        {'_id': '4', 'temperature': 27.0, 'humidity': 55.0, 'pressure': 1011.75, 'timestamp': '2024-01-04'},
        {'_id': '5', 'temperature': 25.0, 'humidity': 59.0, 'pressure': 1013.50, 'timestamp': '2024-01-05'},
    ]
    
    toolbox = MLToolbox()
    data_collection = DataCollectionPipeline(toolbox)
    
    print("\n1. Processing NoSQL query results...")
    try:
        result = data_collection.execute(
            nosql_results,
            source_type='nosql',
            feature_name='nosql_sensor_data'
        )
        print(f"   [OK] Extracted shape: {result.shape}")
        print(f"   Data preview:")
        print(f"   {result[:3]}")
        print(f"   Features: temperature, humidity, pressure (metadata fields excluded)")
    except Exception as e:
        print(f"   [ERROR] {e}")
    
    return result


def example_3_etl_to_feature_pipeline():
    """Example 3: Complete ETL to Feature Pipeline"""
    print("\n" + "="*80)
    print("EXAMPLE 3: COMPLETE ETL TO FEATURE PIPELINE")
    print("="*80)
    
    # User input data
    user_data = {
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'feature3': [100.0, 200.0, 300.0, 400.0, 500.0]
    }
    
    toolbox = MLToolbox()
    data_collection = DataCollectionPipeline(toolbox)
    
    print("\n1. Executing ETL pipeline...")
    collected_data = data_collection.execute(
        user_data,
        source_type='user_input',
        feature_name='etl_features',
        remove_nulls=True,
        handle_missing='fill_mean'
    )
    
    print(f"   [OK] ETL completed. Shape: {collected_data.shape}")
    
    # Now feed to Feature Pipeline
    print("\n2. Feeding to Feature Pipeline...")
    from ml_toolbox.pipelines import FeaturePipeline
    
    feature_pipeline = FeaturePipeline(toolbox)
    features = feature_pipeline.execute(collected_data, feature_name='etl_features')
    
    print(f"   [OK] Feature Pipeline completed. Shape: {features.shape}")
    
    return features


def example_4_unified_pipeline_with_etl():
    """Example 4: Unified Pipeline with Data Collection"""
    print("\n" + "="*80)
    print("EXAMPLE 4: UNIFIED PIPELINE WITH DATA COLLECTION")
    print("="*80)
    
    # Generate sample data (simulating user input)
    # Use list of lists to ensure consistent shape
    X_raw = [
        [25, 50000, 75],
        [30, 60000, 80],
        [35, 70000, 85],
        [40, 80000, 90],
        [45, 90000, 95]
    ]
    y = np.array([0, 1, 0, 1, 1])
    
    toolbox = MLToolbox()
    
    # Create unified pipeline with data collection enabled
    pipeline = UnifiedMLPipeline(
        toolbox,
        enable_feature_store=True,
        enable_data_collection=True
    )
    
    print("\n1. Executing unified pipeline with ETL...")
    result = pipeline.execute(
        X_raw,  # Raw user input (dict)
        y,
        mode='train',
        use_data_collection=True,  # Enable ETL
        source_type='user_input',
        feature_name='unified_etl_features',
        model_name='unified_etl_model'
    )
    
    print(f"\n[OK] Unified pipeline with ETL completed!")
    print(f"   Model ID: {result.get('model_id', 'N/A')}")
    print(f"   Metrics: {result.get('metrics', {})}")
    print(f"   Features shape: {result['features'].shape}")
    
    return result


def example_5_nosql_integration():
    """Example 5: NoSQL Integration (with mock client)"""
    print("\n" + "="*80)
    print("EXAMPLE 5: NOSQL INTEGRATION (MOCK)")
    print("="*80)
    
    # Mock NoSQL client
    class MockNoSQLClient:
        def __init__(self):
            self.data = {
                'sensors': [
                    {'_id': '1', 'temp': 25.5, 'humidity': 60.0},
                    {'_id': '2', 'temp': 26.0, 'humidity': 58.0},
                    {'_id': '3', 'temp': 24.5, 'humidity': 62.0},
                ]
            }
        
        def __getitem__(self, collection):
            return MockCollection(self.data.get(collection, []))
    
    class MockCollection:
        def __init__(self, data):
            self.data = data
        
        def find(self, query=None):
            if query:
                # Simple filtering
                return [doc for doc in self.data if all(doc.get(k) == v for k, v in query.items())]
            return self.data
    
    # Use mock client
    mock_client = MockNoSQLClient()
    
    toolbox = MLToolbox()
    data_collection = DataCollectionPipeline(toolbox)
    
    print("\n1. Querying NoSQL database...")
    try:
        result = data_collection.execute(
            None,  # Will use nosql_client
            source_type='nosql',
            nosql_client=mock_client,
            nosql_collection='sensors',
            nosql_query=None,  # Get all
            feature_name='nosql_sensors'
        )
        print(f"   [OK] Extracted from NoSQL. Shape: {result.shape}")
        print(f"   Data: {result}")
    except Exception as e:
        print(f"   [ERROR] {e}")
        print(f"   Note: This is a mock example. Real NoSQL integration requires actual client.")
    
    return result if 'result' in locals() else None


def main():
    """Run all examples"""
    print("="*80)
    print("DATA COLLECTION PIPELINE EXAMPLES (ETL PATTERN)")
    print("="*80)
    print("\nDemonstrates Extract -> Transform -> Load pattern")
    print("for collecting data from user inputs and NoSQL databases.")
    
    examples = {
        'User Input Extraction': example_1_user_input,
        'NoSQL Extraction (Simulated)': example_2_nosql_simulation,
        'ETL to Feature Pipeline': example_3_etl_to_feature_pipeline,
        'Unified Pipeline with ETL': example_4_unified_pipeline_with_etl,
        'NoSQL Integration (Mock)': example_5_nosql_integration
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
