"""
Phase 3 Examples
Feature Store, Monitoring Dashboard, and Model Compression

Demonstrates:
- Feature Store usage
- Monitoring Dashboard setup
- Model Compression techniques
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available")
    exit(1)

try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    print("Warning: ML Toolbox not available")
    exit(1)


def example_feature_store():
    """Example: Feature Store"""
    print("=" * 60)
    print("Example 1: Feature Store")
    print("=" * 60)
    
    # Initialize toolbox
    toolbox = MLToolbox(include_mlops=True)
    
    # Get feature store
    feature_store = toolbox.mlops.get_feature_store(storage_dir='feature_store_example')
    
    # Generate sample features
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    
    # Register features
    print("\nRegistering features...")
    version = feature_store.register_feature(
        feature_name='classification_features',
        features=X,
        metadata={'source': 'synthetic_data', 'n_samples': len(X)},
        tags=['classification', 'synthetic']
    )
    print(f"Registered features version: {version}")
    
    # Retrieve features
    print("\nRetrieving features...")
    feature_data = feature_store.get_feature('classification_features', version)
    print(f"Features shape: {feature_data['features'].shape}")
    print(f"Metadata: {feature_data['metadata']}")
    
    # List features
    print("\nListing all features...")
    features = feature_store.list_features()
    for feat in features:
        print(f"  - {feat['feature_name']}: {feat['latest_version']}")
    
    # Get feature lineage
    print("\nFeature lineage...")
    lineage = feature_store.get_feature_lineage('classification_features')
    print(f"Created at: {lineage['created_at']}")
    print(f"Tags: {lineage['tags']}")
    
    print("\n✓ Feature Store example complete\n")


def example_monitoring_dashboard():
    """Example: Monitoring Dashboard"""
    print("=" * 60)
    print("Example 2: Monitoring Dashboard")
    print("=" * 60)
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = (
        X[:400], X[400:], y[:400], y[400:]
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize toolbox
    toolbox = MLToolbox(include_mlops=True)
    
    # Create model monitor
    monitor = toolbox.mlops.get_model_monitor(
        model=model,
        reference_data=X_train,
        reference_labels=y_train,
        model_name='dashboard_model'
    )
    
    # Create dashboard
    print("\nCreating monitoring dashboard...")
    dashboard = toolbox.mlops.get_monitoring_dashboard(
        model_monitors={'dashboard_model': monitor},
        port=8080
    )
    
    print("Dashboard created!")
    print("To start dashboard, run:")
    print("  dashboard.run(host='0.0.0.0', port=8080)")
    print("Then visit: http://localhost:8080")
    
    # Add some monitoring data
    monitor.monitor(X_test, y_test)
    
    print("\n✓ Monitoring Dashboard example complete")
    print("  Note: Dashboard server not started in example\n")


def example_model_compression():
    """Example: Model Compression"""
    print("=" * 60)
    print("Example 3: Model Compression")
    print("=" * 60)
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Initialize toolbox
    toolbox = MLToolbox(include_mlops=True)
    
    # Get model compressor
    compressor = toolbox.mlops.get_model_compressor()
    
    # Estimate model size
    print("\nEstimating model size...")
    size_info = compressor.estimate_model_size(model)
    print(f"Serialized size: {size_info['serialized_size_mb']:.2f} MB")
    print(f"Model type: {size_info['model_type']}")
    
    # Compress model (quantization)
    print("\nCompressing model (quantization)...")
    compressed = compressor.compress_model(model, method='quantization', precision='int8')
    if 'error' not in compressed:
        print(f"Compression ratio: {compressed['compression_ratio']:.2f}")
        print(f"Original size: {compressed['original_size_mb']:.2f} MB")
        print(f"Compressed size: {compressed['compressed_size_mb']:.2f} MB")
    else:
        print(f"Compression note: {compressed.get('note', compressed.get('error'))}")
    
    # Try pruning (for tree models)
    print("\nPruning model...")
    pruned = compressor.compress_model(model, method='pruning', pruning_ratio=0.3)
    if 'error' not in pruned:
        if 'suggested_max_depth' in pruned:
            print(f"Suggested max_depth: {pruned['suggested_max_depth']}")
            print(f"Original depth: {pruned['original_depth']}")
        else:
            print(f"Pruning ratio: {pruned['compression_ratio']:.2f}")
    else:
        print(f"Pruning: {pruned.get('error', 'Not applicable')}")
    
    print("\n✓ Model Compression example complete\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Phase 3 Examples - Feature Store, Dashboard, Compression")
    print("=" * 60 + "\n")
    
    if not SKLEARN_AVAILABLE or not TOOLBOX_AVAILABLE:
        print("Required dependencies not available")
        exit(1)
    
    try:
        example_feature_store()
        example_monitoring_dashboard()
        example_model_compression()
        
        print("=" * 60)
        print("All Phase 3 examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
