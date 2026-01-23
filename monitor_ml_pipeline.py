"""
Monitor ML Pipeline
Comprehensive monitoring of ML Toolbox pipeline with CPU/memory tracking
"""
import sys
from pathlib import Path
import time
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    from pipeline_bottleneck_monitor import PipelineBottleneckMonitor, monitor_function
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False
    warnings.warn("Pipeline monitor not available")

try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    warnings.warn("ML Toolbox not available")


def monitor_data_preprocessing():
    """Monitor data preprocessing pipeline"""
    if not MONITOR_AVAILABLE or not TOOLBOX_AVAILABLE:
        print("Required components not available")
        return
    
    monitor = PipelineBottleneckMonitor(sample_interval=0.05)
    monitor.start_monitoring()
    
    print("="*80)
    print("MONITORING DATA PREPROCESSING PIPELINE")
    print("="*80)
    print()
    
    # Create toolbox
    toolbox = MLToolbox()
    data = toolbox.data
    
    # Generate test data
    test_data = [
        "Python programming language tutorial",
        "Machine learning algorithms explained",
        "Deep learning neural networks guide",
        "Data science with Python",
        "Natural language processing techniques"
    ] * 100  # 500 items
    
    # Monitor preprocessing
    with monitor.monitor_pipeline_stage('full_preprocessing'):
        with monitor.monitor_pipeline_stage('get_preprocessor'):
            preprocessor = data.get_preprocessor(advanced=True)
        
        with monitor.monitor_pipeline_stage('preprocess_data'):
            results = preprocessor.preprocess(test_data, verbose=False)
    
    # Monitor individual operations
    @monitor_function(monitor, 'embedding_computation')
    def compute_embeddings():
        if hasattr(preprocessor, 'quantum_kernel') and preprocessor.quantum_kernel:
            embeddings = [preprocessor.quantum_kernel.embed(text) for text in test_data[:10]]
            return embeddings
        return []
    
    embeddings = compute_embeddings()
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Generate report
    print("\n" + "="*80)
    print("MONITORING RESULTS")
    print("="*80)
    print()
    
    report = monitor.generate_report()
    print(report)
    
    # Save report
    with open('pipeline_monitoring_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\nReport saved to: pipeline_monitoring_report.txt")
    
    return monitor


def monitor_ml_training():
    """Monitor ML training pipeline"""
    if not MONITOR_AVAILABLE or not TOOLBOX_AVAILABLE:
        print("Required components not available")
        return
    
    monitor = PipelineBottleneckMonitor(sample_interval=0.05)
    monitor.start_monitoring()
    
    print("="*80)
    print("MONITORING ML TRAINING PIPELINE")
    print("="*80)
    print()
    
    try:
        import numpy as np
        from sklearn.datasets import make_classification
        
        # Generate test data
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
        
        toolbox = MLToolbox()
        algorithms = toolbox.algorithms
        
        # Monitor training
        with monitor.monitor_pipeline_stage('ml_training'):
            with monitor.monitor_pipeline_stage('get_simple_ml_tasks'):
                simple = algorithms.get_simple_ml_tasks()
            
            with monitor.monitor_pipeline_stage('train_classifier'):
                result = simple.train_classifier(X, y, model_type='random_forest')
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Generate report
        print("\n" + "="*80)
        print("MONITORING RESULTS")
        print("="*80)
        print()
        
        report = monitor.generate_report()
        print(report)
        
        # Save report
        with open('ml_training_monitoring_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\nReport saved to: ml_training_monitoring_report.txt")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    return monitor


def monitor_comprehensive_pipeline():
    """Monitor comprehensive ML pipeline"""
    if not MONITOR_AVAILABLE or not TOOLBOX_AVAILABLE:
        print("Required components not available")
        return
    
    monitor = PipelineBottleneckMonitor(sample_interval=0.05)
    monitor.start_monitoring()
    
    print("="*80)
    print("COMPREHENSIVE ML PIPELINE MONITORING")
    print("="*80)
    print()
    
    toolbox = MLToolbox()
    
    # Test data
    test_texts = [
        "Python programming",
        "Machine learning",
        "Data science",
        "Deep learning"
    ] * 50  # 200 items
    
    try:
        import numpy as np
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    except:
        print("Could not generate test data")
        return
    
    # Monitor full pipeline
    with monitor.monitor_pipeline_stage('complete_pipeline'):
        # Data preprocessing
        with monitor.monitor_pipeline_stage('data_preprocessing'):
            data = toolbox.data
            preprocessor = data.get_preprocessor(advanced=True)
            preprocessed = preprocessor.preprocess(test_texts, verbose=False)
        
        # ML training
        with monitor.monitor_pipeline_stage('ml_training'):
            algorithms = toolbox.algorithms
            simple = algorithms.get_simple_ml_tasks()
            model_result = simple.train_classifier(X, y, model_type='random_forest')
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Generate comprehensive report
    print("\n" + "="*80)
    print("COMPREHENSIVE MONITORING RESULTS")
    print("="*80)
    print()
    
    report = monitor.generate_report()
    print(report)
    
    # Save report
    with open('comprehensive_pipeline_monitoring_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\nReport saved to: comprehensive_pipeline_monitoring_report.txt")
    
    # Identify bottlenecks
    bottlenecks = monitor.identify_bottlenecks(threshold_percent=5.0)
    if bottlenecks:
        print("\n" + "="*80)
        print("IDENTIFIED BOTTLENECKS (>5% of total time)")
        print("="*80)
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"\n{i}. {bottleneck['function']}")
            print(f"   Time: {bottleneck['total_time']:.3f}s ({bottleneck['percent_time']:.1f}% of total)")
            print(f"   Calls: {bottleneck['calls']}")
            print(f"   Peak Memory: {bottleneck['peak_memory_mb']:.1f} MB")
    
    return monitor


if __name__ == '__main__':
    print("ML Pipeline Monitoring")
    print("="*80)
    print()
    print("1. Data Preprocessing Monitoring")
    print("2. ML Training Monitoring")
    print("3. Comprehensive Pipeline Monitoring")
    print()
    
    # Run comprehensive monitoring
    monitor = monitor_comprehensive_pipeline()
    
    print("\n" + "="*80)
    print("MONITORING COMPLETE")
    print("="*80)
