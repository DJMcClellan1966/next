"""
Profile ML Toolbox Components
Comprehensive profiling of all ML Toolbox operations
"""
import sys
from pathlib import Path
import time
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from ml_profiler import MLProfiler, ProfiledMLToolbox, profile_ml_pipeline

try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    warnings.warn("ML Toolbox not available")


def profile_data_preprocessing():
    """Profile data preprocessing operations"""
    print("Profiling Data Preprocessing...")
    
    profiler = MLProfiler()
    
    try:
        toolbox = MLToolbox()
        preprocessor = toolbox.data.get_advanced_data_preprocessor()
        
        # Create sample data
        import numpy as np
        import pandas as pd
        
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.rand(1000),
            'feature2': np.random.rand(1000),
            'feature3': np.random.rand(1000),
            'target': np.random.randint(0, 2, 1000)
        })
        
        # Profile preprocessing operations
        with profiler.profile_pipeline('data_preprocessing'):
            # Clean data
            @profiler.profile_function
            def clean_data():
                return preprocessor.clean_data(data.copy())
            
            cleaned = clean_data()
            
            # Transform data
            @profiler.profile_function
            def transform_data():
                return preprocessor.transform_data(cleaned.copy())
            
            transformed = transform_data()
            
            # Feature engineering
            @profiler.profile_function
            def engineer_features():
                return preprocessor.engineer_features(transformed.copy())
            
            engineered = engineer_features()
        
        return profiler
    
    except Exception as e:
        print(f"Error profiling preprocessing: {e}")
        return profiler


def profile_model_training():
    """Profile model training operations"""
    print("Profiling Model Training...")
    
    profiler = MLProfiler()
    
    try:
        toolbox = MLToolbox()
        
        # Create sample data
        import numpy as np
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
        
        # Profile training operations
        with profiler.profile_pipeline('model_training'):
            # Get simple ML tasks
            simple_ml = toolbox.algorithms.get_simple_ml_tasks()
            
            @profiler.profile_function
            def train_classifier():
                return simple_ml.train_classifier(X, y, model_type='random_forest')
            
            result = train_classifier()
            
            # Profile prediction
            @profiler.profile_function
            def predict():
                return result['model'].predict(X[:100])
            
            predictions = predict()
        
        return profiler
    
    except Exception as e:
        print(f"Error profiling training: {e}")
        return profiler


def profile_feature_selection():
    """Profile feature selection operations"""
    print("Profiling Feature Selection...")
    
    profiler = MLProfiler()
    
    try:
        toolbox = MLToolbox()
        
        # Create sample data
        import numpy as np
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=500, n_features=50, n_informative=10, random_state=42)
        
        # Profile feature selection
        with profiler.profile_pipeline('feature_selection'):
            selector = toolbox.algorithms.get_information_theoretic_feature_selector()
            
            @profiler.profile_function
            def select_features():
                if hasattr(selector, 'select_features'):
                    return selector.select_features(X, y, k=10)
                else:
                    return []
            
            selected = select_features()
        
        return profiler
    
    except Exception as e:
        print(f"Error profiling feature selection: {e}")
        return profiler


def profile_ensemble_learning():
    """Profile ensemble learning operations"""
    print("Profiling Ensemble Learning...")
    
    profiler = MLProfiler()
    
    try:
        toolbox = MLToolbox()
        
        # Create sample data
        import numpy as np
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        
        # Profile ensemble
        with profiler.profile_pipeline('ensemble_learning'):
            ensemble = toolbox.algorithms.get_ensemble()
            
            @profiler.profile_function
            def create_ensemble():
                if hasattr(ensemble, 'create_ensemble'):
                    return ensemble.create_ensemble(X, y, methods=['random_forest', 'svm'])
                else:
                    return None
            
            result = create_ensemble()
        
        return profiler
    
    except Exception as e:
        print(f"Error profiling ensemble: {e}")
        return profiler


def profile_full_pipeline():
    """Profile full ML pipeline"""
    print("Profiling Full ML Pipeline...")
    
    profiler = MLProfiler()
    
    try:
        toolbox = MLToolbox()
        
        # Create sample data
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
        data = pd.DataFrame(X)
        data['target'] = y
        
        # Profile full pipeline
        with profiler.profile_pipeline('full_ml_pipeline'):
            # Step 1: Preprocessing
            step_start = time.perf_counter()
            preprocessor = toolbox.data.get_advanced_data_preprocessor()
            cleaned = preprocessor.clean_data(data.copy())
            profiler.pipeline_times['full_ml_pipeline']['steps'].append({
                'name': 'data_cleaning',
                'time': time.perf_counter() - step_start
            })
            
            # Step 2: Feature engineering
            step_start = time.perf_counter()
            engineered = preprocessor.engineer_features(cleaned.copy())
            profiler.pipeline_times['full_ml_pipeline']['steps'].append({
                'name': 'feature_engineering',
                'time': time.perf_counter() - step_start
            })
            
            # Step 3: Model training
            step_start = time.perf_counter()
            simple_ml = toolbox.algorithms.get_simple_ml_tasks()
            X_train = engineered.drop(columns=['target']).values
            y_train = engineered['target'].values
            result = simple_ml.train_classifier(X_train, y_train, model_type='random_forest')
            profiler.pipeline_times['full_ml_pipeline']['steps'].append({
                'name': 'model_training',
                'time': time.perf_counter() - step_start
            })
            
            # Step 4: Prediction
            step_start = time.perf_counter()
            predictions = result['model'].predict(X_train[:100])
            profiler.pipeline_times['full_ml_pipeline']['steps'].append({
                'name': 'prediction',
                'time': time.perf_counter() - step_start
            })
        
        return profiler
    
    except Exception as e:
        print(f"Error profiling full pipeline: {e}")
        import traceback
        traceback.print_exc()
        return profiler


def run_comprehensive_profiling():
    """Run comprehensive profiling of ML Toolbox"""
    print("="*80)
    print("ML TOOLBOX COMPREHENSIVE PROFILING")
    print("="*80)
    print()
    
    all_profilers = {}
    
    # Profile each component
    components = [
        ('data_preprocessing', profile_data_preprocessing),
        ('model_training', profile_model_training),
        ('feature_selection', profile_feature_selection),
        ('ensemble_learning', profile_ensemble_learning),
        ('full_pipeline', profile_full_pipeline)
    ]
    
    for name, profile_func in components:
        try:
            profiler = profile_func()
            all_profilers[name] = profiler
            print(f"✅ {name} profiling complete")
        except Exception as e:
            print(f"❌ {name} profiling failed: {e}")
        print()
    
    # Combine all profiling data
    combined_profiler = MLProfiler()
    
    for name, profiler in all_profilers.items():
        # Merge function times
        for func_name, times in profiler.function_times.items():
            combined_profiler.function_times[func_name].extend(times)
        
        # Merge call counts
        for func_name, count in profiler.call_counts.items():
            combined_profiler.call_counts[func_name] += count
        
        # Merge pipeline times
        for pipeline_name, pipeline_data in profiler.pipeline_times.items():
            combined_profiler.pipeline_times[f"{name}_{pipeline_name}"] = pipeline_data
    
    # Generate comprehensive report
    print("="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    print()
    
    report = combined_profiler.generate_report('ml_toolbox_profiling_report.txt')
    print(report)
    
    # Export data
    combined_profiler.export_data('ml_toolbox_profiling_data.json')
    print("\n✅ Profiling data exported to ml_toolbox_profiling_data.json")
    
    # Show top bottlenecks
    bottlenecks = combined_profiler.identify_bottlenecks()
    if bottlenecks:
        print("\n" + "="*80)
        print("TOP BOTTLENECKS")
        print("="*80)
        for i, bottleneck in enumerate(bottlenecks[:10], 1):
            print(f"\n{i}. {bottleneck['function']} [{bottleneck['priority'].upper()}]")
            print(f"   Time: {bottleneck['total_time']:.4f}s ({bottleneck['percentage']:.2f}%)")
            print(f"   Calls: {bottleneck['call_count']:,}")
            print(f"   Recommendations:")
            for rec in bottleneck['recommendations']:
                print(f"     • {rec}")
    
    return combined_profiler


if __name__ == '__main__':
    profiler = run_comprehensive_profiling()
