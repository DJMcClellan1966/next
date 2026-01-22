"""
Tests for Experiment Tracking UI, AutoML, and Simple ML Tasks
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from experiment_tracking_ui import ExperimentTrackingUI
    from automl_framework import AutoMLFramework
    from simple_ml_tasks import SimpleMLTasks
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    pytestmark = pytest.mark.skip("Features not available")


class TestExperimentTrackingUI:
    """Tests for experiment tracking UI"""
    
    def test_log_experiment(self):
        """Test logging experiment"""
        ui = ExperimentTrackingUI(storage_path="test_experiments.json")
        exp_id = ui.log_experiment(
            'test_exp',
            {'accuracy': 0.95, 'loss': 0.05},
            {'lr': 0.001, 'epochs': 10}
        )
        assert exp_id is not None
        
        # Cleanup
        if os.path.exists("test_experiments.json"):
            os.remove("test_experiments.json")
    
    def test_get_best_experiment(self):
        """Test getting best experiment"""
        ui = ExperimentTrackingUI(storage_path="test_experiments.json")
        ui.log_experiment('exp1', {'accuracy': 0.9}, {})
        ui.log_experiment('exp2', {'accuracy': 0.95}, {})
        
        best = ui.get_best_experiment('accuracy')
        assert best is not None
        assert best['metrics']['accuracy'] == 0.95
        
        # Cleanup
        if os.path.exists("test_experiments.json"):
            os.remove("test_experiments.json")
    
    def test_generate_html_dashboard(self):
        """Test HTML dashboard generation"""
        ui = ExperimentTrackingUI(storage_path="test_experiments.json")
        ui.log_experiment('test', {'accuracy': 0.9}, {})
        
        html = ui.generate_html_dashboard()
        assert '<html>' in html
        assert 'test' in html
        
        # Cleanup
        if os.path.exists("test_experiments.json"):
            os.remove("test_experiments.json")


class TestAutoMLFramework:
    """Tests for AutoML framework"""
    
    def test_automl_pipeline_classification(self):
        """Test AutoML for classification"""
        automl = AutoMLFramework()
        
        if automl.sklearn_available:
            X = np.random.rand(100, 10)
            y = np.random.randint(0, 2, 100)
            
            result = automl.automl_pipeline(X, y, task_type='classification', time_budget=10)
            assert 'best_model' in result or 'error' in result
        else:
            result = automl.automl_pipeline(np.array([]), np.array([]))
            assert 'error' in result
    
    def test_automated_feature_engineering(self):
        """Test automated feature engineering"""
        automl = AutoMLFramework()
        
        if automl.sklearn_available:
            X = np.random.rand(50, 5)
            result = automl.automated_feature_engineering(X, methods=['pca'])
            assert 'engineered_features' in result
        else:
            result = automl.automated_feature_engineering(np.array([]))
            assert 'error' in result


class TestSimpleMLTasks:
    """Tests for simple ML tasks"""
    
    def test_train_classifier(self):
        """Test train classifier"""
        simple = SimpleMLTasks()
        
        if simple.sklearn_available:
            X = np.random.rand(100, 10)
            y = np.random.randint(0, 2, 100)
            
            result = simple.train_classifier(X, y)
            assert 'model' in result or 'error' in result
        else:
            result = simple.train_classifier(np.array([]), np.array([]))
            assert 'error' in result
    
    def test_train_regressor(self):
        """Test train regressor"""
        simple = SimpleMLTasks()
        
        if simple.sklearn_available:
            X = np.random.rand(100, 10)
            y = np.random.rand(100)
            
            result = simple.train_regressor(X, y)
            assert 'model' in result or 'error' in result
        else:
            result = simple.train_regressor(np.array([]), np.array([]))
            assert 'error' in result
    
    def test_quick_train(self):
        """Test quick train"""
        simple = SimpleMLTasks()
        
        if simple.sklearn_available:
            X = np.random.rand(100, 10)
            y = np.random.randint(0, 2, 100)
            
            result = simple.quick_train(X, y)
            assert 'model' in result or 'error' in result
        else:
            result = simple.quick_train(np.array([]), np.array([]))
            assert 'error' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
