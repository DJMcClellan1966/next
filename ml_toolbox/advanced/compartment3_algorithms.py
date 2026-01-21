"""
Advanced Compartment 3: Algorithms
Advanced ML algorithms, evaluation, tuning, and optimization
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AdvancedAlgorithmsCompartment:
    """
    Advanced Compartment 3: Algorithms
    
    Components for advanced ML algorithms:
    - ML Evaluation: Comprehensive model evaluation
    - Hyperparameter Tuning: Advanced optimization
    - Ensemble Learning: Multiple ensemble methods
    - Advanced algorithm utilities
    """
    
    def __init__(self):
        self.components = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all algorithms compartment components"""
        
        # ML Evaluation
        try:
            from ml_evaluation import MLEvaluator, HyperparameterTuner, PreprocessorOptimizer
            self.components['MLEvaluator'] = MLEvaluator
            self.components['HyperparameterTuner'] = HyperparameterTuner
            self.components['PreprocessorOptimizer'] = PreprocessorOptimizer
        except ImportError as e:
            print(f"Warning: Could not import ML evaluation: {e}")
        
        # Ensemble Learning
        try:
            from ensemble_learning import EnsembleLearner, PreprocessorEnsemble
            self.components['EnsembleLearner'] = EnsembleLearner
            self.components['PreprocessorEnsemble'] = PreprocessorEnsemble
        except ImportError as e:
            print(f"Warning: Could not import ensemble learning: {e}")
        
        # Add component descriptions
        self.component_descriptions = {
            'MLEvaluator': {
                'description': 'Comprehensive ML model evaluation',
                'features': [
                    'Cross-validation',
                    'Multiple metrics (accuracy, precision, recall, F1, MSE, MAE, R²)',
                    'Overfitting detection',
                    'Learning curves',
                    'Train/test splits'
                ],
                'location': 'ml_evaluation.py',
                'category': 'Evaluation',
                'placement': 'Advanced Compartment 3: Algorithms'
            },
            'HyperparameterTuner': {
                'description': 'Hyperparameter optimization',
                'features': [
                    'Grid search',
                    'Random search',
                    'Model parameter tuning',
                    'Preprocessor parameter tuning'
                ],
                'location': 'ml_evaluation.py',
                'category': 'Tuning',
                'placement': 'Advanced Compartment 3: Algorithms'
            },
            'EnsembleLearner': {
                'description': 'Ensemble learning methods',
                'features': [
                    'Voting ensembles',
                    'Bagging',
                    'Boosting',
                    'Stacking',
                    'Multiple algorithms'
                ],
                'location': 'ensemble_learning.py',
                'category': 'Ensemble',
                'placement': 'Advanced Compartment 3: Algorithms'
            },
            'PreprocessorEnsemble': {
                'description': 'Ensemble of preprocessing strategies',
                'features': [
                    'Multiple preprocessors',
                    'Voting on preprocessing',
                    'Combined preprocessing'
                ],
                'location': 'ensemble_learning.py',
                'category': 'Ensemble',
                'placement': 'Advanced Compartment 3: Algorithms'
            }
        }
    
    def get_evaluator(self):
        """
        Get an ML Evaluator instance
        
        Returns:
            MLEvaluator instance
        """
        if 'MLEvaluator' in self.components:
            return self.components['MLEvaluator']()
        else:
            raise ImportError("ML Evaluator not available")
    
    def get_tuner(self):
        """
        Get a Hyperparameter Tuner instance
        
        Returns:
            HyperparameterTuner instance
        """
        if 'HyperparameterTuner' in self.components:
            return self.components['HyperparameterTuner']()
        else:
            raise ImportError("Hyperparameter Tuner not available")
    
    def get_ensemble(self):
        """
        Get an Ensemble Learner instance
        
        Returns:
            EnsembleLearner instance
        """
        if 'EnsembleLearner' in self.components:
            return self.components['EnsembleLearner']()
        else:
            raise ImportError("Ensemble Learner not available")
    
    def list_components(self):
        """List all available components in this compartment"""
        print("="*80)
        print("ADVANCED COMPARTMENT 3: ALGORITHMS")
        print("="*80)
        print("\nComponents:")
        for name, component in self.components.items():
            desc = self.component_descriptions.get(name, {})
            print(f"\n{name}:")
            print(f"  Description: {desc.get('description', 'N/A')}")
            print(f"  Location: {desc.get('location', 'N/A')}")
            print(f"  Category: {desc.get('category', 'N/A')}")
            print(f"  Placement: {desc.get('placement', 'N/A')}")
            if 'features' in desc:
                print(f"  Features:")
                for feature in desc['features']:
                    print(f"    - {feature}")
        print("\n" + "="*80)
    
    def get_info(self):
        """Get information about this compartment"""
        return {
            'name': 'Advanced Compartment 3: Algorithms',
            'description': 'Advanced ML algorithms, evaluation, tuning',
            'components': list(self.components.keys()),
            'component_count': len(self.components)
        }
