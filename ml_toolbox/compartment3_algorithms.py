"""
Compartment 3: Algorithms
Machine learning models, evaluation, tuning, and ensembles

Optimizations:
- Parallel cross-validation
- Cached model evaluation
- Big O optimizations
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import functools

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import optimizations
try:
    from .optimizations import cache_result, get_global_monitor, ParallelProcessor
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False
    print("Warning: Optimizations not available")


class AlgorithmsCompartment:
    """
    Compartment 3: Algorithms
    
    Components for machine learning algorithms:
    - ML Evaluation: Cross-validation, metrics, overfitting detection
    - Hyperparameter Tuning: Grid search, random search
    - Ensemble Learning: Voting, bagging, boosting, stacking
    - Model utilities and wrappers
    """
    
    def __init__(self, n_jobs: Optional[int] = None):
        self.components = {}
        self.n_jobs = n_jobs or -1  # -1 means use all cores
        self._monitor = get_global_monitor() if OPTIMIZATIONS_AVAILABLE else None
        self._parallel = ParallelProcessor(n_workers=None) if OPTIMIZATIONS_AVAILABLE else None
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
        
        # Statistical Learning Methods
        try:
            from statistical_learning import (
                StatisticalEvaluator,
                StatisticalValidator,
                BayesianOptimizer,
                StatisticalFeatureSelector
            )
            self.components['StatisticalEvaluator'] = StatisticalEvaluator
            self.components['StatisticalValidator'] = StatisticalValidator
            self.components['BayesianOptimizer'] = BayesianOptimizer
            self.components['StatisticalFeatureSelector'] = StatisticalFeatureSelector
        except ImportError as e:
            print(f"Warning: Could not import statistical learning: {e}")
        
        # Andrew Ng ML Strategy
        try:
            from andrew_ng_ml_strategy import (
                AndrewNgMLStrategy,
                ErrorAnalyzer,
                BiasVarianceDiagnostic,
                LearningCurvesAnalyzer,
                ModelDebugger,
                SystematicModelSelector
            )
            self.components['AndrewNgMLStrategy'] = AndrewNgMLStrategy
            self.components['ErrorAnalyzer'] = ErrorAnalyzer
            self.components['BiasVarianceDiagnostic'] = BiasVarianceDiagnostic
            self.components['LearningCurvesAnalyzer'] = LearningCurvesAnalyzer
            self.components['ModelDebugger'] = ModelDebugger
            self.components['SystematicModelSelector'] = SystematicModelSelector
        except ImportError as e:
            print(f"Warning: Could not import Andrew Ng ML Strategy: {e}")
        
        # Kuhn/Johnson Resampling Methods
        try:
            from kuhn_johnson_resampling import (
                AdvancedResampler,
                RepeatedKFold,
                BootstrapResampler,
                TimeSeriesCV,
                GroupKFoldCV,
                NestedCV
            )
            self.components['AdvancedResampler'] = AdvancedResampler
            self.components['RepeatedKFold'] = RepeatedKFold
            self.components['BootstrapResampler'] = BootstrapResampler
        except ImportError as e:
            print(f"Warning: Could not import Kuhn/Johnson resampling: {e}")
        
        # Variable Importance Analysis
        try:
            from variable_importance import VariableImportanceAnalyzer
            self.components['VariableImportanceAnalyzer'] = VariableImportanceAnalyzer
        except ImportError as e:
            print(f"Warning: Could not import variable importance analyzer: {e}")
        
        # Performance Profiles
        try:
            from performance_profiles import PerformanceProfile
            self.components['PerformanceProfile'] = PerformanceProfile
        except ImportError as e:
            print(f"Warning: Could not import performance profiles: {e}")
        
        # Advanced Feature Selection
        try:
            from advanced_feature_selection import AdvancedFeatureSelector
            self.components['AdvancedFeatureSelector'] = AdvancedFeatureSelector
        except ImportError as e:
            print(f"Warning: Could not import advanced feature selection: {e}")
        
        # Model Calibration
        try:
            from model_calibration import ModelCalibrator
            self.components['ModelCalibrator'] = ModelCalibrator
        except ImportError as e:
            print(f"Warning: Could not import model calibrator: {e}")
        
        # Russell/Norvig Search-Based Feature Selection
        try:
            from search_based_feature_selection import SearchBasedFeatureSelector
            self.components['SearchBasedFeatureSelector'] = SearchBasedFeatureSelector
        except ImportError as e:
            print(f"Warning: Could not import search-based feature selection: {e}")
        
        # Russell/Norvig Simulated Annealing
        try:
            from simulated_annealing_optimizer import SimulatedAnnealingOptimizer
            self.components['SimulatedAnnealingOptimizer'] = SimulatedAnnealingOptimizer
        except ImportError as e:
            print(f"Warning: Could not import simulated annealing optimizer: {e}")
        
        # Russell/Norvig Bayesian Networks
        try:
            from bayesian_networks import BayesianNetworkAnalyzer
            self.components['BayesianNetworkAnalyzer'] = BayesianNetworkAnalyzer
        except ImportError as e:
            print(f"Warning: Could not import Bayesian network analyzer: {e}")
        
        # Russell/Norvig Constraint Satisfaction
        try:
            from constraint_satisfaction_optimizer import CSPOptimizer, Constraint, create_constraint
            self.components['CSPOptimizer'] = CSPOptimizer
            self.components['Constraint'] = Constraint
            self.components['create_constraint'] = create_constraint
        except ImportError as e:
            print(f"Warning: Could not import CSP optimizer: {e}")
        
        # Russell/Norvig Hidden Markov Models
        try:
            from hidden_markov_models import HMMAnalyzer
            self.components['HMMAnalyzer'] = HMMAnalyzer
        except ImportError as e:
            print(f"Warning: Could not import HMM analyzer: {e}")
        
        # Russell/Norvig Genetic Algorithms
        try:
            from genetic_algorithm_optimizer import GeneticAlgorithmOptimizer
            self.components['GeneticAlgorithmOptimizer'] = GeneticAlgorithmOptimizer
        except ImportError as e:
            print(f"Warning: Could not import genetic algorithm optimizer: {e}")
        
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
                'category': 'Evaluation'
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
                'category': 'Tuning'
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
                'category': 'Ensemble'
            },
            'PreprocessorEnsemble': {
                'description': 'Ensemble of preprocessing strategies',
                'features': [
                    'Multiple preprocessors',
                    'Voting on preprocessing',
                    'Combined preprocessing'
                ],
                'location': 'ensemble_learning.py',
                'category': 'Ensemble'
            },
            'StatisticalEvaluator': {
                'description': 'Statistical evaluation with uncertainty quantification',
                'features': [
                    'Confidence intervals for predictions',
                    'Prediction intervals',
                    'Bootstrap-based uncertainty',
                    'Uncertainty scores'
                ],
                'location': 'statistical_learning.py',
                'category': 'Statistical Learning'
            },
            'StatisticalValidator': {
                'description': 'Statistical validation methods',
                'features': [
                    'Permutation tests for model comparison',
                    'Bootstrap validation',
                    'Hypothesis testing',
                    'Statistical significance'
                ],
                'location': 'statistical_learning.py',
                'category': 'Statistical Learning'
            },
            'BayesianOptimizer': {
                'description': 'Bayesian hyperparameter optimization',
                'features': [
                    'Gaussian Process-based optimization',
                    'More efficient than grid/random search',
                    'Handles uncertainty',
                    'Better exploration/exploitation'
                ],
                'location': 'statistical_learning.py',
                'category': 'Statistical Learning'
            },
            'StatisticalFeatureSelector': {
                'description': 'Statistical feature selection methods',
                'features': [
                    'Mutual information selection',
                    'Chi-square tests',
                    'F-tests',
                    'Statistical significance'
                ],
                'location': 'statistical_learning.py',
                'category': 'Statistical Learning'
            },
            'AndrewNgMLStrategy': {
                'description': 'Complete Andrew Ng ML Technical Strategy',
                'features': [
                    'Systematic error analysis',
                    'Bias/variance diagnosis',
                    'Learning curves analysis',
                    'Model debugging framework',
                    'Systematic model selection',
                    'Prioritized recommendations'
                ],
                'location': 'andrew_ng_ml_strategy.py',
                'category': 'ML Strategy'
            },
            'ErrorAnalyzer': {
                'description': 'Systematic error analysis',
                'features': [
                    'Confusion matrix analysis',
                    'Error patterns identification',
                    'Feature importance in errors',
                    'Actionable recommendations'
                ],
                'location': 'andrew_ng_ml_strategy.py',
                'category': 'ML Strategy'
            },
            'BiasVarianceDiagnostic': {
                'description': 'Bias/variance diagnosis',
                'features': [
                    'Underfitting detection',
                    'Overfitting detection',
                    'Diagnosis and recommendations',
                    'Performance gap analysis'
                ],
                'location': 'andrew_ng_ml_strategy.py',
                'category': 'ML Strategy'
            },
            'LearningCurvesAnalyzer': {
                'description': 'Learning curves analysis',
                'features': [
                    'Performance vs training size',
                    'Data collection recommendations',
                    'Overfitting detection',
                    'Training efficiency analysis'
                ],
                'location': 'andrew_ng_ml_strategy.py',
                'category': 'ML Strategy'
            },
            'ModelDebugger': {
                'description': 'Systematic model debugging',
                'features': [
                    'Data quality checks',
                    'Feature analysis',
                    'Model performance issues',
                    'Training process checks'
                ],
                'location': 'andrew_ng_ml_strategy.py',
                'category': 'ML Strategy'
            },
            'SystematicModelSelector': {
                'description': 'Systematic model selection',
                'features': [
                    'Multi-model comparison',
                    'Cross-validation evaluation',
                    'Best model identification',
                    'Model complexity analysis'
                ],
                'location': 'andrew_ng_ml_strategy.py',
                'category': 'ML Strategy'
            },
            'AdvancedResampler': {
                'description': 'Kuhn/Johnson advanced resampling methods',
                'features': [
                    'Repeated K-Fold CV (reduce variance)',
                    'Bootstrap resampling with CI',
                    'Leave-One-Out CV',
                    'Time series CV',
                    'Group K-Fold CV',
                    'Nested CV'
                ],
                'location': 'kuhn_johnson_resampling.py',
                'category': 'Resampling'
            },
            'VariableImportanceAnalyzer': {
                'description': 'Kuhn/Johnson variable importance analysis',
                'features': [
                    'Permutation importance (model-agnostic)',
                    'Built-in importance (model-specific)',
                    'SHAP values (if available)',
                    'Stability analysis across CV folds',
                    'Multiple importance methods',
                    'Combined rankings'
                ],
                'location': 'variable_importance.py',
                'category': 'Analysis'
            },
            'PerformanceProfile': {
                'description': 'Kuhn/Johnson performance profiles',
                'features': [
                    'Visual model comparison',
                    'Boxplots of CV scores',
                    'Statistical significance testing',
                    'Model rankings',
                    'Publication-quality plots'
                ],
                'location': 'performance_profiles.py',
                'category': 'Evaluation'
            },
            'AdvancedFeatureSelector': {
                'description': 'Kuhn/Johnson advanced feature selection',
                'features': [
                    'Forward selection (wrapper)',
                    'Backward elimination (wrapper)',
                    'Recursive Feature Elimination (RFE)',
                    'Stability selection',
                    'Embedded methods (L1 regularization)',
                    'CV-aware selection'
                ],
                'location': 'advanced_feature_selection.py',
                'category': 'Feature Selection'
            },
            'ModelCalibrator': {
                'description': 'Kuhn/Johnson model calibration',
                'features': [
                    'Platt scaling (logistic regression)',
                    'Isotonic regression',
                    'Calibration plots',
                    'Brier score evaluation',
                    'Probability calibration',
                    'Expected calibration error'
                ],
                'location': 'model_calibration.py',
                'category': 'Calibration'
            },
            'SearchBasedFeatureSelector': {
                'description': 'Russell/Norvig search-based feature selection',
                'features': [
                    'A* search for optimal feature subsets',
                    'Beam search for feature selection',
                    'Greedy best-first search',
                    'Better than wrapper methods in some cases',
                    'Finds globally optimal feature sets'
                ],
                'location': 'search_based_feature_selection.py',
                'category': 'Feature Selection'
            },
            'SimulatedAnnealingOptimizer': {
                'description': 'Russell/Norvig simulated annealing for hyperparameter optimization',
                'features': [
                    'Escapes local optima',
                    'Temperature schedule (exponential, linear)',
                    'Acceptance probability based on temperature',
                    'Better exploration than grid/random search',
                    'Alternative optimization strategy'
                ],
                'location': 'simulated_annealing_optimizer.py',
                'category': 'Optimization'
            },
            'BayesianNetworkAnalyzer': {
                'description': 'Russell/Norvig Bayesian Networks for feature relationships',
                'features': [
                    'Model dependencies between features',
                    'Causal relationship analysis',
                    'Structure learning',
                    'Inference (variable elimination)',
                    'Feature importance based on network structure'
                ],
                'location': 'bayesian_networks.py',
                'category': 'Probabilistic Models'
            },
            'CSPOptimizer': {
                'description': 'Russell/Norvig constraint satisfaction for optimization',
                'features': [
                    'Handle complex constraints in hyperparameter spaces',
                    'Backtracking search',
                    'Constraint validation',
                    'Example: n_estimators > max_depth',
                    'Constraint-based feature engineering'
                ],
                'location': 'constraint_satisfaction_optimizer.py',
                'category': 'Optimization'
            },
            'HMMAnalyzer': {
                'description': 'Russell/Norvig Hidden Markov Models for sequential data',
                'features': [
                    'Model sequential patterns',
                    'Temporal dependencies',
                    'Baum-Welch algorithm (EM)',
                    'Viterbi algorithm for decoding',
                    'Forward-backward algorithm for inference'
                ],
                'location': 'hidden_markov_models.py',
                'category': 'Probabilistic Models'
            },
            'GeneticAlgorithmOptimizer': {
                'description': 'Russell/Norvig genetic algorithms for model selection',
                'features': [
                    'Evolutionary algorithm for hyperparameter search',
                    'Population-based optimization',
                    'Crossover and mutation operators',
                    'Selection strategies (tournament, roulette)',
                    'Elitism'
                ],
                'location': 'genetic_algorithm_optimizer.py',
                'category': 'Optimization'
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
    
    def get_statistical_evaluator(self, n_bootstrap: int = 1000):
        """
        Get statistical evaluator instance
        
        Args:
            n_bootstrap: Number of bootstrap samples for uncertainty quantification
            
        Returns:
            StatisticalEvaluator instance
        """
        if 'StatisticalEvaluator' in self.components:
            return self.components['StatisticalEvaluator'](n_bootstrap=n_bootstrap)
        else:
            raise ImportError("StatisticalEvaluator not available. Install scipy: pip install scipy")
    
    def get_statistical_validator(self):
        """
        Get statistical validator instance
        
        Returns:
            StatisticalValidator instance
        """
        if 'StatisticalValidator' in self.components:
            return self.components['StatisticalValidator']()
        else:
            raise ImportError("StatisticalValidator not available. Install scipy: pip install scipy")
    
    def get_bayesian_optimizer(self, n_calls: int = 50):
        """
        Get Bayesian optimizer instance
        
        Args:
            n_calls: Number of optimization iterations
            
        Returns:
            BayesianOptimizer instance
        """
        if 'BayesianOptimizer' in self.components:
            return self.components['BayesianOptimizer'](n_calls=n_calls)
        else:
            raise ImportError("BayesianOptimizer not available. Install scikit-optimize: pip install scikit-optimize")
    
    def get_statistical_feature_selector(self):
        """
        Get statistical feature selector instance
        
        Returns:
            StatisticalFeatureSelector instance
        """
        if 'StatisticalFeatureSelector' in self.components:
            return self.components['StatisticalFeatureSelector']()
        else:
            raise ImportError("StatisticalFeatureSelector not available")
    
    def get_andrew_ng_strategy(self):
        """
        Get Andrew Ng ML Strategy instance
        
        Returns:
            AndrewNgMLStrategy instance
        """
        if 'AndrewNgMLStrategy' in self.components:
            return self.components['AndrewNgMLStrategy']()
        else:
            raise ImportError("AndrewNgMLStrategy not available. Ensure andrew_ng_ml_strategy.py is available")
    
    def get_error_analyzer(self):
        """Get error analyzer instance"""
        if 'ErrorAnalyzer' in self.components:
            return self.components['ErrorAnalyzer']()
        else:
            raise ImportError("ErrorAnalyzer not available")
    
    def get_bias_variance_diagnostic(self):
        """Get bias/variance diagnostic instance"""
        if 'BiasVarianceDiagnostic' in self.components:
            return self.components['BiasVarianceDiagnostic']()
        else:
            raise ImportError("BiasVarianceDiagnostic not available")
    
    def get_learning_curves_analyzer(self):
        """Get learning curves analyzer instance"""
        if 'LearningCurvesAnalyzer' in self.components:
            return self.components['LearningCurvesAnalyzer']()
        else:
            raise ImportError("LearningCurvesAnalyzer not available")
    
    def get_model_debugger(self):
        """Get model debugger instance"""
        if 'ModelDebugger' in self.components:
            return self.components['ModelDebugger']()
        else:
            raise ImportError("ModelDebugger not available")
    
    def get_systematic_model_selector(self):
        """Get systematic model selector instance"""
        if 'SystematicModelSelector' in self.components:
            return self.components['SystematicModelSelector']()
        else:
            raise ImportError("SystematicModelSelector not available")
    
    def get_advanced_resampler(self):
        """Get advanced resampler instance (Kuhn/Johnson)"""
        if 'AdvancedResampler' in self.components:
            return self.components['AdvancedResampler']()
        else:
            raise ImportError("AdvancedResampler not available")
    
    def get_variable_importance_analyzer(self):
        """Get variable importance analyzer instance (Kuhn/Johnson)"""
        if 'VariableImportanceAnalyzer' in self.components:
            return self.components['VariableImportanceAnalyzer']()
        else:
            raise ImportError("VariableImportanceAnalyzer not available")
    
    def get_performance_profile(self):
        """Get performance profile instance (Kuhn/Johnson)"""
        if 'PerformanceProfile' in self.components:
            return self.components['PerformanceProfile']()
        else:
            raise ImportError("PerformanceProfile not available")
    
    def get_advanced_feature_selector(self, method: str = 'rfe', **kwargs):
        """Get advanced feature selector instance (Kuhn/Johnson)"""
        if 'AdvancedFeatureSelector' in self.components:
            return self.components['AdvancedFeatureSelector'](method=method, **kwargs)
        else:
            raise ImportError("AdvancedFeatureSelector not available")
    
    def get_model_calibrator(self, method: str = 'isotonic', cv: int = 5):
        """Get model calibrator instance (Kuhn/Johnson)"""
        if 'ModelCalibrator' in self.components:
            return self.components['ModelCalibrator'](method=method, cv=cv)
        else:
            raise ImportError("ModelCalibrator not available")
    
    def get_search_based_feature_selector(self, estimator: Any, method: str = 'astar', **kwargs):
        """Get search-based feature selector instance (Russell/Norvig)"""
        if 'SearchBasedFeatureSelector' in self.components:
            return self.components['SearchBasedFeatureSelector'](estimator=estimator, method=method, **kwargs)
        else:
            raise ImportError("SearchBasedFeatureSelector not available")
    
    def get_simulated_annealing_optimizer(self, **kwargs):
        """Get simulated annealing optimizer instance (Russell/Norvig)"""
        if 'SimulatedAnnealingOptimizer' in self.components:
            return self.components['SimulatedAnnealingOptimizer'](**kwargs)
        else:
            raise ImportError("SimulatedAnnealingOptimizer not available")
    
    def get_bayesian_network_analyzer(self, **kwargs):
        """Get Bayesian network analyzer instance (Russell/Norvig)"""
        if 'BayesianNetworkAnalyzer' in self.components:
            return self.components['BayesianNetworkAnalyzer'](**kwargs)
        else:
            raise ImportError("BayesianNetworkAnalyzer not available")
    
    def get_csp_optimizer(self, constraints: Optional[List] = None):
        """Get CSP optimizer instance (Russell/Norvig)"""
        if 'CSPOptimizer' in self.components:
            return self.components['CSPOptimizer'](constraints=constraints)
        else:
            raise ImportError("CSPOptimizer not available")
    
    def get_hmm_analyzer(self, n_states: int = 3, **kwargs):
        """Get HMM analyzer instance (Russell/Norvig)"""
        if 'HMMAnalyzer' in self.components:
            return self.components['HMMAnalyzer'](n_states=n_states, **kwargs)
        else:
            raise ImportError("HMMAnalyzer not available")
    
    def get_genetic_algorithm_optimizer(self, **kwargs):
        """Get genetic algorithm optimizer instance (Russell/Norvig)"""
        if 'GeneticAlgorithmOptimizer' in self.components:
            return self.components['GeneticAlgorithmOptimizer'](**kwargs)
        else:
            raise ImportError("GeneticAlgorithmOptimizer not available")
    
    def list_components(self):
        """List all available components in this compartment"""
        print("="*80)
        print("COMPARTMENT 3: ALGORITHMS")
        print("="*80)
        print("\nComponents:")
        for name, component in self.components.items():
            desc = self.component_descriptions.get(name, {})
            print(f"\n{name}:")
            print(f"  Description: {desc.get('description', 'N/A')}")
            print(f"  Location: {desc.get('location', 'N/A')}")
            print(f"  Category: {desc.get('category', 'N/A')}")
            if 'features' in desc:
                print(f"  Features:")
                for feature in desc['features']:
                    print(f"    - {feature}")
        print("\n" + "="*80)
    
    def get_info(self):
        """Get information about this compartment"""
        return {
            'name': 'Algorithms Compartment',
            'description': 'ML models, evaluation, tuning, and ensembles',
            'components': list(self.components.keys()),
            'component_count': len(self.components)
        }
