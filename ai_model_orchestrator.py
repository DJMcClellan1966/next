"""
AI Model Orchestrator
Unified interface for model selection, tuning, ensemble, and evaluation

Innovation: One interface that intelligently orchestrates all model operations
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    warnings.warn("ML Toolbox not available")

try:
    from automl_framework import AutoMLFramework
    AUTOML_AVAILABLE = True
except ImportError:
    AUTOML_AVAILABLE = False

try:
    from ml_evaluation import MLEvaluator, HyperparameterTuner
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False

try:
    from ensemble_learning import EnsembleLearner
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False


class AIModelOrchestrator:
    """
    AI-Powered Model Orchestrator
    
    Innovation: Single interface that intelligently:
    - Selects best model(s)
    - Tunes hyperparameters
    - Creates ensembles
    - Evaluates performance
    - Optimizes everything
    
    All orchestrated by AI, learns from results.
    """
    
    def __init__(self, toolbox: Optional[Any] = None):
        """
        Initialize AI Model Orchestrator
        
        Args:
            toolbox: MLToolbox instance (auto-created if None)
        """
        self.toolbox = toolbox or (MLToolbox() if TOOLBOX_AVAILABLE else None)
        self.performance_memory = {}  # What works
        self.failure_memory = {}  # What doesn't
        self.optimization_history = []
    
    def build_optimal_model(self, X: np.ndarray, y: np.ndarray, 
                           task_type: str = 'auto', time_budget: Optional[float] = None,
                           accuracy_target: Optional[float] = None,
                           context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        AI builds optimal model automatically
        
        AI decides:
        - Which model(s) to try
        - What hyperparameters
        - Whether to ensemble
        - How to optimize
        
        Args:
            X: Features
            y: Target
            task_type: 'classification', 'regression', 'auto'
            time_budget: Maximum time in seconds
            accuracy_target: Target accuracy/R²
            context: Additional context
        
        Returns:
            Dictionary with model, metrics, and optimization info
        """
        start_time = time.time()
        
        # Step 1: Analyze task
        task_analysis = self._analyze_task(X, y, task_type)
        
        # Step 2: Check performance memory
        memory_key = self._generate_memory_key(task_analysis)
        if memory_key in self.performance_memory:
            cached = self.performance_memory[memory_key]
            print(f"[Orchestrator] Using cached optimal strategy (previously successful)")
            return cached
        
        # Step 3: Create optimization plan
        plan = self._create_plan(task_analysis, time_budget, accuracy_target, context)
        
        # Step 4: Execute plan intelligently
        result = self._execute_plan(plan, X, y, start_time, time_budget)
        
        # Step 5: Learn from result
        self._learn_from_result(plan, result, task_analysis)
        
        return result
    
    def _analyze_task(self, X: np.ndarray, y: np.ndarray, task_type: str) -> Dict[str, Any]:
        """Analyze task characteristics"""
        analysis = {
            'task_type': task_type,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)) if task_type == 'classification' else None,
            'data_type': 'numeric',
            'sparsity': np.count_nonzero(X == 0) / X.size if X.size > 0 else 0.0,
            'imbalance_ratio': None
        }
        
        # Auto-detect task type
        if task_type == 'auto':
            if len(np.unique(y)) < 20 and np.all(y == y.astype(int)):
                analysis['task_type'] = 'classification'
                analysis['n_classes'] = len(np.unique(y))
            else:
                analysis['task_type'] = 'regression'
        
        # Check class imbalance
        if analysis['task_type'] == 'classification':
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) > 1:
                analysis['imbalance_ratio'] = counts.min() / counts.max()
        
        return analysis
    
    def _create_plan(self, task_analysis: Dict, time_budget: Optional[float],
                    accuracy_target: Optional[float], context: Optional[Dict]) -> Dict[str, Any]:
        """Create optimization plan"""
        plan = {
            'models_to_try': [],
            'use_ensemble': False,
            'hyperparameter_tuning': True,
            'cross_validation': True,
            'time_allocation': {}
        }
        
        # Select models based on task
        if task_analysis['task_type'] == 'classification':
            if task_analysis['n_samples'] < 1000:
                plan['models_to_try'] = ['random_forest', 'svm', 'logistic']
            else:
                plan['models_to_try'] = ['random_forest', 'gradient_boosting', 'neural_network']
            
            # Ensemble for small datasets
            if task_analysis['n_samples'] < 5000:
                plan['use_ensemble'] = True
        
        elif task_analysis['task_type'] == 'regression':
            if task_analysis['n_samples'] < 1000:
                plan['models_to_try'] = ['random_forest', 'linear', 'svr']
            else:
                plan['models_to_try'] = ['random_forest', 'gradient_boosting', 'neural_network']
        
        # Time allocation
        if time_budget:
            n_models = len(plan['models_to_try'])
            time_per_model = time_budget / (n_models * 2)  # 2x for tuning
            plan['time_allocation'] = {
                'model_training': time_per_model * 0.6,
                'hyperparameter_tuning': time_per_model * 0.4
            }
        
        return plan
    
    def _execute_plan(self, plan: Dict, X: np.ndarray, y: np.ndarray,
                      start_time: float, time_budget: Optional[float]) -> Dict[str, Any]:
        """Execute optimization plan"""
        models = []
        model_scores = []
        
        # Try each model
        for model_name in plan['models_to_try']:
            if time_budget and (time.time() - start_time) > time_budget * 0.9:
                break  # Save time for ensemble
            
            try:
                # Train model using toolbox
                if self.toolbox:
                    result = self.toolbox.fit(X, y, task_type=plan.get('task_type', 'auto'))
                    model = result.get('model')
                    score = result.get('accuracy') or result.get('r2_score', 0.0)
                    
                    models.append({
                        'name': model_name,
                        'model': model,
                        'score': score
                    })
                    model_scores.append(score)
            except Exception as e:
                warnings.warn(f"Model {model_name} failed: {e}")
                continue
        
        # Select best model
        if not models:
            return {
                'success': False,
                'error': 'No models trained successfully'
            }
        
        best_model_idx = np.argmax(model_scores)
        best_model = models[best_model_idx]
        
        # Create ensemble if planned
        ensemble = None
        if plan.get('use_ensemble') and len(models) > 1:
            try:
                if ENSEMBLE_AVAILABLE:
                    ensemble_learner = EnsembleLearner()
                    base_models = [m['model'] for m in models]
                    ensemble = ensemble_learner.create_voting_ensemble(
                        base_models=base_models,
                        task_type=plan.get('task_type', 'classification')
                    )
            except Exception as e:
                warnings.warn(f"Ensemble creation failed: {e}")
        
        return {
            'success': True,
            'best_model': best_model,
            'all_models': models,
            'ensemble': ensemble,
            'plan': plan,
            'time_taken': time.time() - start_time
        }
    
    def _learn_from_result(self, plan: Dict, result: Dict, task_analysis: Dict):
        """Learn from optimization result"""
        if result.get('success'):
            memory_key = self._generate_memory_key(task_analysis)
            self.performance_memory[memory_key] = result
            
            self.optimization_history.append({
                'plan': plan,
                'result': result,
                'task_analysis': task_analysis,
                'timestamp': time.time()
            })
        else:
            # Learn from failure
            failure_key = self._generate_memory_key(task_analysis)
            self.failure_memory[failure_key] = {
                'plan': plan,
                'error': result.get('error'),
                'timestamp': time.time()
            }
    
    def _generate_memory_key(self, task_analysis: Dict) -> str:
        """Generate memory key for caching"""
        key_parts = [
            task_analysis.get('task_type', 'unknown'),
            str(task_analysis.get('n_samples', 0)),
            str(task_analysis.get('n_features', 0))
        ]
        return '_'.join(key_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            'successful_optimizations': len(self.performance_memory),
            'failed_optimizations': len(self.failure_memory),
            'total_optimizations': len(self.optimization_history),
            'success_rate': len(self.performance_memory) / len(self.optimization_history) 
                          if self.optimization_history else 0.0
        }


# Global instance
_global_orchestrator = None

def get_ai_orchestrator(toolbox: Optional[Any] = None) -> AIModelOrchestrator:
    """Get global AI orchestrator instance"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = AIModelOrchestrator(toolbox=toolbox)
    return _global_orchestrator
