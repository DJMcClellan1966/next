"""
Super Power Agent - Main Orchestrator for Super Power Tool

Provides natural language interface, multi-agent coordination, and
end-to-end ML workflow automation.
"""
import numpy as np
from typing import Any, Dict, Optional, Union, List, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of ML tasks"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    DEPLOYMENT = "deployment"
    ANALYSIS = "analysis"
    UNKNOWN = "unknown"


@dataclass
class UserIntent:
    """Parsed user intent"""
    task_type: TaskType
    goal: str
    data_info: Optional[Dict] = None
    requirements: List[str] = None
    constraints: List[str] = None


class SuperPowerAgent:
    """
    Super Power Agent - Main orchestrator
    
    Provides:
    - Natural language understanding
    - Multi-agent coordination
    - End-to-end workflow automation
    - Learning and improvement
    """
    
    def __init__(self, toolbox=None):
        """
        Initialize Super Power Agent
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
        """
        self.toolbox = toolbox
        self.specialist_agents = {}
        self.conversation_history = []
        self.learned_patterns = {}
        self.user_preferences = {}
        
        # Initialize specialist agents
        self._init_specialist_agents()
        
        logger.info("[SuperPowerAgent] Initialized")
    
    def _init_specialist_agents(self):
        """Initialize specialist agents"""
        from .specialist_agents import (
            DataAgent, FeatureAgent, ModelAgent,
            TuningAgent, DeployAgent, InsightAgent
        )
        
        self.specialist_agents = {
            'data': DataAgent(toolbox=self.toolbox),
            'feature': FeatureAgent(toolbox=self.toolbox),
            'model': ModelAgent(toolbox=self.toolbox),
            'tuning': TuningAgent(toolbox=self.toolbox),
            'deploy': DeployAgent(toolbox=self.toolbox),
            'insight': InsightAgent(toolbox=self.toolbox)
        }
    
    def understand_intent(self, user_input: str, context: Optional[Dict] = None) -> UserIntent:
        """
        Understand user intent from natural language
        
        Parameters
        ----------
        user_input : str
            User's natural language input
        context : dict, optional
            Conversation context
            
        Returns
        -------
        intent : UserIntent
            Parsed user intent
        """
        user_input_lower = user_input.lower()
        
        # Simple intent detection (can be enhanced with NLP)
        task_type = TaskType.UNKNOWN
        goal = user_input
        
        # Detect task type
        if any(word in user_input_lower for word in ['classify', 'classification', 'predict category']):
            task_type = TaskType.CLASSIFICATION
        elif any(word in user_input_lower for word in ['predict', 'regression', 'forecast', 'estimate']):
            task_type = TaskType.REGRESSION
        elif any(word in user_input_lower for word in ['cluster', 'group', 'segment']):
            task_type = TaskType.CLUSTERING
        elif any(word in user_input_lower for word in ['feature', 'engineer', 'transform']):
            task_type = TaskType.FEATURE_ENGINEERING
        elif any(word in user_input_lower for word in ['train', 'model', 'learn']):
            task_type = TaskType.MODEL_TRAINING
        elif any(word in user_input_lower for word in ['tune', 'optimize', 'hyperparameter']):
            task_type = TaskType.HYPERPARAMETER_TUNING
        elif any(word in user_input_lower for word in ['deploy', 'serve', 'production']):
            task_type = TaskType.DEPLOYMENT
        elif any(word in user_input_lower for word in ['analyze', 'explore', 'understand']):
            task_type = TaskType.ANALYSIS
        
        # Extract requirements
        requirements = []
        if 'best' in user_input_lower or 'optimal' in user_input_lower:
            requirements.append('best_performance')
        if 'fast' in user_input_lower or 'quick' in user_input_lower:
            requirements.append('speed')
        if 'explain' in user_input_lower or 'understand' in user_input_lower:
            requirements.append('explainability')
        
        intent = UserIntent(
            task_type=task_type,
            goal=goal,
            requirements=requirements,
            constraints=[]
        )
        
        return intent
    
    def execute_task(self, intent: UserIntent, data: Optional[np.ndarray] = None, 
                    target: Optional[np.ndarray] = None, **kwargs) -> Dict:
        """
        Execute ML task based on user intent
        
        Parameters
        ----------
        intent : UserIntent
            User intent
        data : array-like, optional
            Input data
        target : array-like, optional
            Target labels
        **kwargs
            Additional parameters
            
        Returns
        -------
        result : dict
            Execution results
        """
        logger.info(f"[SuperPowerAgent] Executing task: {intent.task_type.value}")
        
        if not self.toolbox:
            raise ValueError("ML Toolbox not available")
        
        # Route to appropriate handler
        if intent.task_type == TaskType.CLASSIFICATION:
            return self._handle_classification(intent, data, target, **kwargs)
        elif intent.task_type == TaskType.REGRESSION:
            return self._handle_regression(intent, data, target, **kwargs)
        elif intent.task_type == TaskType.CLUSTERING:
            return self._handle_clustering(intent, data, **kwargs)
        elif intent.task_type == TaskType.FEATURE_ENGINEERING:
            return self._handle_feature_engineering(intent, data, target, **kwargs)
        elif intent.task_type == TaskType.MODEL_TRAINING:
            return self._handle_training(intent, data, target, **kwargs)
        elif intent.task_type == TaskType.HYPERPARAMETER_TUNING:
            return self._handle_tuning(intent, data, target, **kwargs)
        elif intent.task_type == TaskType.DEPLOYMENT:
            return self._handle_deployment(intent, data, **kwargs)
        elif intent.task_type == TaskType.ANALYSIS:
            return self._handle_analysis(intent, data, target, **kwargs)
        else:
            return {'error': 'Unknown task type', 'intent': intent}
    
    def _handle_classification(self, intent: UserIntent, data: np.ndarray, 
                              target: np.ndarray, **kwargs) -> Dict:
        """Handle classification task"""
        # Use optimization kernels
        if self.toolbox.feature_kernel:
            data = self.toolbox.feature_kernel.auto_engineer(data, target)
        
        if self.toolbox.algorithm_kernel:
            algo_kernel = self.toolbox.algorithm_kernel
            algo_kernel.fit(data, target, algorithm='auto')
            predictions = algo_kernel.predict(data)
        else:
            result = self.toolbox.fit(data, target, task_type='classification')
            if isinstance(result, dict) and 'model' in result:
                predictions = self.toolbox.predict(result['model'], data)
            else:
                predictions = None
        
        # Evaluate
        if self.toolbox.eval_kernel and predictions is not None:
            metrics = self.toolbox.eval_kernel.evaluate(target, predictions)
        else:
            metrics = {}
        
        return {
            'task': 'classification',
            'predictions': predictions,
            'metrics': metrics,
            'status': 'success'
        }
    
    def _handle_regression(self, intent: UserIntent, data: np.ndarray, 
                          target: np.ndarray, **kwargs) -> Dict:
        """Handle regression task"""
        # Similar to classification but for regression
        if self.toolbox.feature_kernel:
            data = self.toolbox.feature_kernel.auto_engineer(data, target)
        
        result = self.toolbox.fit(data, target, task_type='regression')
        
        if isinstance(result, dict) and 'model' in result:
            predictions = self.toolbox.predict(result['model'], data)
            if self.toolbox.eval_kernel:
                metrics = self.toolbox.eval_kernel.evaluate(target, predictions, 
                                                           metrics=['r2', 'mse', 'mae'])
            else:
                metrics = {}
        else:
            predictions = None
            metrics = {}
        
        return {
            'task': 'regression',
            'predictions': predictions,
            'metrics': metrics,
            'status': 'success'
        }
    
    def _handle_clustering(self, intent: UserIntent, data: np.ndarray, **kwargs) -> Dict:
        """Handle clustering task"""
        # Use pipeline kernel for preprocessing
        if self.toolbox.pipeline_kernel:
            data = self.toolbox.pipeline_kernel.execute(data, steps=['preprocess'])
        
        # Clustering (simplified)
        result = self.toolbox.fit(data, None, task_type='clustering')
        
        return {
            'task': 'clustering',
            'clusters': result,
            'status': 'success'
        }
    
    def _handle_feature_engineering(self, intent: UserIntent, data: np.ndarray, 
                                   target: Optional[np.ndarray], **kwargs) -> Dict:
        """Handle feature engineering task"""
        if self.toolbox.feature_kernel:
            engineered = self.toolbox.feature_kernel.auto_engineer(data, target)
            return {
                'task': 'feature_engineering',
                'engineered_data': engineered,
                'original_shape': data.shape,
                'engineered_shape': engineered.shape,
                'status': 'success'
            }
        return {'error': 'Feature kernel not available'}
    
    def _handle_training(self, intent: UserIntent, data: np.ndarray, 
                        target: np.ndarray, **kwargs) -> Dict:
        """Handle model training task"""
        # Use ensemble kernel if best performance requested
        if 'best_performance' in intent.requirements and self.toolbox.ensemble_kernel:
            ens_kernel = self.toolbox.ensemble_kernel
            ens_kernel.create_ensemble(data, target, models=['rf', 'svm', 'lr'], method='voting')
            predictions = ens_kernel.predict(data)
            
            if self.toolbox.eval_kernel:
                metrics = self.toolbox.eval_kernel.evaluate(target, predictions)
            else:
                metrics = {}
            
            return {
                'task': 'training',
                'model_type': 'ensemble',
                'predictions': predictions,
                'metrics': metrics,
                'status': 'success'
            }
        else:
            # Standard training
            return self._handle_classification(intent, data, target, **kwargs)
    
    def _handle_tuning(self, intent: UserIntent, data: np.ndarray, 
                      target: np.ndarray, **kwargs) -> Dict:
        """Handle hyperparameter tuning task"""
        if self.toolbox.tuning_kernel:
            tune_kernel = self.toolbox.tuning_kernel
            search_space = kwargs.get('search_space', {
                'n_estimators': [10, 50, 100],
                'max_depth': [3, 5, 10]
            })
            best_params = tune_kernel.tune('rf', data, target, search_space)
            return {
                'task': 'tuning',
                'best_params': best_params,
                'status': 'success'
            }
        return {'error': 'Tuning kernel not available'}
    
    def _handle_deployment(self, intent: UserIntent, data: Optional[np.ndarray], **kwargs) -> Dict:
        """Handle deployment task"""
        model = kwargs.get('model')
        if model and self.toolbox.serving_kernel:
            serve_kernel = self.toolbox.serving_kernel
            # Deployment preparation
            return {
                'task': 'deployment',
                'status': 'ready',
                'serving_kernel': 'available'
            }
        return {'error': 'Model or serving kernel not available'}
    
    def _handle_analysis(self, intent: UserIntent, data: np.ndarray, 
                        target: Optional[np.ndarray], **kwargs) -> Dict:
        """Handle data analysis task"""
        try:
            # Use DataAgent if available
            if 'data' in self.specialist_agents:
                data_agent = self.specialist_agents['data']
                analysis = data_agent.analyze(data)
                analysis['task'] = 'analysis'
                return analysis
            
            # Fallback to basic analysis
            analysis = {
                'task': 'analysis',
                'data_shape': data.shape,
                'data_info': {
                    'mean': np.mean(data, axis=0).tolist() if len(data.shape) > 1 else [float(np.mean(data))],
                    'std': np.std(data, axis=0).tolist() if len(data.shape) > 1 else [float(np.std(data))],
                    'min': np.min(data, axis=0).tolist() if len(data.shape) > 1 else [float(np.min(data))],
                    'max': np.max(data, axis=0).tolist() if len(data.shape) > 1 else [float(np.max(data))],
                }
            }
            
            if target is not None:
                target = np.asarray(target)
                unique_vals = np.unique(target)
                if len(unique_vals) < 20:
                    try:
                        distribution = np.bincount(target.astype(int)).tolist()
                    except:
                        distribution = f"{len(unique_vals)} unique values"
                else:
                    distribution = 'continuous'
                
                analysis['target_info'] = {
                    'unique_values': len(unique_vals),
                    'distribution': distribution
                }
            
            return analysis
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                'task': 'analysis',
                'error': str(e),
                'data_shape': data.shape if hasattr(data, 'shape') else 'unknown'
            }
    
    def chat(self, user_input: str, data: Optional[np.ndarray] = None, 
             target: Optional[np.ndarray] = None, **kwargs) -> Dict:
        """
        Conversational interface
        
        Parameters
        ----------
        user_input : str
            User's natural language input
        data : array-like, optional
            Input data
        target : array-like, optional
            Target labels
        **kwargs
            Additional parameters
            
        Returns
        -------
        response : dict
            Agent response with results
        """
        # Store conversation
        self.conversation_history.append({'user': user_input, 'timestamp': None})
        
        # Understand intent
        intent = self.understand_intent(user_input)
        
        # Execute task
        if intent.task_type != TaskType.UNKNOWN:
            result = self.execute_task(intent, data, target, **kwargs)
            
            # Generate response
            response = self._generate_response(intent, result)
        else:
            response = {
                'message': "I'm not sure what you'd like me to do. Could you clarify?",
                'suggestions': [
                    "Try: 'Predict house prices from this data'",
                    "Try: 'Classify these images'",
                    "Try: 'Train a model to predict sales'"
                ]
            }
        
        # Store response
        self.conversation_history[-1]['agent'] = response
        
        return response
    
    def _generate_response(self, intent: UserIntent, result: Dict) -> Dict:
        """Generate natural language response"""
        if result.get('status') == 'success':
            if 'metrics' in result:
                metrics = result['metrics']
                if 'accuracy' in metrics:
                    message = f"Task completed! Accuracy: {metrics['accuracy']:.2%}"
                elif 'r2' in metrics:
                    message = f"Task completed! R² score: {metrics['r2']:.4f}"
                else:
                    message = "Task completed successfully!"
            else:
                message = "Task completed successfully!"
            
            return {
                'message': message,
                'result': result,
                'suggestions': self._generate_suggestions(intent, result)
            }
        else:
            return {
                'message': f"Task failed: {result.get('error', 'Unknown error')}",
                'result': result
            }
    
    def _generate_suggestions(self, intent: UserIntent, result: Dict) -> List[str]:
        """Generate helpful suggestions"""
        suggestions = []
        
        if intent.task_type in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
            if 'metrics' in result:
                metrics = result['metrics']
                if 'accuracy' in metrics and metrics['accuracy'] < 0.8:
                    suggestions.append("Accuracy is below 80%. Consider feature engineering or ensemble methods.")
                elif 'r2' in metrics and metrics['r2'] < 0.7:
                    suggestions.append("R² score is below 0.7. Consider hyperparameter tuning.")
            
            suggestions.append("Would you like to tune hyperparameters for better performance?")
            suggestions.append("Would you like to deploy this model?")
        
        return suggestions
    
    def learn_from_interaction(self, intent: UserIntent, result: Dict, user_feedback: Optional[str] = None):
        """Learn from user interactions"""
        # Store successful patterns
        if result.get('status') == 'success':
            pattern_key = f"{intent.task_type.value}_{intent.goal[:50]}"
            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = {
                    'count': 0,
                    'success_rate': 0.0,
                    'avg_metrics': {}
                }
            
            self.learned_patterns[pattern_key]['count'] += 1
            if 'metrics' in result:
                # Update average metrics
                for metric, value in result['metrics'].items():
                    if metric not in self.learned_patterns[pattern_key]['avg_metrics']:
                        self.learned_patterns[pattern_key]['avg_metrics'][metric] = value
                    else:
                        # Running average
                        current = self.learned_patterns[pattern_key]['avg_metrics'][metric]
                        count = self.learned_patterns[pattern_key]['count']
                        self.learned_patterns[pattern_key]['avg_metrics'][metric] = (current * (count - 1) + value) / count
