"""
Training Pipeline

Stages:
1. Model Training
2. Hyperparameter Tuning
3. Model Evaluation
4. Model Validation
5. Model Registry
"""
import numpy as np
from typing import Any, Dict, Optional, List
import logging

from .base import BasePipeline, PipelineStage

logger = logging.getLogger(__name__)


class ModelTrainingStage(PipelineStage):
    """Stage 1: Model Training"""
    
    def __init__(self, toolbox=None, model_type: str = 'auto'):
        super().__init__("model_training")
        self.toolbox = toolbox
        self.model_type = model_type
    
    def execute(self, input_data: tuple, **kwargs) -> Dict[str, Any]:
        """Train model"""
        X, y = input_data
        model_type = kwargs.get('model_type', self.model_type)
        
        if self.toolbox:
            # Use toolbox fit
            result = self.toolbox.fit(X, y, model_type=model_type, **kwargs)
            
            if isinstance(result, dict):
                model = result.get('model', result)
                metrics = {k: v for k, v in result.items() if k != 'model'}
            else:
                model = result
                metrics = {}
        else:
            # Fallback
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import LogisticRegression, LinearRegression
            
            # Simple auto-detection
            if len(np.unique(y)) < 20:
                if model_type == 'auto' or model_type == 'rf':
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = LogisticRegression(random_state=42)
            else:
                if model_type == 'auto' or model_type == 'rf':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = LinearRegression()
            
            model.fit(X, y)
            metrics = {}
        
        self.metadata['model_type'] = str(type(model).__name__)
        self.metadata['metrics'] = metrics
        
        return {
            'model': model,
            'metrics': metrics,
            'X': X,
            'y': y
        }


class HyperparameterTuningStage(PipelineStage):
    """Stage 2: Hyperparameter Tuning"""
    
    def __init__(self, toolbox=None, enable_tuning: bool = False):
        super().__init__("hyperparameter_tuning", enabled=enable_tuning)
        self.toolbox = toolbox
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Tune hyperparameters"""
        model = input_data['model']
        X = input_data['X']
        y = input_data['y']
        
        if self.toolbox and hasattr(self.toolbox, 'tuning_kernel') and self.toolbox.tuning_kernel:
            # Use tuning kernel
            tuned_model = self.toolbox.tuning_kernel.tune(model, X, y, **kwargs)
            input_data['model'] = tuned_model
            self.metadata['tuned'] = True
        else:
            # Skip tuning
            self.metadata['tuned'] = False
        
        return input_data


class ModelEvaluationStage(PipelineStage):
    """Stage 3: Model Evaluation"""
    
    def __init__(self, toolbox=None):
        super().__init__("model_evaluation")
        self.toolbox = toolbox
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Evaluate model"""
        model = input_data['model']
        X = input_data['X']
        y = input_data['y']
        
        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
        else:
            y_pred = model
        
        # Compute metrics
        if self.toolbox and hasattr(self.toolbox, 'eval_kernel') and self.toolbox.eval_kernel:
            # Use evaluation kernel (remove model_name from kwargs if present)
            eval_kwargs = {k: v for k, v in kwargs.items() if k != 'model_name'}
            metrics = self.toolbox.eval_kernel.evaluate(y, y_pred, **eval_kwargs)
        else:
            # Simple metrics
            from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
            
            if len(np.unique(y)) < 20:  # Classification
                metrics = {
                    'accuracy': accuracy_score(y, y_pred),
                    'task_type': 'classification'
                }
            else:  # Regression
                metrics = {
                    'r2_score': r2_score(y, y_pred),
                    'mse': mean_squared_error(y, y_pred),
                    'task_type': 'regression'
                }
        
        input_data['evaluation_metrics'] = metrics
        input_data['metrics'].update(metrics)
        
        self.metadata['metrics'] = metrics
        
        return input_data


class ModelValidationStage(PipelineStage):
    """Stage 4: Model Validation"""
    
    def __init__(self, toolbox=None, validation_split: float = 0.2):
        super().__init__("model_validation")
        self.toolbox = toolbox
        self.validation_split = validation_split
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Validate model"""
        X = input_data['X']
        y = input_data['y']
        validation_split = kwargs.get('validation_split', self.validation_split)
        
        if validation_split > 0:
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train on training set
            model = input_data['model']
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            
            # Evaluate on validation set
            if hasattr(model, 'predict'):
                y_val_pred = model.predict(X_val)
                
                # Compute validation metrics
                from sklearn.metrics import accuracy_score, r2_score
                
                if len(np.unique(y)) < 20:  # Classification
                    val_metric = accuracy_score(y_val, y_val_pred)
                    metric_name = 'val_accuracy'
                else:  # Regression
                    val_metric = r2_score(y_val, y_val_pred)
                    metric_name = 'val_r2_score'
                
                input_data['validation_metrics'] = {metric_name: val_metric}
                input_data['metrics'][metric_name] = val_metric
                
                self.metadata['validation_split'] = validation_split
                self.metadata['validation_metric'] = val_metric
        
        return input_data


class ModelRegistryStage(PipelineStage):
    """Stage 5: Model Registry"""
    
    def __init__(self, toolbox=None, enable_registry: bool = True):
        super().__init__("model_registry", enabled=enable_registry)
        self.toolbox = toolbox
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Register model"""
        model = input_data['model']
        model_name = kwargs.get('model_name', 'default')
        metadata = input_data.get('metrics', {})
        
        if self.toolbox and hasattr(self.toolbox, 'model_registry') and self.toolbox.model_registry:
            try:
                model_id = self.toolbox.register_model(
                    model,
                    model_name=model_name,
                    metadata=metadata,
                    **kwargs
                )
                input_data['model_id'] = model_id
                self.metadata['model_id'] = model_id
                self.metadata['registered'] = True
            except Exception as e:
                logger.warning(f"[ModelRegistryStage] Failed to register model: {e}")
                self.metadata['registered'] = False
        else:
            self.metadata['registered'] = False
        
        return input_data


class TrainingPipeline(BasePipeline):
    """
    Training Pipeline
    
    Orchestrates:
    1. Model Training
    2. Hyperparameter Tuning
    3. Model Evaluation
    4. Model Validation
    5. Model Registry
    """
    
    def __init__(self, toolbox=None, enable_tuning: bool = False, enable_registry: bool = True):
        """
        Initialize training pipeline
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
        enable_tuning : bool, default=False
            Whether to enable hyperparameter tuning
        enable_registry : bool, default=True
            Whether to enable model registry
        """
        super().__init__("training_pipeline", toolbox)
        
        # Add stages
        self.add_stage(ModelTrainingStage(toolbox))
        self.add_stage(HyperparameterTuningStage(toolbox, enable_tuning))
        self.add_stage(ModelEvaluationStage(toolbox))
        self.add_stage(ModelValidationStage(toolbox))
        self.add_stage(ModelRegistryStage(toolbox, enable_registry))
    
    def execute(self, X: np.ndarray, y: np.ndarray, model_name: str = "default", **kwargs) -> Dict[str, Any]:
        """
        Execute training pipeline
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training labels
        model_name : str, default="default"
            Name for registered model
        **kwargs
            Additional parameters for stages
            
        Returns
        -------
        result : dict
            Training result with model, metrics, and metadata
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Start monitoring if enabled
        if self.monitor:
            metrics = self.monitor.start_pipeline(self.name)
        
        # Execute stages sequentially
        result = (X, y)
        for stage in self.stages:
            if stage.enabled:
                result = stage.run(result, monitor=self.monitor, retry_handler=self.retry_handler,
                                  debugger=self.debugger, model_name=model_name, **kwargs)
                if isinstance(result, dict):
                    self.state[stage.name] = {
                        'metadata': stage.metadata
                    }
        
        # End monitoring if enabled
        if self.monitor and self.monitor.current_metrics:
            self.monitor.end_pipeline()
        
        # Store final result in state
        self.state['final_result'] = result
        self.state['model_name'] = model_name
        
        logger.info(f"[TrainingPipeline] Pipeline completed. Model: {result.get('model_id', 'unregistered')}")
        
        return result
