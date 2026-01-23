"""
Machine Learning Toolbox
Organized into four compartments:
1. Data: Preprocessing, validation, transformation
2. Infrastructure: Kernels, AI components, LLM
3. Algorithms: Models, evaluation, tuning, ensembles
4. MLOps: Production deployment, monitoring, A/B testing, experiment tracking

Also includes Advanced ML Toolbox for big data and advanced features
"""
from typing import Any, Optional, Dict
from .compartment1_data import DataCompartment
from .compartment2_infrastructure import InfrastructureCompartment
from .compartment3_algorithms import AlgorithmsCompartment

# Try to import MLOps compartment
try:
    from .compartment4_mlops import MLOpsCompartment
    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False
    MLOpsCompartment = None

# Import advanced toolbox
try:
    from .advanced import AdvancedMLToolbox
    __all__ = [
        'DataCompartment',
        'InfrastructureCompartment',
        'AlgorithmsCompartment',
        'MLToolbox',
        'AdvancedMLToolbox'
    ]
except ImportError:
    __all__ = [
        'DataCompartment',
        'InfrastructureCompartment',
        'AlgorithmsCompartment',
        'MLToolbox'
    ]


class MLToolbox:
    """
    Complete Machine Learning Toolbox
    
    Four compartments:
    1. Data: Preprocessing, validation, transformation
    2. Infrastructure: Kernels, AI components, LLM
    3. Algorithms: Models, evaluation, tuning, ensembles
    4. MLOps: Production deployment, monitoring, A/B testing, experiment tracking
    
    Also includes:
    - Medulla Oblongata System: Automatic resource regulation
    - Virtual Quantum Computer: CPU-based quantum simulation (optional)
    """
    
    def __init__(self, include_mlops: bool = True, auto_start_optimizer: bool = True, 
                 enable_caching: bool = True, enable_ml_math: bool = True,
                 check_dependencies: bool = True, verbose_errors: bool = False):
        """
        Initialize ML Toolbox
        
        Args:
            include_mlops: Include MLOps compartment
            auto_start_optimizer: Automatically start Medulla Toolbox Optimizer
            enable_caching: Enable model caching (50-90% faster for repeated operations)
            enable_ml_math: Enable ML Math Optimizer (15-20% faster operations)
        """
        # Initialize model cache
        self.enable_caching = enable_caching
        if enable_caching:
            try:
                from .model_cache import get_model_cache
                self.model_cache = get_model_cache(max_size=100, enable_disk_cache=False)
                print("[MLToolbox] Model caching enabled")
            except Exception as e:
                print(f"[MLToolbox] Warning: Model cache not available: {e}")
                self.model_cache = None
        else:
            self.model_cache = None
        
        # Initialize ML Math Optimizer (automatic optimization)
        self.enable_ml_math = enable_ml_math
        self._ml_math_optimizer = None
        if enable_ml_math:
            try:
                from ml_math_optimizer import get_ml_math_optimizer
                self._ml_math_optimizer = get_ml_math_optimizer()
                print("[MLToolbox] ML Math Optimizer enabled (15-20% faster operations)")
            except Exception as e:
                print(f"[MLToolbox] Warning: ML Math Optimizer not available: {e}")
        
        # Initialize Medulla Toolbox Optimizer (automatic ML operation optimization)
        self.optimizer = None
        if auto_start_optimizer:
            try:
                from medulla_toolbox_optimizer import MedullaToolboxOptimizer, MLTaskType
                self.optimizer = MedullaToolboxOptimizer(
                    max_cpu_percent=85.0,
                    max_memory_percent=80.0,
                    min_cpu_reserve=15.0,
                    min_memory_reserve_mb=1024.0,
                    enable_caching=enable_caching,
                    enable_adaptive_allocation=True
                )
                self.optimizer.start_regulation()
                self.MLTaskType = MLTaskType  # Expose for use
                print("[MLToolbox] Medulla Toolbox Optimizer started (automatic ML operation optimization)")
            except ImportError as e:
                print(f"[MLToolbox] Warning: Medulla optimizer not available: {e}")
            except Exception as e:
                print(f"[MLToolbox] Warning: Could not start Medulla optimizer: {e}")
        
        # Keep legacy medulla reference for backward compatibility
        self.medulla = self.optimizer
        
        # Initialize compartments (pass medulla to infrastructure)
        self.data = DataCompartment()
        self.infrastructure = InfrastructureCompartment(medulla=self.medulla)
        self.algorithms = AlgorithmsCompartment()
        
        # MLOps compartment (optional)
        if include_mlops and MLOPS_AVAILABLE:
            self.mlops = MLOpsCompartment()
        else:
            self.mlops = None
        
        # Model Registry (automatic)
        try:
            from .model_registry import get_model_registry
            self.model_registry = get_model_registry()
            print("[MLToolbox] Model Registry enabled")
        except Exception as e:
            print(f"[MLToolbox] Warning: Model Registry not available: {e}")
            self.model_registry = None
        
        # Universal Adaptive Preprocessor (optional - replaces 6+ preprocessors)
        try:
            from universal_adaptive_preprocessor import get_universal_preprocessor
            self.universal_preprocessor = get_universal_preprocessor()
            print("[MLToolbox] Universal Adaptive Preprocessor available (AI-powered)")
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_import_error('universal_adaptive_preprocessor', 'Universal Preprocessor', is_optional=True)
            self.universal_preprocessor = None
        
        # AI Model Orchestrator (optional - unified model operations)
        try:
            from ai_model_orchestrator import get_ai_orchestrator
            self.ai_orchestrator = get_ai_orchestrator(toolbox=self)
            print("[MLToolbox] AI Model Orchestrator available (unified model operations)")
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_import_error('ai_model_orchestrator', 'AI Orchestrator', is_optional=True)
            self.ai_orchestrator = None
        
        # AI Ensemble Feature Selector (optional - unifies feature selection)
        try:
            from ai_ensemble_feature_selector import get_ai_ensemble_selector
            self.ai_feature_selector = get_ai_ensemble_selector()
            print("[MLToolbox] AI Ensemble Feature Selector available (unified feature selection)")
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_import_error('ai_ensemble_feature_selector', 'AI Feature Selector', is_optional=True)
            self.ai_feature_selector = None
        
        # Revolutionary Features (mindblowing upgrades) - LAZY LOADED
        # Features load on demand for faster startup
        self._predictive_intelligence = None
        self._self_healing_code = None
        self._natural_language_pipeline = None
        self._collaborative_intelligence = None
        self._auto_optimizer = None
        self._third_eye = None
        self._code_personality = None
        self._code_dreams = None
        self._parallel_universe_testing = None
        self._code_alchemy = None
        self._telepathic_code = None
        
        # Mark as available (will load on first access)
        self._revolutionary_features_available = True
        print("[MLToolbox] Revolutionary Features available (lazy-loaded)")
        print("[MLToolbox] Fun & Daring Features available (lazy-loaded)")
    
    def __repr__(self):
        mlops_info = f", mlops={len(self.mlops.components)}" if self.mlops else ""
        optimizer_info = ", optimizer=active" if self.optimizer and self.optimizer.regulation_running else ""
        return f"MLToolbox(data={len(self.data.components)}, infrastructure={len(self.infrastructure.components)}, algorithms={len(self.algorithms.components)}{mlops_info}{optimizer_info})"
    
    def __enter__(self):
        """Context manager entry - Medulla already started"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - Stop optimizer if running"""
        if self.optimizer and self.optimizer.regulation_running:
            self.optimizer.stop_regulation()
    
    def get_system_status(self):
        """Get optimizer system status"""
        if self.optimizer:
            return self.optimizer.get_system_status()
        return {"status": "optimizer_not_available"}
    
    def optimize_operation(self, operation_name: str, operation_func, task_type=None, use_cache: bool = True, *args, **kwargs):
        """
        Optimize an ML operation using Medulla optimizer
        
        Args:
            operation_name: Name of the operation (for caching)
            operation_func: Function to execute
            task_type: MLTaskType (DATA_PREPROCESSING, MODEL_TRAINING, etc.)
            use_cache: Whether to use result caching
            *args, **kwargs: Arguments to pass to operation_func
        
        Returns:
            Result of operation_func
        """
        if self.optimizer:
            if task_type is None:
                # Default to MODEL_TRAINING
                task_type = self.MLTaskType.MODEL_TRAINING if hasattr(self, 'MLTaskType') else None
            
            if task_type:
                return self.optimizer.optimize_operation(
                    operation_name,
                    operation_func,
                    task_type=task_type,
                    use_cache=use_cache,
                    *args,
                    **kwargs
                )
            else:
                # Fallback if optimizer not available
                return operation_func(*args, **kwargs)
        else:
            # No optimizer, just execute
            return operation_func(*args, **kwargs)
    
    def get_optimization_stats(self):
        """Get optimization statistics"""
        if self.optimizer:
            return self.optimizer.get_optimization_stats()
        return {"status": "optimizer_not_available"}
    
    def get_ml_math_optimizer(self):
        """Get ML Math Optimizer for optimized mathematical operations"""
        if self._ml_math_optimizer:
            return self._ml_math_optimizer
        try:
            from ml_math_optimizer import get_ml_math_optimizer
            self._ml_math_optimizer = get_ml_math_optimizer()
            return self._ml_math_optimizer
        except ImportError:
            raise ImportError("ML Math Optimizer not available. Install required dependencies.")
    
    def fit(self, X, y, task_type: str = 'auto', model_type: str = 'auto', 
            use_cache: bool = True, **kwargs):
        """
        Unified fit method - auto-detects task and trains model
        
        This is the simple, unified API for ML Toolbox.
        Auto-detects task type, preprocesses data, selects model, and trains.
        
        Args:
            X: Input features (numpy array or list)
            y: Target values/labels (numpy array or list)
            task_type: 'auto', 'classification', 'regression', 'clustering'
            model_type: 'auto' or specific model name
            use_cache: Use model caching (50-90% faster for repeated operations)
            **kwargs: Additional parameters for model training
        
        Returns:
            Trained model and metrics
        
        Example:
            >>> toolbox = MLToolbox()
            >>> result = toolbox.fit(X, y)
            >>> model = result['model']
            >>> accuracy = result.get('accuracy', result.get('r2_score'))
        """
        import numpy as np
        
        # Convert to numpy if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Auto-detect task type
        if task_type == 'auto':
            if len(np.unique(y)) < 20 and np.all(y == y.astype(int)):
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        # Check cache first
        if use_cache and self.model_cache:
            cached_model = self.model_cache.get(X, y, kwargs, operation='fit')
            if cached_model is not None:
                print("[MLToolbox] Using cached model (50-90% faster!)")
                return cached_model
        
        # Use simple ML tasks for quick training
        try:
            from simple_ml_tasks import SimpleMLTasks
            simple_tasks = SimpleMLTasks()
            
            if task_type == 'classification':
                result = simple_tasks.train_classifier(X, y, model_type=model_type)
            elif task_type == 'regression':
                result = simple_tasks.train_regressor(X, y, model_type=model_type)
            else:
                result = simple_tasks.quick_train(X, y)
            
            # Cache the result
            if use_cache and self.model_cache:
                self.model_cache.set(X, y, kwargs, operation='fit', model=result)
            
            return result
        except Exception as e:
            # Fallback to manual training
            print(f"[MLToolbox] Warning: Simple training failed, using fallback: {e}")
            # Use optimize_operation for manual training
            if self.optimizer:
                task_type_enum = self.MLTaskType.MODEL_TRAINING if hasattr(self, 'MLTaskType') else None
                def train_func():
                    from simple_ml_tasks import SimpleMLTasks
                    simple_tasks = SimpleMLTasks()
                    return simple_tasks.quick_train(X, y)
                
                result = self.optimize_operation('fit', train_func, task_type=task_type_enum, use_cache=use_cache)
                
                # Cache the result
                if use_cache and self.model_cache:
                    self.model_cache.set(X, y, kwargs, operation='fit', model=result)
                
                return result
            else:
                # No optimizer, just train
                from simple_ml_tasks import SimpleMLTasks
                simple_tasks = SimpleMLTasks()
                result = simple_tasks.quick_train(X, y)
                
                # Cache the result
                if use_cache and self.model_cache:
                    self.model_cache.set(X, y, kwargs, operation='fit', model=result)
                
                return result
    
    def predict(self, model, X, use_cache: bool = True):
        """
        Make predictions using trained model
        
        Args:
            model: Trained model (from fit() or manual training)
            X: Input features
            use_cache: Use prediction caching
        
        Returns:
            Predictions
        """
        import numpy as np
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Check cache
        if use_cache and self.model_cache:
            cached_pred = self.model_cache.get(X, None, {'model_id': id(model)}, operation='predict')
            if cached_pred is not None:
                return cached_pred
        
        # Make prediction
        if hasattr(model, 'predict'):
            pred = model.predict(X)
        elif isinstance(model, dict) and 'model' in model:
            pred = model['model'].predict(X)
        else:
            raise ValueError("Model does not have predict method")
        
        # Cache prediction
        if use_cache and self.model_cache:
            self.model_cache.set(X, None, {'model_id': id(model)}, operation='predict', model=pred)
        
        return pred
    
    def get_cache_stats(self):
        """Get model cache statistics"""
        if self.model_cache:
            return self.model_cache.get_stats()
        return {"status": "caching_disabled"}
    
    def register_model(self, model: Any, model_name: str, version: Optional[str] = None,
                       metadata: Optional[Dict] = None, stage: str = 'dev'):
        """
        Register a model in the model registry
        
        Args:
            model: Trained model
            model_name: Name of the model
            version: Version string (e.g., "1.0.0"). If None, auto-increments
            metadata: Additional metadata (metrics, parameters, etc.)
            stage: Initial stage ('dev', 'staging', 'production', 'archived')
        
        Returns:
            Model identifier (name:version)
        
        Example:
            >>> result = toolbox.fit(X, y)
            >>> model_id = toolbox.register_model(
            ...     result['model'],
            ...     model_name='iris_classifier',
            ...     metadata={'accuracy': result['accuracy']}
            ... )
        """
        if not self.model_registry:
            raise RuntimeError("Model Registry not available")
        
        from .model_registry import ModelStage
        stage_enum = ModelStage(stage.lower())
        
        return self.model_registry.register_model(
            model=model,
            model_name=model_name,
            version=version,
            metadata=metadata,
            stage=stage_enum
        )
    
    def get_registered_model(self, model_id: str):
        """
        Get a registered model by ID
        
        Args:
            model_id: Model identifier (name:version)
        
        Returns:
            Tuple of (model, metadata)
        """
        if not self.model_registry:
            raise RuntimeError("Model Registry not available")
        
        return self.model_registry.get_model(model_id)
    
    def get_quantum_computer(self, num_qubits: int = 8, use_architecture_optimizations: bool = True):
        """
        Get a Virtual Quantum Computer instance (optional/experimental feature)
        
        Note: Quantum simulation is resource-intensive and provides no quantum advantage
        on regular laptops. Consider using ML Math Optimizer instead for better performance.
        """
        try:
            from virtual_quantum_computer import VirtualQuantumComputer
            # Quantum computer is optional - don't allocate resources by default
            return VirtualQuantumComputer(
                num_qubits=num_qubits,
                medulla=None,  # Don't allocate resources for quantum by default
                use_architecture_optimizations=use_architecture_optimizations
            )
        except ImportError:
            raise ImportError("Virtual Quantum Computer not available. Install required dependencies.")
