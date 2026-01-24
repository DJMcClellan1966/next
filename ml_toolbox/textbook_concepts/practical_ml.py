"""
Practical Machine Learning (Hands-On ML - Géron)

Implements:
- Feature Engineering
- Model Selection
- Hyperparameter Tuning
- Ensemble Methods
- Cross-Validation
- Production ML
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from itertools import product
import logging

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Feature Engineering Techniques
    
    Comprehensive feature engineering methods
    """
    
    @staticmethod
    def polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Create polynomial features
        
        Parameters
        ----------
        X : array
            Input features
        degree : int
            Polynomial degree
            
        Returns
        -------
        features : array
            Polynomial features
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Generate polynomial combinations
        features = [X]
        
        for d in range(2, degree + 1):
            for combo in product(range(n_features), repeat=d):
                if len(set(combo)) == len(combo):  # No repeated indices
                    feature = np.prod(X[:, combo], axis=1)
                    features.append(feature.reshape(-1, 1))
        
        return np.hstack(features)
    
    @staticmethod
    def interaction_features(X: np.ndarray) -> np.ndarray:
        """Create interaction features"""
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        interactions = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
        
        if interactions:
            return np.hstack([X] + interactions)
        return X
    
    @staticmethod
    def binning(X: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """Create binned features"""
        X = np.asarray(X)
        binned = np.zeros_like(X)
        
        for i in range(X.shape[1]):
            binned[:, i] = np.digitize(X[:, i], np.linspace(X[:, i].min(), X[:, i].max(), n_bins))
        
        return binned
    
    @staticmethod
    def log_transform(X: np.ndarray) -> np.ndarray:
        """Log transformation"""
        X = np.asarray(X)
        return np.log1p(X - X.min(axis=0) + 1)  # log1p to handle zeros
    
    @staticmethod
    def normalize(X: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Normalize features"""
        X = np.asarray(X)
        
        if method == 'standard':
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            return (X - mean) / (std + 1e-10)
        elif method == 'minmax':
            min_val = np.min(X, axis=0)
            max_val = np.max(X, axis=0)
            return (X - min_val) / (max_val - min_val + 1e-10)
        else:
            return X


class ModelSelection:
    """
    Model Selection
    
    Select best model for task
    """
    
    def __init__(self, models: Dict[str, Any], scoring: str = 'accuracy'):
        """
        Initialize model selection
        
        Parameters
        ----------
        models : dict
            Dictionary of {name: model} pairs
        scoring : str
            Scoring metric
        """
        self.models = models
        self.scoring = scoring
        self.best_model_ = None
        self.best_score_ = 0.0
    
    def select(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> str:
        """
        Select best model using cross-validation
        
        Parameters
        ----------
        X : array
            Training data
        y : array
            Target values
        cv : int
            Number of folds
            
        Returns
        -------
        best_model_name : str
            Name of best model
        """
        from .practical_ml import CrossValidation
        
        best_score = -np.inf
        best_name = None
        
        for name, model in self.models.items():
            cv_scores = CrossValidation.cross_val_score(model, X, y, cv=cv, scoring=self.scoring)
            avg_score = np.mean(cv_scores)
            
            logger.info(f"[ModelSelection] {name}: {avg_score:.4f} (+/- {np.std(cv_scores):.4f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_name = name
        
        self.best_model_ = self.models[best_name]
        self.best_score_ = best_score
        
        logger.info(f"[ModelSelection] Best model: {best_name} (score: {best_score:.4f})")
        return best_name


class HyperparameterTuning:
    """
    Hyperparameter Tuning
    
    Grid search and random search
    """
    
    @staticmethod
    def grid_search(model_class: type, param_grid: Dict[str, List],
                   X: np.ndarray, y: np.ndarray, cv: int = 5,
                   scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Grid search for hyperparameters
        
        Parameters
        ----------
        model_class : type
            Model class
        param_grid : dict
            Parameter grid {param: [values]}
        X : array
            Training data
        y : array
            Target values
        cv : int
            Number of folds
        scoring : str
            Scoring metric
            
        Returns
        -------
        best_params : dict
            Best parameters
        """
        from .practical_ml import CrossValidation
        
        best_score = -np.inf
        best_params = None
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            
            # Create model with parameters
            try:
                model = model_class(**params)
            except TypeError:
                # If model doesn't accept these params, skip
                continue
            
            # Cross-validation
            cv_scores = CrossValidation.cross_val_score(model, X, y, cv=cv, scoring=scoring)
            avg_score = np.mean(cv_scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        logger.info(f"[HyperparameterTuning] Best params: {best_params} (score: {best_score:.4f})")
        return best_params
    
    @staticmethod
    def random_search(model_class: type, param_distributions: Dict[str, Callable],
                     X: np.ndarray, y: np.ndarray, n_iter: int = 10,
                     cv: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Random search for hyperparameters
        
        Parameters
        ----------
        model_class : type
            Model class
        param_distributions : dict
            Parameter distributions {param: sampling_function}
        X : array
            Training data
        y : array
            Target values
        n_iter : int
            Number of iterations
        cv : int
            Number of folds
        scoring : str
            Scoring metric
            
        Returns
        -------
        best_params : dict
            Best parameters
        """
        from .practical_ml import CrossValidation
        
        best_score = -np.inf
        best_params = None
        
        for _ in range(n_iter):
            # Sample parameters
            params = {name: dist() for name, dist in param_distributions.items()}
            
            # Create model
            try:
                model = model_class(**params)
            except TypeError:
                continue
            
            # Cross-validation
            cv_scores = CrossValidation.cross_val_score(model, X, y, cv=cv, scoring=scoring)
            avg_score = np.mean(cv_scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        logger.info(f"[HyperparameterTuning] Best params: {best_params} (score: {best_score:.4f})")
        return best_params


class EnsembleMethods:
    """
    Ensemble Methods
    
    Combine multiple models
    """
    
    @staticmethod
    def voting_classifier(models: List[Any], X: np.ndarray, y: np.ndarray,
                         voting: str = 'hard') -> np.ndarray:
        """
        Voting classifier
        
        Parameters
        ----------
        models : list
            List of models
        X : array
            Training data
        y : array
            Target values
        voting : str
            Voting method ('hard' or 'soft')
            
        Returns
        -------
        predictions : array
            Ensemble predictions
        """
        # Train all models
        for model in models:
            model.fit(X, y)
        
        # Get predictions
        if voting == 'hard':
            predictions = np.array([model.predict(X) for model in models])
            # Majority vote
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions
            )
        else:  # soft
            # Average probabilities
            probas = np.array([model.predict_proba(X) for model in models])
            ensemble_proba = np.mean(probas, axis=0)
            ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return ensemble_pred
    
    @staticmethod
    def bagging(model_class: type, X: np.ndarray, y: np.ndarray,
               n_estimators: int = 10, sample_ratio: float = 0.8) -> List[Any]:
        """
        Bagging (Bootstrap Aggregating)
        
        Parameters
        ----------
        model_class : type
            Model class
        X : array
            Training data
        y : array
            Target values
        n_estimators : int
            Number of estimators
        sample_ratio : float
            Sample ratio for each estimator
            
        Returns
        -------
        models : list
            Trained models
        """
        models = []
        n_samples = len(X)
        n_sample = int(n_samples * sample_ratio)
        
        for _ in range(n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_sample, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Train model
            model = model_class()
            model.fit(X_sample, y_sample)
            models.append(model)
        
        return models
    
    @staticmethod
    def stacking(models: List[Any], meta_model: Any, X: np.ndarray, y: np.ndarray,
                cv: int = 5) -> Any:
        """
        Stacking
        
        Parameters
        ----------
        models : list
            Base models
        meta_model : any
            Meta-learner
        X : array
            Training data
        y : array
            Target values
        cv : int
            Number of folds
            
        Returns
        -------
        meta_model : any
            Trained meta-model
        """
        from .practical_ml import CrossValidation
        
        # Generate meta-features using cross-validation
        meta_features = np.zeros((len(X), len(models)))
        
        for fold_idx, (train_idx, val_idx) in enumerate(CrossValidation.k_fold_split(X, y, cv)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            for model_idx, model in enumerate(models):
                # Train on fold
                model.fit(X_train, y_train)
                # Predict on validation
                if hasattr(model, 'predict_proba'):
                    meta_features[val_idx, model_idx] = model.predict_proba(X_val)[:, 1]
                else:
                    meta_features[val_idx, model_idx] = model.predict(X_val)
        
        # Train meta-model
        meta_model.fit(meta_features, y)
        return meta_model


class CrossValidation:
    """
    Cross-Validation
    
    K-fold and stratified cross-validation
    """
    
    @staticmethod
    def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 5):
        """
        K-fold split generator
        
        Parameters
        ----------
        X : array
            Data
        y : array
            Target
        k : int
            Number of folds
            
        Yields
        ------
        train_idx, val_idx : arrays
            Training and validation indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        fold_size = n_samples // k
        
        for i in range(k):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < k - 1 else n_samples
            
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            yield train_idx, val_idx
    
    @staticmethod
    def cross_val_score(model: Any, X: np.ndarray, y: np.ndarray,
                       cv: int = 5, scoring: str = 'accuracy') -> np.ndarray:
        """
        Cross-validation scores
        
        Parameters
        ----------
        model : any
            Model with fit/predict methods
        X : array
            Training data
        y : array
            Target values
        cv : int
            Number of folds
        scoring : str
            Scoring metric
            
        Returns
        -------
        scores : array
            Cross-validation scores
        """
        scores = []
        
        for train_idx, val_idx in CrossValidation.k_fold_split(X, y, cv):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Score
            if scoring == 'accuracy':
                score = np.mean(y_pred == y_val)
            elif scoring == 'r2':
                from ml_toolbox.core_models.evaluation_metrics import precision_score
                score = 1 - np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
            else:
                score = np.mean(y_pred == y_val)
            
            scores.append(score)
        
        return np.array(scores)


class ProductionML:
    """
    Production ML Practices
    
    Model versioning, monitoring, deployment
    """
    
    def __init__(self):
        self.model_registry: Dict[str, Any] = {}
        self.model_versions: Dict[str, List] = {}
    
    def register_model(self, model_name: str, model: Any, version: str = "1.0",
                      metadata: Dict = None):
        """Register model for production"""
        if model_name not in self.model_registry:
            self.model_registry[model_name] = {}
            self.model_versions[model_name] = []
        
        self.model_registry[model_name][version] = {
            'model': model,
            'metadata': metadata or {},
            'registered_at': None  # Would use datetime
        }
        self.model_versions[model_name].append(version)
        
        logger.info(f"[ProductionML] Registered {model_name} v{version}")
    
    def get_model(self, model_name: str, version: str = "latest") -> Optional[Any]:
        """Get model from registry"""
        if model_name not in self.model_registry:
            return None
        
        if version == "latest":
            versions = self.model_versions[model_name]
            version = versions[-1] if versions else None
        
        if version and version in self.model_registry[model_name]:
            return self.model_registry[model_name][version]['model']
        
        return None
    
    def monitor_model(self, model_name: str, predictions: np.ndarray,
                     actuals: np.ndarray) -> Dict[str, Any]:
        """Monitor model performance"""
        metrics = {
            'accuracy': np.mean(predictions == actuals),
            'error_rate': np.mean(predictions != actuals),
            'n_predictions': len(predictions)
        }
        
        logger.info(f"[ProductionML] Model {model_name} metrics: {metrics}")
        return metrics
