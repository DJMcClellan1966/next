"""
Ensemble Learning with AdvancedDataPreprocessor
Best practices for combining multiple models to improve performance
"""
import sys
from pathlib import Path
import time
from typing import List, Dict, Tuple, Any, Optional, Callable
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessor import AdvancedDataPreprocessor
from ml_evaluation import MLEvaluator

# Try to import sklearn for ensemble learning
try:
    from sklearn.ensemble import (
        VotingClassifier, VotingRegressor,
        BaggingClassifier, BaggingRegressor,
        AdaBoostClassifier, AdaBoostRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        RandomForestClassifier, RandomForestRegressor
    )
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class EnsembleLearner:
    """
    Ensemble Learning with AdvancedDataPreprocessor
    
    Features:
    - Voting ensembles (hard/soft voting)
    - Bagging ensembles
    - Boosting ensembles
    - Stacking ensembles
    - Custom ensemble combinations
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.ensemble_results = []
    
    def create_voting_ensemble(
        self,
        models: List[Tuple[str, Any]],
        task_type: str = 'classification',
        voting: str = 'hard',  # 'hard' or 'soft' (for classification only)
        weights: Optional[List[float]] = None
    ) -> Any:
        """
        Create voting ensemble
        
        Args:
            models: List of (name, model) tuples
            task_type: 'classification' or 'regression'
            voting: 'hard' (majority vote) or 'soft' (probability average) - classification only
            weights: Optional weights for each model
            
        Returns:
            Voting ensemble model
        """
        if not SKLEARN_AVAILABLE:
            return None
        
        if task_type == 'classification':
            ensemble = VotingClassifier(
                estimators=models,
                voting=voting,
                weights=weights,
                n_jobs=-1
            )
        else:
            ensemble = VotingRegressor(
                estimators=models,
                weights=weights,
                n_jobs=-1
            )
        
        return ensemble
    
    def create_bagging_ensemble(
        self,
        base_model: Any,
        n_estimators: int = 10,
        task_type: str = 'classification'
    ) -> Any:
        """
        Create bagging ensemble
        
        Args:
            base_model: Base model to bag
            n_estimators: Number of models in ensemble
            task_type: 'classification' or 'regression'
            
        Returns:
            Bagging ensemble model
        """
        if not SKLEARN_AVAILABLE:
            return None
        
        if task_type == 'classification':
            ensemble = BaggingClassifier(
                estimator=base_model,
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            ensemble = BaggingRegressor(
                estimator=base_model,
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        return ensemble
    
    def create_boosting_ensemble(
        self,
        base_model: Any,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        task_type: str = 'classification'
    ) -> Any:
        """
        Create boosting ensemble
        
        Args:
            base_model: Base model to boost
            n_estimators: Number of models in ensemble
            learning_rate: Learning rate
            task_type: 'classification' or 'regression'
            
        Returns:
            Boosting ensemble model
        """
        if not SKLEARN_AVAILABLE:
            return None
        
        if task_type == 'classification':
            ensemble = AdaBoostClassifier(
                estimator=base_model,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=self.random_state
            )
        else:
            ensemble = AdaBoostRegressor(
                estimator=base_model,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=self.random_state
            )
        
        return ensemble
    
    def create_stacking_ensemble(
        self,
        base_models: List[Tuple[str, Any]],
        meta_model: Any,
        task_type: str = 'classification',
        cv_folds: int = 5
    ) -> Any:
        """
        Create stacking ensemble
        
        Args:
            base_models: List of (name, model) tuples for base layer
            meta_model: Meta-learner for final prediction
            task_type: 'classification' or 'regression'
            cv_folds: Cross-validation folds for stacking
            
        Returns:
            Stacking ensemble (custom implementation)
        """
        return StackingEnsemble(
            base_models=base_models,
            meta_model=meta_model,
            task_type=task_type,
            cv_folds=cv_folds,
            random_state=self.random_state
        )
    
    def evaluate_ensemble(
        self,
        ensemble: Any,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = 'classification',
        cv_folds: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble model
        
        Args:
            ensemble: Ensemble model
            X: Features
            y: Labels
            task_type: 'classification' or 'regression'
            cv_folds: Number of CV folds
            verbose: Print detailed results
            
        Returns:
            Dictionary with evaluation results
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        start_time = time.time()
        
        # Encode labels for classification
        if task_type == 'classification':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y
            le = None
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=self.random_state,
            stratify=y_encoded if task_type == 'classification' else None
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = ensemble.predict(X_train)
        y_test_pred = ensemble.predict(X_test)
        
        # Calculate metrics
        if task_type == 'classification':
            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)
            scoring = 'accuracy'
        else:
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            scoring = 'neg_mean_squared_error'
        
        # Cross-validation (skip for custom stacking ensemble)
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        
        if isinstance(ensemble, StackingEnsemble):
            # Manual CV for custom ensemble
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            cv_scores_list = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                # Recreate ensemble with same parameters
                # Create fresh meta-model
                if hasattr(ensemble.meta_model, 'get_params'):
                    meta_model_params = ensemble.meta_model.get_params()
                    fresh_meta_model = type(ensemble.meta_model)(**meta_model_params)
                else:
                    fresh_meta_model = type(ensemble.meta_model)()
                
                ensemble_cv = StackingEnsemble(
                    base_models=ensemble.base_models,
                    meta_model=fresh_meta_model,
                    task_type=ensemble.task_type,
                    cv_folds=ensemble.cv_folds,
                    random_state=ensemble.random_state
                )
                ensemble_cv.fit(X_train[train_idx], y_train[train_idx])
                val_pred = ensemble_cv.predict(X_train[val_idx])
                if task_type == 'classification':
                    cv_scores_list.append(accuracy_score(y_train[val_idx], val_pred))
                else:
                    cv_scores_list.append(r2_score(y_train[val_idx], val_pred))
            cv_scores = np.array(cv_scores_list)
        else:
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            cv_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring=scoring)
        
        results = {
            'task_type': task_type,
            'train_score': float(train_score),
            'test_score': float(test_score),
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'cv_scores': cv_scores.tolist(),
            'overfitting_gap': float(train_score - test_score),
            'processing_time': time.time() - start_time
        }
        
        if verbose:
            self._print_ensemble_results(results)
        
        self.ensemble_results.append(results)
        return results
    
    def _print_ensemble_results(self, results: Dict):
        """Print ensemble evaluation results"""
        print("\n" + "="*80)
        print("ENSEMBLE EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nTask Type: {results['task_type']}")
        print(f"Train Score: {results['train_score']:.4f}")
        print(f"Test Score: {results['test_score']:.4f}")
        print(f"CV Mean: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        print(f"Overfitting Gap: {results['overfitting_gap']:.4f}")
        print(f"Processing Time: {results['processing_time']:.3f}s")
        print("="*80 + "\n")


class StackingEnsemble:
    """
    Custom Stacking Ensemble Implementation
    
    Uses base models to create predictions, then meta-model learns from those
    """
    
    def __init__(
        self,
        base_models: List[Tuple[str, Any]],
        meta_model: Any,
        task_type: str = 'classification',
        cv_folds: int = 5,
        random_state: int = 42
    ):
        # Make sklearn-compatible
        if SKLEARN_AVAILABLE:
            from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
            if task_type == 'classification':
                class StackingClassifier(StackingEnsemble, BaseEstimator, ClassifierMixin):
                    pass
                self.__class__ = StackingClassifier
            else:
                class StackingRegressor(StackingEnsemble, BaseEstimator, RegressorMixin):
                    pass
                self.__class__ = StackingRegressor
        self.base_models = base_models
        self.meta_model = meta_model
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.fitted_base_models = []
        self.fitted_meta_model = None
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for sklearn compatibility"""
        return {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'task_type': self.task_type,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state
        }
    
    def set_params(self, **params) -> 'StackingEnsemble':
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit stacking ensemble"""
        from sklearn.model_selection import StratifiedKFold, KFold
        
        if self.task_type == 'classification':
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Train base models and create meta-features
        meta_features = []
        
        for name, model in self.base_models:
            # Create fresh model instance for each base model
            if hasattr(model, 'get_params'):
                model_params = model.get_params()
                fresh_model = type(model)(**model_params)
            else:
                fresh_model = type(model)()
            # Train on full data
            fresh_model.fit(X, y)
            self.fitted_base_models.append((name, fresh_model))  # Store fitted model, not template
            
            # Create out-of-fold predictions for meta-features
            fold_predictions = np.zeros(len(X))
            
            for train_idx, val_idx in cv.split(X, y):
                # Create fresh model for each fold
                if hasattr(model, 'get_params'):
                    fold_model_params = model.get_params()
                    fold_model = type(model)(**fold_model_params)
                else:
                    fold_model = type(model)()
                fold_model.fit(X[train_idx], y[train_idx])
                
                if self.task_type == 'classification':
                    fold_predictions[val_idx] = fold_model.predict_proba(X[val_idx])[:, 1]
                else:
                    fold_predictions[val_idx] = fold_model.predict(X[val_idx])
            
            meta_features.append(fold_predictions)
        
        # Stack meta-features
        X_meta = np.column_stack(meta_features)
        
        # Train meta-model
        self.meta_model.fit(X_meta, y)
        self.fitted_meta_model = self.meta_model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking ensemble"""
        # Get predictions from base models
        meta_features = []
        
        for name, model in self.fitted_base_models:
            if self.task_type == 'classification':
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            meta_features.append(pred)
        
        # Stack meta-features
        X_meta = np.column_stack(meta_features)
        
        # Predict with meta-model
        return self.fitted_meta_model.predict(X_meta)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for sklearn compatibility"""
        return {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'task_type': self.task_type,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state
        }
    
    def set_params(self, **params) -> 'StackingEnsemble':
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class PreprocessorEnsemble:
    """
    Ensemble Learning with Multiple Preprocessors
    
    Uses different preprocessing strategies and combines results
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessors = []
        self.ensemble_results = []
    
    def add_preprocessor(
        self,
        name: str,
        preprocessor: AdvancedDataPreprocessor
    ):
        """Add preprocessor to ensemble"""
        self.preprocessors.append((name, preprocessor))
    
    def preprocess_ensemble(
        self,
        raw_data: List[str],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Preprocess data with multiple preprocessors and combine results
        
        Args:
            raw_data: Raw text data
            verbose: Print detailed progress
            
        Returns:
            Dictionary with ensemble preprocessing results
        """
        ensemble_results = {
            'original_count': len(raw_data),
            'preprocessors': {},
            'combined_embeddings': None,
            'consensus_categories': {},
            'avg_quality': 0.0
        }
        
        all_embeddings = []
        all_categories = defaultdict(list)
        quality_scores = []
        
        for name, preprocessor in self.preprocessors:
            if verbose:
                print(f"\n[Preprocessor: {name}]")
            
            results = preprocessor.preprocess(raw_data, verbose=verbose)
            ensemble_results['preprocessors'][name] = results
            
            # Collect embeddings
            if results['compressed_embeddings'] is not None:
                all_embeddings.append(results['compressed_embeddings'])
            else:
                # Use original embeddings
                embeddings = np.array([
                    preprocessor.quantum_kernel.embed(item)
                    for item in results['deduplicated']
                ])
                all_embeddings.append(embeddings)
            
            # Collect categories
            for category, items in results['categorized'].items():
                all_categories[category].extend(items)
            
            # Collect quality scores
            quality_scores.append(results['stats']['avg_quality'])
        
        # Combine embeddings (average)
        if all_embeddings:
            try:
                # Find common dimensions
                max_samples = max(emb.shape[0] for emb in all_embeddings)
                max_features = max(emb.shape[1] for emb in all_embeddings)
                
                # Pad all embeddings to same shape
                padded_embeddings = []
                for emb in all_embeddings:
                    # Create padded embedding with same shape
                    padded_emb = np.zeros((max_samples, max_features))
                    
                    # Copy original embedding
                    samples_to_copy = min(emb.shape[0], max_samples)
                    features_to_copy = min(emb.shape[1], max_features)
                    padded_emb[:samples_to_copy, :features_to_copy] = emb[:samples_to_copy, :features_to_copy]
                    
                    padded_embeddings.append(padded_emb)
                
                # Convert to numpy array first, then average
                padded_embeddings = np.array(padded_embeddings)
                
                # Average embeddings across preprocessors (axis 0 = preprocessors, axis 1 = samples, axis 2 = features)
                ensemble_results['combined_embeddings'] = np.mean(padded_embeddings, axis=0)
                
            except Exception as e:
                # Fallback: use first embedding if combination fails
                if all_embeddings:
                    ensemble_results['combined_embeddings'] = all_embeddings[0]
                if verbose:
                    print(f"  Warning: Could not combine embeddings: {e}. Using first preprocessor's embeddings.")
        
        # Consensus categories (items that appear in multiple preprocessors)
        for category, items in all_categories.items():
            from collections import Counter
            item_counts = Counter(items)
            # Items that appear in at least 2 preprocessors
            consensus_items = [item for item, count in item_counts.items() if count >= 2]
            if consensus_items:
                ensemble_results['consensus_categories'][category] = consensus_items
        
        # Average quality
        ensemble_results['avg_quality'] = np.mean(quality_scores) if quality_scores else 0.0
        
        if verbose:
            print(f"\n[Ensemble Results]")
            print(f"  Preprocessors: {len(self.preprocessors)}")
            print(f"  Combined embeddings shape: {ensemble_results['combined_embeddings'].shape}")
            print(f"  Consensus categories: {len(ensemble_results['consensus_categories'])}")
            print(f"  Average quality: {ensemble_results['avg_quality']:.4f}")
        
        self.ensemble_results.append(ensemble_results)
        return ensemble_results
