"""
AutoML Framework
Automated machine learning for ML Toolbox

Features:
- Automated model selection
- Automated hyperparameter tuning
- Automated feature engineering
- Automated pipeline creation
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
import numpy as np
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AutoMLFramework:
    """
    AutoML Framework
    
    Automated machine learning
    """
    
    def __init__(self):
        """Initialize AutoML framework"""
        self.dependencies = ['sklearn', 'numpy']
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check dependencies"""
        try:
            import sklearn
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            warnings.warn("sklearn not available. AutoML features will be limited.")
    
    def automl_pipeline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = 'auto',
        time_budget: int = 300,
        metric: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Automated ML pipeline
        
        Args:
            X: Features
            y: Labels
            task_type: 'classification', 'regression', or 'auto'
            time_budget: Time budget in seconds
            metric: Metric to optimize ('auto', 'accuracy', 'f1', 'roc_auc', 'r2', 'mse')
            
        Returns:
            Dictionary with best model and results
        """
        if not self.sklearn_available:
            return {'error': 'sklearn required for AutoML'}
        
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        import time
        
        # Auto-detect task type
        if task_type == 'auto':
            if len(np.unique(y)) < 20 and np.all(y == y.astype(int)):
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        # Auto-detect metric
        if metric == 'auto':
            if task_type == 'classification':
                metric = 'accuracy'
            else:
                metric = 'r2'
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define models to try
        if task_type == 'classification':
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                'SVM': SVC(random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=5)
            }
            scoring = 'accuracy'
        else:
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Ridge': Ridge(random_state=42),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor(n_neighbors=5)
            }
            scoring = 'r2'
        
        # Try each model
        results = {}
        start_time = time.time()
        
        for name, model in models.items():
            if time.time() - start_time > time_budget:
                break
            
            try:
                # Cross-validation
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                
                # Train on full training set
                model.fit(X_train, y_train)
                
                # Test score
                test_score = model.score(X_test, y_test)
                
                results[name] = {
                    'model': model,
                    'cv_mean': scores.mean(),
                    'cv_std': scores.std(),
                    'test_score': test_score,
                    'time': time.time() - start_time
                }
            except Exception as e:
                results[name] = {'error': str(e)}
        
        # Find best model
        if results:
            best_name = max(results.keys(), 
                          key=lambda k: results[k].get('cv_mean', 0) 
                          if 'error' not in results[k] else -1)
            best_result = results[best_name]
            
            return {
                'best_model': best_result.get('model'),
                'best_model_name': best_name,
                'best_cv_score': best_result.get('cv_mean'),
                'best_test_score': best_result.get('test_score'),
                'all_results': results,
                'task_type': task_type,
                'metric': metric,
                'time_taken': time.time() - start_time
            }
        
        return {'error': 'No models could be trained'}
    
    def automated_feature_engineering(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Automated feature engineering
        
        Args:
            X: Features
            y: Labels (optional, for supervised feature selection)
            methods: List of methods ('pca', 'polynomial', 'interaction')
            
        Returns:
            Engineered features
        """
        if not self.sklearn_available:
            return {'error': 'sklearn required for feature engineering'}
        
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        import time
        
        if methods is None:
            methods = ['pca', 'polynomial']
        
        engineered_features = {}
        start_time = time.time()
        
        # PCA
        if 'pca' in methods:
            try:
                pca = PCA(n_components=min(50, X.shape[1]))
                X_pca = pca.fit_transform(X)
                engineered_features['pca'] = {
                    'features': X_pca,
                    'explained_variance': pca.explained_variance_ratio_.sum(),
                    'n_components': pca.n_components_
                }
            except Exception as e:
                engineered_features['pca'] = {'error': str(e)}
        
        # Polynomial features
        if 'polynomial' in methods:
            try:
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(X)
                # Limit to avoid memory issues
                if X_poly.shape[1] > 1000:
                    # Select top features
                    if y is not None:
                        selector = SelectKBest(
                            f_classif if len(np.unique(y)) < 20 else f_regression,
                            k=min(100, X_poly.shape[1])
                        )
                        X_poly = selector.fit_transform(X_poly, y)
                    else:
                        X_poly = X_poly[:, :100]
                
                engineered_features['polynomial'] = {
                    'features': X_poly,
                    'n_features': X_poly.shape[1]
                }
            except Exception as e:
                engineered_features['polynomial'] = {'error': str(e)}
        
        # Feature selection
        if y is not None and 'selection' in methods:
            try:
                selector = SelectKBest(
                    f_classif if len(np.unique(y)) < 20 else f_regression,
                    k=min(50, X.shape[1])
                )
                X_selected = selector.fit_transform(X, y)
                engineered_features['selection'] = {
                    'features': X_selected,
                    'selected_features': selector.get_support().tolist()
                }
            except Exception as e:
                engineered_features['selection'] = {'error': str(e)}
        
        return {
            'engineered_features': engineered_features,
            'time_taken': time.time() - start_time,
            'original_shape': X.shape,
            'methods_used': methods
        }
    
    def automated_hyperparameter_tuning(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv: int = 5,
        n_iter: int = 20
    ) -> Dict[str, Any]:
        """
        Automated hyperparameter tuning
        
        Args:
            model: Model to tune
            X: Features
            y: Labels
            param_grid: Parameter grid (if None, uses default)
            cv: Cross-validation folds
            n_iter: Number of iterations for random search
            
        Returns:
            Best parameters and model
        """
        if not self.sklearn_available:
            return {'error': 'sklearn required for hyperparameter tuning'}
        
        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        import time
        
        # Default parameter grids
        if param_grid is None:
            model_name = type(model).__name__
            if 'RandomForest' in model_name:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            elif 'LogisticRegression' in model_name:
                param_grid = {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            else:
                param_grid = {}
        
        start_time = time.time()
        
        # Use random search for efficiency
        search = RandomizedSearchCV(
            model, param_grid, cv=cv, n_iter=n_iter,
            scoring='accuracy' if len(np.unique(y)) < 20 else 'r2',
            random_state=42, n_jobs=-1
        )
        
        search.fit(X, y)
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_model': search.best_estimator_,
            'time_taken': time.time() - start_time,
            'cv_results': search.cv_results_
        }
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'sklearn': 'scikit-learn>=1.3.0',
            'numpy': 'numpy>=1.26.0'
        }
