"""
Simple ML Tasks
Simplified interfaces for common ML tasks

Features:
- One-line ML training
- Simple API for common tasks
- Automatic model selection
- Easy-to-use interfaces
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import warnings

sys.path.insert(0, str(Path(__file__).parent))


class SimpleMLTasks:
    """
    Simple ML Tasks
    
    Simplified interfaces for common ML tasks
    """
    
    def __init__(self):
        """Initialize Simple ML Tasks"""
        self.dependencies = ['sklearn', 'numpy']
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check dependencies"""
        try:
            import sklearn
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            warnings.warn("sklearn not available. Simple ML tasks will be limited.")
    
    def train_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Train classifier (simple interface)
        
        Args:
            X: Features
            y: Labels
            model_type: 'auto', 'random_forest', 'svm', 'logistic', 'knn'
            
        Returns:
            Trained model and metrics
        """
        if not self.sklearn_available:
            return {'error': 'sklearn required'}
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, classification_report
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Select model
        if model_type == 'auto':
            # Simple heuristic: use RandomForest for most cases
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            model = SVC(random_state=42)
        elif model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'classification_report': report,
            'model_type': model_type
        }
    
    def train_regressor(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Train regressor (simple interface)
        
        Args:
            X: Features
            y: Target values
            model_type: 'auto', 'random_forest', 'linear', 'svr', 'knn'
            
        Returns:
            Trained model and metrics
        """
        if not self.sklearn_available:
            return {'error': 'sklearn required'}
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Select model
        if model_type == 'auto':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear':
            model = Ridge(random_state=42)
        elif model_type == 'svr':
            model = SVR()
        elif model_type == 'knn':
            model = KNeighborsRegressor(n_neighbors=5)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            'model': model,
            'r2_score': r2,
            'mse': mse,
            'mae': mae,
            'model_type': model_type
        }
    
    def predict(
        self,
        model: Any,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions (simple interface)
        
        Args:
            model: Trained model
            X: Features
            
        Returns:
            Predictions
        """
        return model.predict(X)
    
    def predict_proba(
        self,
        model: Any,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Get prediction probabilities (simple interface)
        
        Args:
            model: Trained model
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            return None
    
    def quick_train(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Quick train (auto-detect task and train)
        
        Args:
            X: Features
            y: Labels or target values
            
        Returns:
            Trained model and results
        """
        # Auto-detect task type
        if len(np.unique(y)) < 20 and np.all(y == y.astype(int)):
            # Classification
            return self.train_classifier(X, y, model_type='auto')
        else:
            # Regression
            return self.train_regressor(X, y, model_type='auto')
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'sklearn': 'scikit-learn>=1.3.0',
            'numpy': 'numpy>=1.26.0'
        }
