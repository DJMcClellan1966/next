"""
Core Regression and Classification Models

Implements:
- Linear Regression
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVMs)
"""
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LinearRegression:
    """
    Linear Regression Model
    
    y = X @ w + b
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 fit_intercept: bool = True):
        """
        Initialize linear regression
        
        Parameters
        ----------
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum iterations
        fit_intercept : bool
            Whether to fit intercept term
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.n_features_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit linear regression model
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Add intercept term
        if self.fit_intercept:
            X = np.column_stack([np.ones(n_samples), X])
            n_features += 1
        
        # Initialize weights
        self.coef_ = np.zeros(n_features)
        
        # Gradient descent
        for iteration in range(self.max_iter):
            # Predictions
            y_pred = X @ self.coef_
            
            # Error
            error = y_pred - y
            
            # Gradient
            gradient = X.T @ error / n_samples
            
            # Update weights
            self.coef_ -= self.learning_rate * gradient
            
            # Check convergence
            if iteration % 100 == 0:
                mse = np.mean(error ** 2)
                if mse < 1e-6:
                    break
        
        # Separate intercept and coefficients
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0
        
        logger.info(f"[LinearRegression] Fitted with {n_samples} samples, {n_features} features")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using linear regression
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        predictions : array, shape (n_samples,)
            Predicted values
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.coef_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return X @ self.coef_ + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score
        
        Parameters
        ----------
        X : array-like
            Input data
        y : array-like
            True values
            
        Returns
        -------
        r2 : float
            R² score
        """
        y_pred = self.predict(X)
        y = np.asarray(y).ravel()
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


class LogisticRegression:
    """
    Logistic Regression Model
    
    Binary classification using sigmoid function
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000,
                 fit_intercept: bool = True, regularization: float = 0.0):
        """
        Initialize logistic regression
        
        Parameters
        ----------
        learning_rate : float
            Learning rate
        max_iter : int
            Maximum iterations
        fit_intercept : bool
            Whether to fit intercept
        regularization : float
            L2 regularization strength
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.coef_ = None
        self.intercept_ = None
        self.n_features_ = None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function"""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit logistic regression model
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Binary target values (0 or 1)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Add intercept
        if self.fit_intercept:
            X = np.column_stack([np.ones(n_samples), X])
            n_features += 1
        
        # Initialize weights
        self.coef_ = np.zeros(n_features)
        
        # Gradient descent
        for iteration in range(self.max_iter):
            # Predictions
            z = X @ self.coef_
            y_pred = self._sigmoid(z)
            
            # Gradient
            error = y_pred - y
            gradient = X.T @ error / n_samples
            
            # Add regularization
            if self.regularization > 0:
                gradient[1:] += self.regularization * self.coef_[1:] / n_samples
            
            # Update weights
            self.coef_ -= self.learning_rate * gradient
            
            # Check convergence
            if iteration % 100 == 0:
                loss = -np.mean(y * np.log(y_pred + 1e-10) + 
                               (1 - y) * np.log(1 - y_pred + 1e-10))
                if loss < 1e-6:
                    break
        
        # Separate intercept and coefficients
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0
        
        logger.info(f"[LogisticRegression] Fitted with {n_samples} samples")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Parameters
        ----------
        X : array-like
            Input data
            
        Returns
        -------
        probabilities : array, shape (n_samples, 2)
            Class probabilities [P(class=0), P(class=1)]
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.coef_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        z = X @ self.coef_ + self.intercept_
        prob_1 = self._sigmoid(z)
        prob_0 = 1 - prob_1
        
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels
        
        Parameters
        ----------
        X : array-like
            Input data
        threshold : float
            Classification threshold
            
        Returns
        -------
        predictions : array
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)


class DecisionTree:
    """
    Decision Tree Classifier/Regressor
    
    Simple implementation using recursive splitting
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 criterion: str = 'gini'):
        """
        Initialize decision tree
        
        Parameters
        ----------
        max_depth : int
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split
        criterion : str
            Splitting criterion ('gini' or 'entropy' for classification, 'mse' for regression)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None
        self.is_classification = None
    
    def _gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        proportions = counts / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy using information_theory module"""
        if len(y) == 0:
            return 0.0
        try:
            from ml_toolbox.textbook_concepts.information_theory import entropy
            counts = np.bincount(y.astype(int))
            proportions = counts[counts > 0] / len(y)
            return entropy(proportions, base=2.0)
        except ImportError:
            # Fallback to original implementation
            counts = np.bincount(y.astype(int))
            proportions = counts[counts > 0] / len(y)
            return -np.sum(proportions * np.log2(proportions))
    
    def _mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error"""
        if len(y) == 0:
            return 0.0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Optional[Dict]:
        """Find best split"""
        best_gain = -1
        best_split = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            # Get unique values for this feature
            values = np.unique(X[:, feature_idx])
            
            for value in values:
                # Split
                left_mask = X[:, feature_idx] <= value
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_split or \
                   np.sum(right_mask) < self.min_samples_split:
                    continue
                
                # Calculate gain
                if self.criterion == 'gini':
                    parent_impurity = self._gini(y)
                    left_impurity = self._gini(y[left_mask])
                    right_impurity = self._gini(y[right_mask])
                    n_left = np.sum(left_mask)
                    n_right = np.sum(right_mask)
                    n_total = len(y)
                    gain = parent_impurity - (n_left / n_total * left_impurity + 
                                             n_right / n_total * right_impurity)
                elif self.criterion == 'entropy':
                    # Use information gain from information_theory module
                    try:
                        from ml_toolbox.textbook_concepts.information_theory import information_gain
                        y_left = y[left_mask]
                        y_right = y[right_mask]
                        gain = information_gain(y, [y_left, y_right])
                    except (ImportError, Exception):
                        # Fallback to manual calculation
                        parent_impurity = self._entropy(y)
                        left_impurity = self._entropy(y[left_mask])
                        right_impurity = self._entropy(y[right_mask])
                        n_left = np.sum(left_mask)
                        n_right = np.sum(right_mask)
                        n_total = len(y)
                        gain = parent_impurity - (n_left / n_total * left_impurity + 
                                                 n_right / n_total * right_impurity)
                else:  # mse
                    parent_impurity = self._mse(y)
                    left_impurity = self._mse(y[left_mask])
                    right_impurity = self._mse(y[right_mask])
                    n_left = np.sum(left_mask)
                    n_right = np.sum(right_mask)
                    n_total = len(y)
                    gain = parent_impurity - (n_left / n_total * left_impurity + 
                                             n_right / n_total * right_impurity)
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'value': value,
                        'gain': gain
                    }
        
        return best_split
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict:
        """Build decision tree recursively"""
        # Check stopping conditions
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            # Leaf node
            if self.is_classification:
                counts = np.bincount(y.astype(int))
                return {'class': np.argmax(counts), 'is_leaf': True}
            else:
                return {'value': np.mean(y), 'is_leaf': True}
        
        # Find best split
        split = self._best_split(X, y)
        
        if split is None or split['gain'] <= 0:
            # Leaf node
            if self.is_classification:
                counts = np.bincount(y.astype(int))
                return {'class': np.argmax(counts), 'is_leaf': True}
            else:
                return {'value': np.mean(y), 'is_leaf': True}
        
        # Split data
        left_mask = X[:, split['feature_idx']] <= split['value']
        right_mask = ~left_mask
        
        # Build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature_idx': split['feature_idx'],
            'value': split['value'],
            'left': left_tree,
            'right': right_tree,
            'is_leaf': False
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit decision tree"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Determine if classification or regression
        if y.dtype == int or len(np.unique(y)) < 20:
            self.is_classification = True
            if self.criterion not in ['gini', 'entropy']:
                self.criterion = 'gini'
        else:
            self.is_classification = False
            self.criterion = 'mse'
        
        self.tree = self._build_tree(X, y)
        logger.info(f"[DecisionTree] Fitted {self.is_classification and 'classification' or 'regression'} tree")
    
    def _predict_sample(self, x: np.ndarray, node: Dict) -> Any:
        """Predict single sample"""
        if node['is_leaf']:
            return node.get('class', node.get('value'))
        
        if x[node['feature_idx']] <= node['value']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using decision tree"""
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if self.tree is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = [self._predict_sample(x, self.tree) for x in X]
        return np.array(predictions)


class SVM:
    """
    Support Vector Machine
    
    Simple implementation using gradient descent
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'linear',
                 learning_rate: float = 0.01, max_iter: int = 1000):
        """
        Initialize SVM
        
        Parameters
        ----------
        C : float
            Regularization parameter
        kernel : str
            Kernel type ('linear' or 'rbf')
        learning_rate : float
            Learning rate
        max_iter : int
            Maximum iterations
        """
        self.C = C
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.b = 0.0
        self.gamma = 1.0  # For RBF kernel
    
    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Linear kernel"""
        return X1 @ X2.T
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel"""
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
        
        # Compute pairwise squared distances
        dists = np.sum(X1 ** 2, axis=1)[:, np.newaxis] + \
                np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-self.gamma * dists)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit SVM model
        
        Parameters
        ----------
        X : array-like
            Training data
        y : array-like
            Target labels (-1 or 1)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        # Convert to -1, 1 labels
        y = np.where(y == 0, -1, y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize alphas
        self.alpha = np.random.rand(n_samples) * 0.1
        
        # Compute kernel matrix
        if self.kernel == 'linear':
            K = self._linear_kernel(X, X)
        else:
            K = self._rbf_kernel(X, X)
        
        # Gradient descent for dual problem
        for iteration in range(self.max_iter):
            # Compute gradient
            gradient = 1 - (y * (K @ (self.alpha * y)))
            
            # Update alphas
            self.alpha += self.learning_rate * gradient
            
            # Clip to [0, C]
            self.alpha = np.clip(self.alpha, 0, self.C)
        
        # Find support vectors
        sv_mask = self.alpha > 1e-5
        self.support_vectors_ = X[sv_mask]
        self.support_vector_labels_ = y[sv_mask]
        self.alpha = self.alpha[sv_mask]
        
        # Compute bias
        if len(self.alpha) > 0:
            if self.kernel == 'linear':
                self.b = np.mean(self.support_vector_labels_ - 
                               (self.alpha * self.support_vector_labels_) @ 
                               self._linear_kernel(self.support_vectors_, self.support_vectors_))
            else:
                self.b = np.mean(self.support_vector_labels_ - 
                               (self.alpha * self.support_vector_labels_) @ 
                               self._rbf_kernel(self.support_vectors_, self.support_vectors_))
        
        logger.info(f"[SVM] Fitted with {len(self.support_vectors_)} support vectors")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using SVM"""
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if self.support_vectors_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Compute kernel
        if self.kernel == 'linear':
            K = self._linear_kernel(X, self.support_vectors_)
        else:
            K = self._rbf_kernel(X, self.support_vectors_)
        
        # Predictions
        predictions = (self.alpha * self.support_vector_labels_) @ K.T + self.b
        
        return np.sign(predictions)
