"""
Russell/Norvig Search-Based Feature Selection
Use search algorithms (A*, Beam Search) for optimal feature selection

Methods:
- A* search for feature selection
- Beam search for feature selection
- Greedy best-first search
- Uniform cost search
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Set
import numpy as np
from collections import deque
import heapq
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class SearchBasedFeatureSelector:
    """
    Search-based feature selection using Russell/Norvig search algorithms
    
    Uses A*, Beam Search, or Greedy Best-First to find optimal feature subsets
    """
    
    def __init__(
        self,
        estimator: Any,
        method: str = 'astar',
        scoring: Optional[str] = None,
        cv: int = 5,
        max_features: Optional[int] = None,
        beam_width: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            estimator: Base estimator for evaluation
            method: Search method ('astar', 'beam', 'greedy', 'ucs')
            scoring: Scoring metric
            cv: Cross-validation folds
            max_features: Maximum number of features to select
            beam_width: Beam width for beam search
            random_state: Random seed
        """
        self.estimator = estimator
        self.method = method
        self.scoring = scoring
        self.cv = cv
        self.max_features = max_features
        self.beam_width = beam_width
        self.random_state = random_state
        self.selected_features_ = None
        self.search_history_ = []
    
    def _evaluate_feature_set(self, X: np.ndarray, y: np.ndarray, features: Set[int]) -> float:
        """Evaluate a feature set using cross-validation"""
        if not SKLEARN_AVAILABLE:
            return 0.0
        
        if len(features) == 0:
            return 0.0
        
        # Select features
        feature_list = sorted(list(features))
        X_selected = X[:, feature_list]
        
        # Determine scoring
        if self.scoring is None:
            self.scoring = 'accuracy' if len(np.unique(y)) < 10 else 'r2'
        
        # CV splitter
        if len(np.unique(y)) < 10:  # Classification
            cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # Cross-validation score
        scores = cross_val_score(
            self.estimator, X_selected, y,
            cv=cv_splitter,
            scoring=self.scoring,
            n_jobs=-1
        )
        
        return np.mean(scores)
    
    def _heuristic(self, current_features: Set[int], all_features: Set[int], X: np.ndarray, y: np.ndarray) -> float:
        """
        Heuristic function for A* search
        
        Estimates remaining potential improvement
        """
        remaining = all_features - current_features
        
        if len(remaining) == 0:
            return 0.0
        
        # Simple heuristic: assume remaining features add average value
        # More sophisticated: could use mutual information or correlation
        if len(current_features) > 0:
            current_score = self._evaluate_feature_set(X, y, current_features)
            # Estimate: remaining features could add up to 10% improvement
            return 0.1 * len(remaining) / len(all_features)
        else:
            return 0.05 * len(remaining) / len(all_features)
    
    def astar_search(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        A* search for feature selection
        
        Finds optimal feature subset using A* algorithm
        """
        n_features = X.shape[1]
        all_features = set(range(n_features))
        
        # Priority queue: (f_score, g_score, features_set)
        # f_score = g_score + heuristic
        # g_score = actual cost (negative CV score)
        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, frozenset()))
        
        # Track best solution
        best_score = -np.inf
        best_features = set()
        visited = set()
        
        max_iterations = 1000  # Prevent infinite loops
        iteration = 0
        
        while open_set and iteration < max_iterations:
            iteration += 1
            f_score, g_score, current_features = heapq.heappop(open_set)
            
            # Skip if already visited
            if current_features in visited:
                continue
            
            visited.add(current_features)
            
            # Evaluate current feature set
            if len(current_features) > 0:
                score = self._evaluate_feature_set(X, y, current_features)
                
                if score > best_score:
                    best_score = score
                    best_features = current_features.copy()
                    self.search_history_.append({
                        'iteration': iteration,
                        'features': sorted(list(current_features)),
                        'score': score
                    })
            
            # Check if we should stop
            if self.max_features and len(current_features) >= self.max_features:
                continue
            
            # Generate neighbors (add one feature at a time)
            remaining = all_features - current_features
            for feature in remaining:
                new_features = current_features | {feature}
                
                if new_features in visited:
                    continue
                
                # Calculate g_score (negative CV score)
                new_score = self._evaluate_feature_set(X, y, new_features)
                new_g_score = -new_score
                
                # Calculate heuristic
                h_score = self._heuristic(new_features, all_features, X, y)
                
                # f_score = g_score + h_score
                new_f_score = new_g_score + h_score
                
                heapq.heappush(open_set, (new_f_score, new_g_score, frozenset(new_features)))
        
        self.selected_features_ = np.array(sorted(list(best_features)))
        
        return {
            'selected_features': self.selected_features_.tolist(),
            'best_score': float(best_score),
            'n_iterations': iteration,
            'search_history': self.search_history_
        }
    
    def beam_search(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Beam search for feature selection
        
        Maintains top-k candidates at each step
        """
        n_features = X.shape[1]
        all_features = set(range(n_features))
        
        # Start with empty set
        beam = [(set(), 0.0)]  # (features, score)
        
        best_score = -np.inf
        best_features = set()
        iteration = 0
        max_iterations = n_features
        
        while beam and iteration < max_iterations:
            iteration += 1
            next_beam = []
            
            for current_features, current_score in beam:
                # Check if we should stop
                if self.max_features and len(current_features) >= self.max_features:
                    continue
                
                # Generate candidates (add one feature)
                remaining = all_features - current_features
                for feature in remaining:
                    new_features = current_features | {feature}
                    new_score = self._evaluate_feature_set(X, y, new_features)
                    
                    next_beam.append((new_features, new_score))
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_features = new_features.copy()
                        self.search_history_.append({
                            'iteration': iteration,
                            'features': sorted(list(new_features)),
                            'score': new_score
                        })
            
            # Keep top beam_width candidates
            next_beam.sort(key=lambda x: x[1], reverse=True)
            beam = next_beam[:self.beam_width]
        
        self.selected_features_ = np.array(sorted(list(best_features)))
        
        return {
            'selected_features': self.selected_features_.tolist(),
            'best_score': float(best_score),
            'n_iterations': iteration,
            'search_history': self.search_history_
        }
    
    def greedy_best_first(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Greedy best-first search for feature selection
        
        Always expands the most promising node
        """
        n_features = X.shape[1]
        all_features = set(range(n_features))
        
        current_features = set()
        best_score = -np.inf
        best_features = set()
        iteration = 0
        max_iterations = n_features
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check if we should stop
            if self.max_features and len(current_features) >= self.max_features:
                break
            
            # Find best feature to add
            remaining = all_features - current_features
            if len(remaining) == 0:
                break
            
            best_candidate = None
            best_candidate_score = -np.inf
            
            for feature in remaining:
                candidate_features = current_features | {feature}
                score = self._evaluate_feature_set(X, y, candidate_features)
                
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = feature
            
            if best_candidate is None:
                break
            
            # Add best feature
            current_features.add(best_candidate)
            
            if best_candidate_score > best_score:
                best_score = best_candidate_score
                best_features = current_features.copy()
                self.search_history_.append({
                    'iteration': iteration,
                    'features': sorted(list(current_features)),
                    'score': best_candidate_score
                })
        
        self.selected_features_ = np.array(sorted(list(best_features)))
        
        return {
            'selected_features': self.selected_features_.tolist(),
            'best_score': float(best_score),
            'n_iterations': iteration,
            'search_history': self.search_history_
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SearchBasedFeatureSelector':
        """
        Fit feature selector
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Run search based on method
        if self.method == 'astar':
            result = self.astar_search(X, y)
        elif self.method == 'beam':
            result = self.beam_search(X, y)
        elif self.method == 'greedy':
            result = self.greedy_best_first(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform to selected features
        
        Args:
            X: Features
            
        Returns:
            Selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Must fit before transform")
        
        X = np.asarray(X)
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X, y).transform(X)
    
    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features"""
        if self.selected_features_ is None:
            raise ValueError("Must fit before get_support")
        
        if indices:
            return self.selected_features_
        else:
            mask = np.zeros(self.selected_features_.max() + 1, dtype=bool)
            mask[self.selected_features_] = True
            return mask
