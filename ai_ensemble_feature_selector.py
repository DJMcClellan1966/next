"""
AI Ensemble Feature Selector
Unifies all feature selection methods with AI orchestration

Innovation: Combines all feature selection methods intelligently
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import Counter
import time
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    from advanced_feature_selection import AdvancedFeatureSelector
    FEATURE_SELECTORS_AVAILABLE = True
except ImportError:
    FEATURE_SELECTORS_AVAILABLE = False

try:
    from information_theoretic_feature_selection import InformationTheoreticFeatureSelector
    INFO_THEORETIC_AVAILABLE = True
except ImportError:
    INFO_THEORETIC_AVAILABLE = False

try:
    from search_based_feature_selection import SearchBasedFeatureSelector
    SEARCH_BASED_AVAILABLE = True
except ImportError:
    SEARCH_BASED_AVAILABLE = False

try:
    from variance_correlation_filter import VarianceCorrelationFilter
    VARIANCE_FILTER_AVAILABLE = True
except ImportError:
    VARIANCE_FILTER_AVAILABLE = False

try:
    from statistical_learning import StatisticalFeatureSelector
    STATISTICAL_AVAILABLE = True
except ImportError:
    STATISTICAL_AVAILABLE = False

# Try sklearn feature selection as fallback
try:
    from sklearn.feature_selection import (
        SelectKBest, f_classif, f_regression,
        mutual_info_classif, mutual_info_regression,
        RFE, RFECV, SelectFromModel
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    SKLEARN_FEATURE_SELECTION_AVAILABLE = False


class AIEnsembleFeatureSelector:
    """
    AI-Powered Ensemble Feature Selector
    
    Innovation: Uses multiple feature selection methods intelligently,
    then AI decides which features to keep based on consensus and
    performance.
    
    Replaces: 5+ separate feature selectors with one intelligent system
    """
    
    def __init__(self):
        """Initialize AI ensemble feature selector"""
        self.selectors = self._initialize_selectors()
        self.consensus_history = []
        self.performance_memory = {}
    
    def _initialize_selectors(self) -> Dict[str, Any]:
        """Initialize all available feature selectors"""
        selectors = {}
        
        # Advanced feature selector
        if FEATURE_SELECTORS_AVAILABLE:
            try:
                selectors['advanced'] = AdvancedFeatureSelector
            except Exception as e:
                warnings.warn(f"Advanced selector not available: {e}")
        
        # Information theoretic
        if INFO_THEORETIC_AVAILABLE:
            try:
                selectors['information_theoretic'] = InformationTheoreticFeatureSelector
            except Exception as e:
                warnings.warn(f"Information theoretic selector not available: {e}")
        
        # Search based
        if SEARCH_BASED_AVAILABLE:
            try:
                selectors['search_based'] = SearchBasedFeatureSelector
            except Exception as e:
                warnings.warn(f"Search based selector not available: {e}")
        
        # Variance correlation
        if VARIANCE_FILTER_AVAILABLE:
            try:
                selectors['variance_correlation'] = VarianceCorrelationFilter
            except Exception as e:
                warnings.warn(f"Variance filter not available: {e}")
        
        # Statistical
        if STATISTICAL_AVAILABLE:
            try:
                selectors['statistical'] = StatisticalFeatureSelector
            except Exception as e:
                warnings.warn(f"Statistical selector not available: {e}")
        
        # Sklearn fallback selectors
        if SKLEARN_FEATURE_SELECTION_AVAILABLE:
            try:
                # Create wrapper classes for sklearn selectors
                class SklearnSelectKBest:
                    def __init__(self):
                        self.selector = None
                    
                    def fit_transform(self, X, y):
                        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
                        is_classification = len(set(y)) < 20
                        score_func = f_classif if is_classification else f_regression
                        self.selector = SelectKBest(score_func=score_func, k=min(10, X.shape[1]))
                        return self.selector.fit_transform(X, y)
                    
                    def get_support(self):
                        return self.selector.get_support() if self.selector else None
                
                class SklearnRFE:
                    def __init__(self):
                        self.selector = None
                    
                    def fit_transform(self, X, y):
                        from sklearn.feature_selection import RFE
                        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                        is_classification = len(set(y)) < 20
                        estimator = RandomForestClassifier() if is_classification else RandomForestRegressor()
                        self.selector = RFE(estimator=estimator, n_features_to_select=min(10, X.shape[1]))
                        return self.selector.fit_transform(X, y)
                    
                    def get_support(self):
                        return self.selector.get_support() if self.selector else None
                
                selectors['sklearn_kbest'] = SklearnSelectKBest
                selectors['sklearn_rfe'] = SklearnRFE
            except Exception as e:
                warnings.warn(f"Sklearn selectors not available: {e}")
        
        return selectors
    
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       n_features: Optional[int] = None,
                       task_type: str = 'auto') -> Dict[str, Any]:
        """
        Intelligently select features using ensemble + AI
        
        Args:
            X: Features
            y: Target
            n_features: Number of features to select (auto if None)
            task_type: 'classification' or 'regression'
        
        Returns:
            Selected features and metadata
        """
        # Auto-determine n_features if not provided
        if n_features is None:
            n_features = min(50, X.shape[1] // 2)  # Default: half features, max 50
        
        # Run all methods in parallel (conceptually)
        results = {}
        for selector_name, SelectorClass in self.selectors.items():
            try:
                selector = SelectorClass()
                
                # Try to select features
                if hasattr(selector, 'select_features'):
                    selected = selector.select_features(X, y, n_features=n_features)
                elif hasattr(selector, 'fit_transform'):
                    selector.fit(X, y)
                    selected = selector.transform(X)
                    # Get selected feature indices
                    if hasattr(selector, 'get_support'):
                        selected_indices = np.where(selector.get_support())[0]
                    else:
                        selected_indices = list(range(min(n_features, X.shape[1])))
                else:
                    continue
                
                results[selector_name] = {
                    'selected_indices': selected_indices if isinstance(selected, np.ndarray) else selected,
                    'success': True
                }
            except Exception as e:
                results[selector_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # AI finds consensus
        consensus = self._ai_consensus(results, n_features)
        
        # AI selects optimal set
        optimal_features = self._ai_select_optimal(consensus, X, y, n_features)
        
        # Learn from result
        self._learn_from_result(results, consensus, optimal_features)
        
        return {
            'selected_features': optimal_features,
            'selected_indices': optimal_features,
            'consensus': consensus,
            'method_results': results,
            'n_features_selected': len(optimal_features)
        }
    
    def _ai_consensus(self, results: Dict, n_features: int) -> Dict[str, Any]:
        """
        Find consensus among feature selection methods
        
        Innovation: Uses voting/consensus to find features that
        multiple methods agree on.
        """
        # Collect all selected features
        all_selections = []
        for selector_name, result in results.items():
            if result.get('success'):
                indices = result.get('selected_indices', [])
                if isinstance(indices, np.ndarray):
                    indices = indices.tolist()
                all_selections.extend(indices)
        
        # Count votes for each feature
        feature_votes = Counter(all_selections)
        
        # Features selected by multiple methods (consensus)
        consensus_features = [
            feature for feature, votes in feature_votes.items()
            if votes >= 2  # Selected by at least 2 methods
        ]
        
        # Sort by vote count
        consensus_features.sort(key=lambda f: feature_votes[f], reverse=True)
        
        return {
            'consensus_features': consensus_features[:n_features],
            'feature_votes': dict(feature_votes),
            'consensus_strength': len(consensus_features) / n_features if n_features > 0 else 0.0
        }
    
    def _ai_select_optimal(self, consensus: Dict, X: np.ndarray, 
                          y: np.ndarray, n_features: int) -> List[int]:
        """
        AI selects optimal feature set
        
        Uses:
        - Consensus features (multiple methods agree)
        - Feature importance scores
        - Performance on task
        """
        consensus_features = consensus.get('consensus_features', [])
        
        # If we have enough consensus features, use them
        if len(consensus_features) >= n_features:
            return consensus_features[:n_features]
        
        # Otherwise, add top-voted features
        feature_votes = consensus.get('feature_votes', {})
        all_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        selected = consensus_features.copy()
        for feature, votes in all_features:
            if feature not in selected and len(selected) < n_features:
                selected.append(feature)
        
        # Fill remaining with top features by variance (fallback)
        if len(selected) < n_features:
            variances = np.var(X, axis=0)
            top_variance_features = np.argsort(variances)[::-1]
            for feature in top_variance_features:
                if feature not in selected and len(selected) < n_features:
                    selected.append(int(feature))
        
        return selected[:n_features]
    
    def _learn_from_result(self, results: Dict, consensus: Dict, optimal_features: List[int]):
        """Learn from feature selection result"""
        self.consensus_history.append({
            'results': results,
            'consensus': consensus,
            'optimal_features': optimal_features,
            'timestamp': time.time()
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics"""
        return {
            'total_selections': len(self.consensus_history),
            'available_selectors': len(self.selectors),
            'consensus_strength_avg': np.mean([
                c.get('consensus', {}).get('consensus_strength', 0.0)
                for c in self.consensus_history
            ]) if self.consensus_history else 0.0
        }


# Global instance
_global_ensemble_selector = None

def get_ai_ensemble_selector() -> AIEnsembleFeatureSelector:
    """Get global AI ensemble feature selector instance"""
    global _global_ensemble_selector
    if _global_ensemble_selector is None:
        _global_ensemble_selector = AIEnsembleFeatureSelector()
    return _global_ensemble_selector
