"""
Tests for Knuth ML Integrations
Test practical ML applications of Knuth algorithms
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    pytestmark = pytest.mark.skip("sklearn not available")

try:
    from knuth_ml_integrations import (
        KnuthFeatureSelector,
        KnuthHyperparameterSearch,
        KnuthKnowledgeGraph,
        KnuthDataSampling,
        KnuthDataPreprocessing,
        KnuthMLIntegration
    )
    KNUTH_ML_AVAILABLE = True
except ImportError:
    KNUTH_ML_AVAILABLE = False
    pytestmark = pytest.mark.skip("Knuth ML integrations not available")


class TestKnuthFeatureSelector:
    """Tests for feature selection with Knuth algorithms"""
    
    def test_forward_selection(self):
        """Test forward selection"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        selector = KnuthFeatureSelector(random_seed=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        result = selector.forward_selection_knuth(X, y, model, k=5)
        
        assert 'selected_features' in result
        assert 'score' in result
        assert len(result['selected_features']) == 5
        assert result['score'] >= 0


class TestKnuthHyperparameterSearch:
    """Tests for hyperparameter search"""
    
    def test_grid_search(self):
        """Test grid search"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        search = KnuthHyperparameterSearch(random_seed=42)
        model = RandomForestClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [5, 10]
        }
        
        result = search.grid_search_knuth(model, X, y, param_grid, cv=3)
        
        assert 'best_params' in result
        assert 'best_score' in result
        assert result['best_params'] is not None


class TestKnuthKnowledgeGraph:
    """Tests for knowledge graph operations"""
    
    def test_build_graph(self):
        """Test graph building"""
        kg = KnuthKnowledgeGraph()
        relationships = [
            ('A', 'B'),
            ('B', 'C'),
            ('A', 'C')
        ]
        
        kg.build_graph_from_relationships(relationships)
        
        assert 'A' in kg.graph
        assert 'B' in kg.graph['A']
    
    def test_find_related_concepts(self):
        """Test finding related concepts"""
        kg = KnuthKnowledgeGraph()
        relationships = [
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'D')
        ]
        kg.build_graph_from_relationships(relationships)
        
        related = kg.find_related_concepts('A', max_depth=3, method='bfs')
        
        assert len(related) > 0
        assert 'A' in related


class TestKnuthDataSampling:
    """Tests for data sampling"""
    
    def test_stratified_sample(self):
        """Test stratified sampling"""
        X, y = make_classification(n_samples=200, n_features=10, n_classes=3, random_state=42)
        
        sampler = KnuthDataSampling(seed=42)
        X_sample, y_sample = sampler.stratified_sample(X, y, n_samples=100, stratify=True)
        
        assert len(X_sample) == 100
        assert len(y_sample) == 100
        assert len(np.unique(y_sample)) <= len(np.unique(y))
    
    def test_bootstrap_sample(self):
        """Test bootstrap sampling"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        sampler = KnuthDataSampling(seed=42)
        X_boot, y_boot = sampler.bootstrap_sample(X, y, n_samples=50)
        
        assert len(X_boot) == 50
        assert len(y_boot) == 50
    
    def test_shuffle_data(self):
        """Test data shuffling"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        original_y = y.copy()
        
        sampler = KnuthDataSampling(seed=42)
        X_shuffled, y_shuffled = sampler.shuffle_data(X, y)
        
        assert len(X_shuffled) == len(X)
        assert len(y_shuffled) == len(y)
        # Should be different order (with high probability)
        assert not np.array_equal(original_y, y_shuffled) or len(y) <= 1


class TestKnuthDataPreprocessing:
    """Tests for data preprocessing"""
    
    def test_sort_by_importance(self):
        """Test sorting by feature importance"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        feature_importance = np.random.rand(10)
        
        preprocessor = KnuthDataPreprocessing()
        sorted_indices = preprocessor.sort_by_feature_importance(
            X, feature_importance, descending=True
        )
        
        assert len(sorted_indices) == 10
        # Should be sorted by importance
        sorted_importance = feature_importance[sorted_indices]
        assert all(sorted_importance[i] >= sorted_importance[i+1] for i in range(9))
    
    def test_find_similar_samples(self):
        """Test finding similar samples"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        preprocessor = KnuthDataPreprocessing()
        query = X[0]
        similar_indices = preprocessor.find_similar_samples(X, query, k=5)
        
        assert len(similar_indices) == 5
        assert 0 in similar_indices  # Query itself should be most similar


class TestKnuthMLIntegration:
    """Tests for integrated Knuth ML"""
    
    def test_integration(self):
        """Test integrated interface"""
        knuth_ml = KnuthMLIntegration(seed=42)
        
        # Test components
        assert knuth_ml.feature_selector is not None
        assert knuth_ml.hyperparameter_search is not None
        assert knuth_ml.knowledge_graph is not None
        assert knuth_ml.data_sampling is not None
        assert knuth_ml.data_preprocessing is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
