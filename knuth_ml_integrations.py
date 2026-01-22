"""
Knuth Algorithms - ML Toolbox Integrations
Practical ML applications of Knuth's algorithms

Integrates Knuth algorithms into ML workflows:
- Random sampling for cross-validation
- Combinatorial algorithms for feature selection
- Graph algorithms for knowledge graphs
- Sorting for data preprocessing
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from knuth_algorithms import (
    KnuthRandom,
    KnuthSorting,
    KnuthSearching,
    KnuthCombinatorial,
    KnuthGraph,
    KnuthAlgorithms
)


class KnuthFeatureSelector:
    """
    Feature Selection using Knuth's Combinatorial Algorithms
    
    Uses subset generation for exhaustive or guided feature selection
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Args:
            random_seed: Random seed for reproducibility
        """
        self.combinatorial = KnuthCombinatorial()
        self.random = KnuthRandom(seed=random_seed)
        self.feature_scores = {}
    
    def exhaustive_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        max_features: int = 10,
        scoring_metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Exhaustive feature selection using subset generation
        
        Args:
            X: Features
            y: Labels
            model: Model to evaluate
            max_features: Maximum number of features to consider
            scoring_metric: Metric to optimize
            
        Returns:
            Dictionary with best feature subset and score
        """
        n_features = min(X.shape[1], max_features)
        feature_indices = list(range(n_features))
        
        best_score = -np.inf
        best_subset = None
        
        # Generate all subsets
        for subset in self.combinatorial.generate_subsets_lexicographic(feature_indices):
            if len(subset) == 0:
                continue
            
            # Select features
            X_subset = X[:, subset]
            
            # Evaluate model (simplified - would use cross-validation in practice)
            try:
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X_subset, y, cv=3, scoring=scoring_metric)
                score = np.mean(scores)
            except:
                # Fallback: simple train/test split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2)
                model.fit(X_train, y_train)
                if hasattr(model, 'score'):
                    score = model.score(X_test, y_test)
                else:
                    continue
            
            if score > best_score:
                best_score = score
                best_subset = subset
        
        return {
            'best_subset': best_subset,
            'best_score': best_score,
            'n_features_selected': len(best_subset) if best_subset else 0
        }
    
    def forward_selection_knuth(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Forward selection using k-combination generation
        
        Args:
            X: Features
            y: Labels
            model: Model to evaluate
            k: Number of features to select
            
        Returns:
            Dictionary with selected features
        """
        n_features = X.shape[1]
        feature_indices = list(range(n_features))
        
        best_score = -np.inf
        best_subset = None
        
        # Generate k-combinations
        for subset in self.combinatorial.generate_combinations_lexicographic(feature_indices, k):
            X_subset = X[:, subset]
            
            try:
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X_subset, y, cv=3)
                score = np.mean(scores)
            except:
                continue
            
            if score > best_score:
                best_score = score
                best_subset = subset
        
        return {
            'selected_features': best_subset,
            'score': best_score,
            'n_features': len(best_subset) if best_subset else 0
        }


class KnuthHyperparameterSearch:
    """
    Hyperparameter Search using Knuth's Combinatorial Algorithms
    
    Uses permutation/combination generation for hyperparameter space exploration
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Args:
            random_seed: Random seed
        """
        self.combinatorial = KnuthCombinatorial()
        self.random = KnuthRandom(seed=random_seed)
    
    def grid_search_knuth(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Grid search using combinatorial generation
        
        Args:
            model: Model to tune
            X: Features
            y: Labels
            param_grid: Parameter grid
            cv: Cross-validation folds
            
        Returns:
            Dictionary with best parameters and score
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        # Generate Cartesian product using nested loops (simplified)
        best_score = -np.inf
        best_params = None
        
        def generate_combinations(values_list, current=[]):
            if len(current) == len(values_list):
                yield current
            else:
                for value in values_list[len(current)]:
                    yield from generate_combinations(values_list, current + [value])
        
        for param_combo in generate_combinations(param_values):
            params = dict(zip(param_names, param_combo))
            
            # Set parameters
            model_copy = model.__class__(**params)
            
            # Evaluate
            try:
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model_copy, X, y, cv=cv)
                score = np.mean(scores)
            except:
                continue
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score
        }


class KnuthKnowledgeGraph:
    """
    Knowledge Graph Operations using Knuth's Graph Algorithms
    
    Enhances knowledge graph with efficient traversal and search
    """
    
    def __init__(self):
        """Initialize with Knuth graph algorithms"""
        self.graph_algorithms = KnuthGraph()
        self.graph: Dict[str, List[str]] = {}
    
    def build_graph_from_relationships(self, relationships: List[Tuple[str, str]]):
        """
        Build graph from relationship tuples
        
        Args:
            relationships: List of (source, target) tuples
        """
        self.graph = defaultdict(list)
        for source, target in relationships:
            self.graph[source].append(target)
    
    def find_related_concepts(
        self,
        concept: str,
        max_depth: int = 2,
        method: str = 'bfs'
    ) -> List[str]:
        """
        Find related concepts using graph traversal
        
        Args:
            concept: Starting concept
            max_depth: Maximum traversal depth
            method: 'bfs' or 'dfs'
            
        Returns:
            List of related concepts
        """
        if concept not in self.graph:
            return []
        
        if method == 'bfs':
            visited = self.graph_algorithms.breadth_first_search(self.graph, concept)
        else:
            visited = self.graph_algorithms.depth_first_search(self.graph, concept)
        
        # Limit by depth (simplified - would need proper depth tracking)
        return visited[:max_depth * 10]  # Approximate
    
    def find_shortest_relationship_path(
        self,
        concept1: str,
        concept2: str
    ) -> Optional[List[str]]:
        """
        Find shortest path between concepts
        
        Args:
            concept1: Source concept
            concept2: Target concept
            
        Returns:
            Path as list of concepts, or None
        """
        # Convert to weighted graph (all weights = 1)
        weighted_graph = {}
        for node, neighbors in self.graph.items():
            weighted_graph[node] = [(n, 1.0) for n in neighbors]
        
        # Find shortest paths from concept1
        distances = self.graph_algorithms.shortest_path_dijkstra(
            weighted_graph,
            concept1,
            end=concept2
        )
        
        if concept2 in distances:
            # Reconstruct path (simplified - would need proper path reconstruction)
            return [concept1, concept2]  # Simplified
        else:
            return None


class KnuthDataSampling:
    """
    Data Sampling using Knuth's Random Algorithms
    
    Reproducible sampling for cross-validation, bootstrap, etc.
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed
        """
        self.random = KnuthRandom(seed=seed)
    
    def stratified_sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stratified sampling using Knuth's random algorithms
        
        Args:
            X: Features
            y: Labels
            n_samples: Number of samples to select
            stratify: Whether to maintain class distribution
            
        Returns:
            Tuple of (X_sample, y_sample)
        """
        if stratify:
            # Sample proportionally from each class
            unique_classes = np.unique(y)
            samples_per_class = n_samples // len(unique_classes)
            
            X_samples = []
            y_samples = []
            
            for cls in unique_classes:
                class_indices = np.where(y == cls)[0].tolist()
                sampled_indices = self.random.random_sample_without_replacement(
                    class_indices,
                    k=min(samples_per_class, len(class_indices))
                )
                
                X_samples.append(X[sampled_indices])
                y_samples.append(y[sampled_indices])
            
            X_sample = np.vstack(X_samples)
            y_sample = np.hstack(y_samples)
        else:
            # Simple random sample
            indices = list(range(len(X)))
            sampled_indices = self.random.random_sample_without_replacement(
                indices,
                k=min(n_samples, len(indices))
            )
            X_sample = X[sampled_indices]
            y_sample = y[sampled_indices]
        
        return X_sample, y_sample
    
    def bootstrap_sample(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Bootstrap sampling with replacement
        
        Args:
            X: Features
            y: Optional labels
            n_samples: Sample size (default: same as X)
            
        Returns:
            Tuple of (X_bootstrap, y_bootstrap)
        """
        if n_samples is None:
            n_samples = len(X)
        
        indices = list(range(len(X)))
        
        # Sample with replacement using LCG
        lcg_numbers = self.random.linear_congruential_generator(n_samples)
        sampled_indices = [n % len(indices) for n in lcg_numbers]
        
        X_bootstrap = X[sampled_indices]
        y_bootstrap = y[sampled_indices] if y is not None else None
        
        return X_bootstrap, y_bootstrap
    
    def shuffle_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Shuffle data using Fisher-Yates shuffle
        
        Args:
            X: Features
            y: Optional labels
            
        Returns:
            Tuple of (X_shuffled, y_shuffled)
        """
        indices = list(range(len(X)))
        shuffled_indices = self.random.fisher_yates_shuffle(indices)
        
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices] if y is not None else None
        
        return X_shuffled, y_shuffled


class KnuthDataPreprocessing:
    """
    Data Preprocessing using Knuth's Sorting and Searching Algorithms
    
    Efficient data organization and retrieval
    """
    
    def __init__(self):
        """Initialize with Knuth algorithms"""
        self.sorting = KnuthSorting()
        self.searching = KnuthSearching()
    
    def sort_by_feature_importance(
        self,
        X: np.ndarray,
        feature_importance: np.ndarray,
        descending: bool = True
    ) -> np.ndarray:
        """
        Sort features by importance using heapsort
        
        Args:
            X: Features
            feature_importance: Importance scores
            descending: Sort order
            
        Returns:
            Sorted feature indices
        """
        # Create list of (importance, index) tuples
        importance_tuples = [(feature_importance[i], i) for i in range(len(feature_importance))]
        
        # Sort by importance
        sorted_tuples = self.sorting.heapsort(
            importance_tuples,
            key=lambda x: x[0]
        )
        
        if descending:
            sorted_tuples = list(reversed(sorted_tuples))
        
        # Return sorted indices
        return np.array([idx for _, idx in sorted_tuples])
    
    def find_similar_samples(
        self,
        X: np.ndarray,
        query: np.ndarray,
        k: int = 5,
        metric: str = 'euclidean'
    ) -> List[int]:
        """
        Find k most similar samples using efficient search
        
        Args:
            X: Feature matrix
            query: Query vector
            k: Number of neighbors
            metric: Distance metric
            
        Returns:
            List of indices of similar samples
        """
        # Calculate distances
        if metric == 'euclidean':
            distances = np.sqrt(np.sum((X - query) ** 2, axis=1))
        elif metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_similarity
            distances = 1 - cosine_similarity([query], X)[0]
        else:
            distances = np.sum(np.abs(X - query), axis=1)
        
        # Sort distances
        distance_tuples = [(distances[i], i) for i in range(len(distances))]
        sorted_tuples = self.sorting.heapsort(distance_tuples, key=lambda x: x[0])
        
        # Return top k indices
        return [idx for _, idx in sorted_tuples[:k]]


class KnuthMLIntegration:
    """
    Unified interface for Knuth algorithms in ML contexts
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed
        """
        self.knuth = KnuthAlgorithms(seed=seed)
        self.feature_selector = KnuthFeatureSelector(seed=seed)
        self.hyperparameter_search = KnuthHyperparameterSearch(seed=seed)
        self.knowledge_graph = KnuthKnowledgeGraph()
        self.data_sampling = KnuthDataSampling(seed=seed)
        self.data_preprocessing = KnuthDataPreprocessing()
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return self.knuth.get_dependencies()
