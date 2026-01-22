"""
Knuth Algorithms - ML Examples
Practical examples of using Knuth algorithms in ML workflows
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available")
    exit(1)

try:
    from knuth_ml_integrations import (
        KnuthFeatureSelector,
        KnuthHyperparameterSearch,
        KnuthKnowledgeGraph,
        KnuthDataSampling,
        KnuthDataPreprocessing,
        KnuthMLIntegration
    )
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    print("Warning: Knuth ML integrations not available")
    exit(1)


def example_feature_selection():
    """Example: Feature selection using Knuth combinatorial algorithms"""
    print("=" * 60)
    print("Example 1: Feature Selection with Knuth Algorithms")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10, random_state=42)
    
    # Initialize feature selector
    selector = KnuthFeatureSelector(random_seed=42)
    
    # Forward selection (k-combinations)
    print("\nForward selection (selecting 5 features)...")
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    result = selector.forward_selection_knuth(X, y, model, k=5)
    
    print(f"Selected features: {result['selected_features']}")
    print(f"Score: {result['score']:.4f}")
    
    print("\n✓ Feature selection example complete\n")


def example_hyperparameter_search():
    """Example: Hyperparameter search using Knuth algorithms"""
    print("=" * 60)
    print("Example 2: Hyperparameter Search with Knuth Algorithms")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    
    # Initialize hyperparameter search
    search = KnuthHyperparameterSearch(random_seed=42)
    
    # Grid search
    print("\nGrid search...")
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [10, 20],
        'max_depth': [5, 10]
    }
    
    result = search.grid_search_knuth(model, X, y, param_grid, cv=3)
    
    print(f"Best parameters: {result['best_params']}")
    print(f"Best score: {result['best_score']:.4f}")
    
    print("\n✓ Hyperparameter search example complete\n")


def example_knowledge_graph():
    """Example: Knowledge graph operations using Knuth graph algorithms"""
    print("=" * 60)
    print("Example 3: Knowledge Graph with Knuth Algorithms")
    print("=" * 60)
    
    # Build knowledge graph
    kg = KnuthKnowledgeGraph()
    
    # Define relationships
    relationships = [
        ('machine_learning', 'neural_networks'),
        ('machine_learning', 'decision_trees'),
        ('neural_networks', 'deep_learning'),
        ('decision_trees', 'random_forest'),
        ('deep_learning', 'transformer'),
        ('transformer', 'attention_mechanism')
    ]
    
    kg.build_graph_from_relationships(relationships)
    
    # Find related concepts
    print("\nFinding concepts related to 'machine_learning' (BFS)...")
    related = kg.find_related_concepts('machine_learning', max_depth=3, method='bfs')
    print(f"Related concepts: {related}")
    
    # Find shortest path
    print("\nFinding shortest path from 'machine_learning' to 'attention_mechanism'...")
    path = kg.find_shortest_relationship_path('machine_learning', 'attention_mechanism')
    print(f"Path: {path}")
    
    print("\n✓ Knowledge graph example complete\n")


def example_data_sampling():
    """Example: Data sampling using Knuth random algorithms"""
    print("=" * 60)
    print("Example 4: Data Sampling with Knuth Algorithms")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, random_state=42)
    
    # Initialize sampler
    sampler = KnuthDataSampling(seed=42)
    
    # Stratified sampling
    print("\nStratified sampling (100 samples)...")
    X_sample, y_sample = sampler.stratified_sample(X, y, n_samples=100, stratify=True)
    print(f"Sampled shape: {X_sample.shape}")
    print(f"Class distribution: {np.bincount(y_sample)}")
    
    # Bootstrap sampling
    print("\nBootstrap sampling...")
    X_boot, y_boot = sampler.bootstrap_sample(X, y, n_samples=500)
    print(f"Bootstrap shape: {X_boot.shape}")
    
    # Shuffle data
    print("\nShuffling data (Fisher-Yates)...")
    X_shuffled, y_shuffled = sampler.shuffle_data(X, y)
    print(f"Shuffled - first 5 labels: {y_shuffled[:5]}")
    
    print("\n✓ Data sampling example complete\n")


def example_data_preprocessing():
    """Example: Data preprocessing using Knuth sorting/searching"""
    print("=" * 60)
    print("Example 5: Data Preprocessing with Knuth Algorithms")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    
    # Calculate feature importance
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    feature_importance = model.feature_importances_
    
    # Initialize preprocessor
    preprocessor = KnuthDataPreprocessing()
    
    # Sort by importance
    print("\nSorting features by importance (heapsort)...")
    sorted_indices = preprocessor.sort_by_feature_importance(X, feature_importance, descending=True)
    print(f"Top 5 features: {sorted_indices[:5]}")
    print(f"Top 5 importances: {feature_importance[sorted_indices[:5]]}")
    
    # Find similar samples
    print("\nFinding 5 most similar samples to first sample...")
    query = X[0]
    similar_indices = preprocessor.find_similar_samples(X, query, k=5)
    print(f"Similar sample indices: {similar_indices}")
    
    print("\n✓ Data preprocessing example complete\n")


def example_integrated_workflow():
    """Example: Complete ML workflow using Knuth algorithms"""
    print("=" * 60)
    print("Example 6: Integrated ML Workflow with Knuth Algorithms")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)
    
    # Initialize integrated Knuth ML
    knuth_ml = KnuthMLIntegration(seed=42)
    
    # Step 1: Sample data (reproducible)
    print("\nStep 1: Sampling data...")
    X_sample, y_sample = knuth_ml.data_sampling.stratified_sample(X, y, n_samples=200)
    print(f"Sampled: {X_sample.shape}")
    
    # Step 2: Feature selection
    print("\nStep 2: Feature selection...")
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    feature_result = knuth_ml.feature_selector.forward_selection_knuth(
        X_sample, y_sample, model, k=10
    )
    print(f"Selected {len(feature_result['selected_features'])} features")
    
    # Step 3: Use selected features
    X_selected = X_sample[:, feature_result['selected_features']]
    
    # Step 4: Hyperparameter search
    print("\nStep 3: Hyperparameter search...")
    param_result = knuth_ml.hyperparameter_search.grid_search_knuth(
        RandomForestClassifier(random_state=42),
        X_selected,
        y_sample,
        param_grid={'n_estimators': [10, 20], 'max_depth': [5, 10]},
        cv=3
    )
    print(f"Best parameters: {param_result['best_params']}")
    
    print("\n✓ Integrated workflow example complete\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Knuth Algorithms - ML Examples")
    print("=" * 60 + "\n")
    
    if not SKLEARN_AVAILABLE or not TOOLBOX_AVAILABLE:
        print("Required dependencies not available")
        exit(1)
    
    try:
        example_feature_selection()
        example_hyperparameter_search()
        example_knowledge_graph()
        example_data_sampling()
        example_data_preprocessing()
        example_integrated_workflow()
        
        print("=" * 60)
        print("All Knuth ML examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
