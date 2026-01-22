"""
Advanced Algorithms - ML Examples
Practical ML applications of advanced algorithms
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available")
    exit(1)

try:
    from advanced_algorithms import (
        NumericalMethods,
        DynamicProgramming,
        GreedyAlgorithms,
        AdvancedDataStructures,
        AdvancedAlgorithms
    )
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    print("Warning: Advanced algorithms not available")
    exit(1)


def example_polynomial_features():
    """Example: Polynomial feature evaluation using Horner's method"""
    print("=" * 60)
    print("Example 1: Polynomial Feature Evaluation (Horner's Method)")
    print("=" * 60)
    
    # Generate polynomial features: x^2 + 2x + 1
    coeffs = [1, 2, 1]
    x_values = np.array([1, 2, 3, 4, 5])
    
    # Evaluate using Horner's method
    results = [NumericalMethods.horner_method(coeffs, x) for x in x_values]
    
    print(f"\nPolynomial: x^2 + 2x + 1")
    print(f"Values: {x_values}")
    print(f"Results: {results}")
    
    print("\n✓ Polynomial evaluation example complete\n")


def example_feature_selection_knapsack():
    """Example: Feature selection as knapsack problem"""
    print("=" * 60)
    print("Example 2: Feature Selection as Knapsack Problem")
    print("=" * 60)
    
    # Features with "costs" (e.g., computation time) and "values" (importance)
    feature_costs = [10, 20, 30, 15, 25]  # Computation time
    feature_values = [60, 100, 120, 80, 90]  # Importance scores
    max_cost = 50  # Maximum computation budget
    
    # Solve as 0/1 knapsack
    max_value, selected = DynamicProgramming.knapsack_01(
        feature_costs, feature_values, max_cost
    )
    
    print(f"\nFeature costs: {feature_costs}")
    print(f"Feature values: {feature_values}")
    print(f"Max cost: {max_cost}")
    print(f"Selected features: {selected}")
    print(f"Total value: {max_value}")
    
    print("\n✓ Feature selection knapsack example complete\n")


def example_model_compression_huffman():
    """Example: Model compression using Huffman coding"""
    print("=" * 60)
    print("Example 3: Model Compression (Huffman Coding)")
    print("=" * 60)
    
    # Character frequencies in model weights (example)
    frequencies = {
        '0': 100,
        '1': 50,
        '2': 30,
        '3': 20,
        '4': 10
    }
    
    # Generate Huffman codes
    codes = GreedyAlgorithms.huffman_coding(frequencies)
    
    print(f"\nFrequencies: {frequencies}")
    print(f"Huffman codes: {codes}")
    
    # Calculate compression ratio
    original_bits = sum(freq * 3 for freq in frequencies.values())  # 3 bits per symbol
    compressed_bits = sum(frequencies[char] * len(codes[char]) for char in frequencies)
    ratio = compressed_bits / original_bits
    
    print(f"Original bits: {original_bits}")
    print(f"Compressed bits: {compressed_bits}")
    print(f"Compression ratio: {ratio:.2f}")
    
    print("\n✓ Model compression example complete\n")


def example_clustering_union_find():
    """Example: Clustering using Union-Find"""
    print("=" * 60)
    print("Example 4: Clustering with Union-Find")
    print("=" * 60)
    
    # Create Union-Find for 10 data points
    uf = AdvancedDataStructures.UnionFind(10)
    
    # Merge clusters based on similarity
    # Points 0,1,2 are similar
    uf.union(0, 1)
    uf.union(1, 2)
    
    # Points 3,4,5 are similar
    uf.union(3, 4)
    uf.union(4, 5)
    
    # Points 6,7 are similar
    uf.union(6, 7)
    
    # Check clusters
    clusters = {}
    for i in range(10):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
    
    print(f"\nClusters: {list(clusters.values())}")
    print(f"Number of clusters: {len(clusters)}")
    
    print("\n✓ Clustering example complete\n")


def example_text_search_trie():
    """Example: Text search using Trie"""
    print("=" * 60)
    print("Example 5: Text Search with Trie")
    print("=" * 60)
    
    # Build Trie from vocabulary
    trie = AdvancedDataStructures.Trie()
    words = ['machine', 'learning', 'model', 'data', 'algorithm', 'model']
    
    for i, word in enumerate(words):
        trie.insert(word, i)
    
    # Search
    print(f"\nVocabulary: {words}")
    print(f"Search 'model': {trie.search('model')}")
    print(f"Search 'mach': {trie.search('mach')}")  # Not a complete word
    
    # Prefix search
    matches = trie.starts_with('ma')
    print(f"Words starting with 'ma': {matches}")
    
    print("\n✓ Text search example complete\n")


def example_sequence_alignment_lcs():
    """Example: Sequence alignment using LCS"""
    print("=" * 60)
    print("Example 6: Sequence Alignment (LCS)")
    print("=" * 60)
    
    # Compare two sequences (e.g., time series, text)
    seq1 = "ABCDGH"
    seq2 = "AEDFHR"
    
    length, lcs = DynamicProgramming.longest_common_subsequence(seq1, seq2)
    
    print(f"\nSequence 1: {seq1}")
    print(f"Sequence 2: {seq2}")
    print(f"LCS length: {length}")
    print(f"LCS: {lcs}")
    
    # Similarity score
    similarity = length / max(len(seq1), len(seq2))
    print(f"Similarity: {similarity:.2f}")
    
    print("\n✓ Sequence alignment example complete\n")


def example_integrated_workflow():
    """Example: Integrated workflow using multiple advanced algorithms"""
    print("=" * 60)
    print("Example 7: Integrated ML Workflow")
    print("=" * 60)
    
    algo = AdvancedAlgorithms()
    
    # Step 1: Polynomial feature evaluation
    print("\nStep 1: Polynomial features...")
    coeffs = [1, 2, 1]
    x = 5
    poly_value = algo.numerical.horner_method(coeffs, x)
    print(f"Polynomial({x}) = {poly_value}")
    
    # Step 2: Feature selection (knapsack)
    print("\nStep 2: Feature selection...")
    costs = [10, 20, 30]
    values = [60, 100, 120]
    max_value, selected = algo.dp.knapsack_01(costs, values, 50)
    print(f"Selected features: {selected}, Value: {max_value}")
    
    # Step 3: Clustering (Union-Find)
    print("\nStep 3: Clustering...")
    uf = algo.data_structures.UnionFind(5)
    uf.union(0, 1)
    uf.union(2, 3)
    print(f"Clusters: {uf.connected(0, 1)}, {uf.connected(2, 3)}")
    
    # Step 4: Text search (Trie)
    print("\nStep 4: Text search...")
    trie = algo.data_structures.Trie()
    trie.insert('feature', 1)
    trie.insert('selection', 2)
    matches = trie.starts_with('feat')
    print(f"Matches: {matches}")
    
    print("\n✓ Integrated workflow example complete\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Advanced Algorithms - ML Examples")
    print("=" * 60 + "\n")
    
    if not SKLEARN_AVAILABLE or not ADVANCED_AVAILABLE:
        print("Required dependencies not available")
        exit(1)
    
    try:
        example_polynomial_features()
        example_feature_selection_knapsack()
        example_model_compression_huffman()
        example_clustering_union_find()
        example_text_search_trie()
        example_sequence_alignment_lcs()
        example_integrated_workflow()
        
        print("=" * 60)
        print("All advanced algorithm examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
