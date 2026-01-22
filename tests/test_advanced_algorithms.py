"""
Tests for Advanced Algorithms
Test numerical methods, dynamic programming, greedy, data structures, and graph algorithms
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from advanced_algorithms import (
        NumericalMethods,
        DynamicProgramming,
        GreedyAlgorithms,
        AdvancedDataStructures,
        AdvancedGraphAlgorithms,
        AdvancedAlgorithms
    )
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    pytestmark = pytest.mark.skip("Advanced algorithms not available")


class TestNumericalMethods:
    """Tests for numerical methods"""
    
    def test_horner_method(self):
        """Test Horner's method"""
        # Polynomial: 2x^2 + 3x + 1
        coeffs = [2, 3, 1]
        result = NumericalMethods.horner_method(coeffs, 2)
        assert abs(result - (2*4 + 3*2 + 1)) < 1e-10
    
    def test_euclidean_gcd(self):
        """Test Euclidean GCD"""
        assert NumericalMethods.euclidean_gcd(48, 18) == 6
        assert NumericalMethods.euclidean_gcd(17, 5) == 1
    
    def test_extended_euclidean(self):
        """Test Extended Euclidean"""
        gcd, x, y = NumericalMethods.extended_euclidean(48, 18)
        assert gcd == 6
        assert 48*x + 18*y == 6
    
    def test_modular_inverse(self):
        """Test modular inverse"""
        inv = NumericalMethods.modular_inverse(3, 11)
        assert (3 * inv) % 11 == 1
        
        # No inverse case
        inv = NumericalMethods.modular_inverse(4, 8)
        assert inv is None
    
    def test_modular_exponentiation(self):
        """Test modular exponentiation"""
        result = NumericalMethods.modular_exponentiation(2, 10, 1000)
        assert result == (2**10) % 1000


class TestDynamicProgramming:
    """Tests for dynamic programming"""
    
    def test_longest_common_subsequence(self):
        """Test LCS"""
        length, lcs = DynamicProgramming.longest_common_subsequence("ABCDGH", "AEDFHR")
        assert length == 3
        assert lcs == "ADH"
    
    def test_knapsack_01(self):
        """Test 0/1 Knapsack"""
        weights = [10, 20, 30]
        values = [60, 100, 120]
        capacity = 50
        
        max_value, selected = DynamicProgramming.knapsack_01(weights, values, capacity)
        assert max_value == 220
        assert len(selected) == 2
    
    def test_matrix_chain_multiplication(self):
        """Test matrix chain multiplication"""
        dims = [1, 2, 3, 4, 5]
        min_cost, splits = DynamicProgramming.matrix_chain_multiplication(dims)
        assert min_cost > 0
        assert splits is not None


class TestGreedyAlgorithms:
    """Tests for greedy algorithms"""
    
    def test_huffman_coding(self):
        """Test Huffman coding"""
        frequencies = {'a': 5, 'b': 9, 'c': 12, 'd': 13, 'e': 16, 'f': 45}
        codes = GreedyAlgorithms.huffman_coding(frequencies)
        
        assert len(codes) == len(frequencies)
        # All codes should be unique
        assert len(set(codes.values())) == len(codes)
    
    def test_kruskal_mst(self):
        """Test Kruskal's MST"""
        edges = [
            (0, 1, 10),
            (0, 2, 6),
            (0, 3, 5),
            (1, 3, 15),
            (2, 3, 4)
        ]
        mst = GreedyAlgorithms.kruskal_mst(edges, 4)
        
        assert len(mst) == 3  # n-1 edges
        total_weight = sum(edge[2] for edge in mst)
        assert total_weight == 19
    
    def test_fractional_knapsack(self):
        """Test fractional knapsack"""
        weights = [10, 20, 30]
        values = [60, 100, 120]
        capacity = 50
        
        max_value, fractions = GreedyAlgorithms.fractional_knapsack(weights, values, capacity)
        assert max_value > 0
        assert len(fractions) == 3


class TestAdvancedDataStructures:
    """Tests for advanced data structures"""
    
    def test_min_heap(self):
        """Test MinHeap"""
        heap = AdvancedDataStructures.MinHeap()
        heap.push('a', 3)
        heap.push('b', 1)
        heap.push('c', 2)
        
        priority, item = heap.pop()
        assert item == 'b'
        assert priority == 1
    
    def test_binary_search_tree(self):
        """Test BST"""
        bst = AdvancedDataStructures.BinarySearchTree()
        bst.insert(5, 'five')
        bst.insert(3, 'three')
        bst.insert(7, 'seven')
        
        assert bst.search(5) == 'five'
        assert bst.search(3) == 'three'
        assert bst.search(10) is None
    
    def test_union_find(self):
        """Test Union-Find"""
        uf = AdvancedDataStructures.UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        
        assert uf.connected(0, 1)
        assert uf.connected(2, 3)
        assert not uf.connected(0, 2)
    
    def test_hash_table(self):
        """Test HashTable"""
        ht = AdvancedDataStructures.HashTable()
        ht.insert('key1', 'value1')
        ht.insert('key2', 'value2')
        
        assert ht.get('key1') == 'value1'
        assert ht.get('key2') == 'value2'
        assert ht.get('key3') is None
        
        assert ht.delete('key1')
        assert ht.get('key1') is None
    
    def test_trie(self):
        """Test Trie"""
        trie = AdvancedDataStructures.Trie()
        trie.insert('hello', 1)
        trie.insert('world', 2)
        trie.insert('help', 3)
        
        assert trie.search('hello') == 1
        assert trie.search('world') == 2
        assert trie.search('help') == 3
        assert trie.search('hel') is None
        
        words = trie.starts_with('hel')
        assert 'hello' in words
        assert 'help' in words


class TestAdvancedGraphAlgorithms:
    """Tests for advanced graph algorithms"""
    
    def test_strongly_connected_components(self):
        """Test SCC"""
        graph = {
            0: [1],
            1: [2],
            2: [0, 3],
            3: [4],
            4: [3]
        }
        sccs = AdvancedGraphAlgorithms.strongly_connected_components(graph)
        
        assert len(sccs) > 0
        # All nodes should be in some SCC
        all_nodes = set()
        for scc in sccs:
            all_nodes.update(scc)
        assert len(all_nodes) == 5
    
    def test_floyd_warshall(self):
        """Test Floyd-Warshall"""
        graph = {
            0: [(1, 1), (2, 4)],
            1: [(2, 2), (3, 5)],
            2: [(3, 1)],
            3: []
        }
        dist = AdvancedGraphAlgorithms.floyd_warshall(graph, 4)
        
        assert dist[0][3] < np.inf
        assert dist[0][0] == 0


class TestAdvancedAlgorithmsIntegration:
    """Test unified interface"""
    
    def test_unified_interface(self):
        """Test AdvancedAlgorithms"""
        algo = AdvancedAlgorithms()
        
        assert algo.numerical is not None
        assert algo.dp is not None
        assert algo.greedy is not None
        assert algo.data_structures is not None
        assert algo.graph is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
