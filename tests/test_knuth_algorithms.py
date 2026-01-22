"""
Tests for Knuth Algorithms
Verify correctness against reference implementations and Knuth's specifications
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from knuth_algorithms import (
        KnuthRandom,
        KnuthSorting,
        KnuthSearching,
        KnuthCombinatorial,
        KnuthGraph,
        KnuthString,
        KnuthAlgorithms
    )
    KNUTH_AVAILABLE = True
except ImportError:
    KNUTH_AVAILABLE = False
    pytestmark = pytest.mark.skip("Knuth algorithms not available")


class TestKnuthRandom:
    """Tests for random number generation"""
    
    def test_linear_congruential_generator(self):
        """Test LCG correctness"""
        rng = KnuthRandom(seed=42)
        numbers = rng.linear_congruential_generator(10)
        
        # Should generate 10 numbers
        assert len(numbers) == 10
        
        # Should be integers
        assert all(isinstance(n, int) for n in numbers)
        
        # Should be reproducible
        rng2 = KnuthRandom(seed=42)
        numbers2 = rng2.linear_congruential_generator(10)
        assert numbers == numbers2
    
    def test_fisher_yates_shuffle(self):
        """Test Fisher-Yates shuffle"""
        rng = KnuthRandom(seed=42)
        arr = list(range(10))
        shuffled = rng.fisher_yates_shuffle(arr)
        
        # Should have same elements
        assert sorted(shuffled) == sorted(arr)
        
        # Should be different order (with high probability)
        assert shuffled != arr or len(arr) <= 1
    
    def test_random_sample_without_replacement(self):
        """Test random sampling"""
        rng = KnuthRandom(seed=42)
        population = list(range(100))
        sample = rng.random_sample_without_replacement(population, k=10)
        
        # Should have correct size
        assert len(sample) == 10
        
        # Should have no duplicates
        assert len(sample) == len(set(sample))
        
        # All should be from population
        assert all(x in population for x in sample)


class TestKnuthSorting:
    """Tests for sorting algorithms"""
    
    def test_heapsort(self):
        """Test heapsort correctness"""
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        sorted_arr = KnuthSorting.heapsort(arr)
        
        # Should be sorted
        assert sorted_arr == sorted(arr)
        
        # Should not modify original
        assert arr == [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    
    def test_heapsort_with_key(self):
        """Test heapsort with key function"""
        arr = ['apple', 'banana', 'cherry']
        sorted_arr = KnuthSorting.heapsort(arr, key=len)
        
        # Should be sorted by length
        assert [len(x) for x in sorted_arr] == sorted([len(x) for x in arr])
    
    def test_quicksort_median_of_three(self):
        """Test quicksort correctness"""
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        sorted_arr = KnuthSorting.quicksort_median_of_three(arr)
        
        # Should be sorted
        assert sorted_arr == sorted(arr)


class TestKnuthSearching:
    """Tests for searching algorithms"""
    
    def test_binary_search(self):
        """Test binary search"""
        arr = [1, 3, 5, 7, 9, 11, 13, 15]
        
        # Find existing element
        idx = KnuthSearching.binary_search(arr, 7)
        assert idx == 3
        assert arr[idx] == 7
        
        # Find non-existing element
        idx = KnuthSearching.binary_search(arr, 8)
        assert idx is None
    
    def test_interpolation_search(self):
        """Test interpolation search"""
        arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        # Find existing element
        idx = KnuthSearching.interpolation_search(arr, 5.0)
        assert idx == 4
        assert arr[idx] == 5.0
        
        # Find non-existing element
        idx = KnuthSearching.interpolation_search(arr, 5.5)
        assert idx is None


class TestKnuthCombinatorial:
    """Tests for combinatorial algorithms"""
    
    def test_generate_subsets(self):
        """Test subset generation"""
        items = ['a', 'b', 'c']
        subsets = list(KnuthCombinatorial.generate_subsets_lexicographic(items))
        
        # Should generate 2^3 = 8 subsets
        assert len(subsets) == 8
        
        # Should include empty set and full set
        assert [] in subsets
        assert items in subsets
    
    def test_generate_permutations(self):
        """Test permutation generation"""
        items = [1, 2, 3]
        perms = list(KnuthCombinatorial.generate_permutations_lexicographic(items))
        
        # Should generate 3! = 6 permutations
        assert len(perms) == 6
        
        # Should all be unique
        assert len(perms) == len(set(tuple(p) for p in perms))
    
    def test_generate_combinations(self):
        """Test combination generation"""
        items = [1, 2, 3, 4]
        combs = list(KnuthCombinatorial.generate_combinations_lexicographic(items, k=2))
        
        # Should generate C(4,2) = 6 combinations
        assert len(combs) == 6
        
        # Each should have size 2
        assert all(len(c) == 2 for c in combs)
    
    def test_backtracking_search(self):
        """Test backtracking constraint satisfaction"""
        # 3 variables, each from domain; final constraint: sum < 10, len == 3
        constraints = [
            lambda x: sum(x) < 10,
            lambda x: sum(x) < 10,
            lambda x: sum(x) < 10 and len(x) == 3
        ]
        domain = [1, 2, 3, 4]
        
        solutions = KnuthCombinatorial.backtracking_search(constraints, domain, max_solutions=10)
        
        # All solutions should satisfy constraints
        for sol in solutions:
            assert sum(sol) < 10
            assert len(sol) == 3


class TestKnuthGraph:
    """Tests for graph algorithms"""
    
    def test_depth_first_search(self):
        """Test DFS"""
        graph = {
            'A': ['B', 'C'],
            'B': ['D'],
            'C': ['D'],
            'D': []
        }
        
        visited = KnuthGraph.depth_first_search(graph, 'A')
        
        # Should visit all nodes
        assert len(visited) == 4
        assert 'A' in visited
        assert 'D' in visited
    
    def test_breadth_first_search(self):
        """Test BFS"""
        graph = {
            'A': ['B', 'C'],
            'B': ['D'],
            'C': ['D'],
            'D': []
        }
        
        visited = KnuthGraph.breadth_first_search(graph, 'A')
        
        # Should visit all nodes
        assert len(visited) == 4
        # BFS should visit A first
        assert visited[0] == 'A'
    
    def test_topological_sort(self):
        """Test topological sort"""
        graph = {
            'A': ['B', 'C'],
            'B': ['D'],
            'C': ['D'],
            'D': []
        }
        
        sorted_nodes = KnuthGraph.topological_sort(graph)
        
        # D should come after B and C
        assert sorted_nodes.index('D') > sorted_nodes.index('B')
        assert sorted_nodes.index('D') > sorted_nodes.index('C')
    
    def test_shortest_path_dijkstra(self):
        """Test Dijkstra's algorithm"""
        graph = {
            'A': [('B', 1), ('C', 4)],
            'B': [('C', 2), ('D', 5)],
            'C': [('D', 1)],
            'D': []
        }
        
        distances = KnuthGraph.shortest_path_dijkstra(graph, 'A')
        
        # A to D should be 4 (A->B->C->D)
        assert distances['D'] == 4
        assert distances['A'] == 0


class TestKnuthString:
    """Tests for string algorithms"""
    
    def test_knuth_morris_pratt(self):
        """Test KMP algorithm"""
        # Pattern "ABABCABAB" appears in text at index 10
        text = "ABABDABACDABABCABAB"
        pattern = "ABABCABAB"
        
        matches = KnuthString.knuth_morris_pratt(text, pattern)
        
        assert len(matches) > 0
        for idx in matches:
            assert text[idx:idx+len(pattern)] == pattern
        
        # Non-overlapping: "abc" in "abcabc" at 0 and 3
        matches2 = KnuthString.knuth_morris_pratt("abcabc", "abc")
        assert len(matches2) == 2
        assert matches2[0] == 0 and matches2[1] == 3
    
    def test_edit_distance(self):
        """Test edit distance"""
        # "kitten" -> "sitting" = 3 edits
        dist = KnuthString.edit_distance("kitten", "sitting")
        assert dist == 3
        
        # Same string = 0 edits
        dist = KnuthString.edit_distance("abc", "abc")
        assert dist == 0


class TestKnuthAlgorithmsIntegration:
    """Test Knuth algorithms integration"""
    
    def test_unified_interface(self):
        """Test unified KnuthAlgorithms interface"""
        knuth = KnuthAlgorithms(seed=42)
        
        # Test random
        numbers = knuth.random.linear_congruential_generator(10)
        assert len(numbers) == 10
        
        # Test sorting
        sorted_arr = knuth.sorting.heapsort([3, 1, 4, 1, 5])
        assert sorted_arr == [1, 1, 3, 4, 5]
        
        # Test searching
        idx = knuth.searching.binary_search([1, 3, 5, 7], 5)
        assert idx == 2
        
        # Test combinatorial
        subsets = list(knuth.combinatorial.generate_subsets_lexicographic([1, 2, 3]))
        assert len(subsets) == 8
        
        # Test graph
        graph = {'A': ['B'], 'B': []}
        visited = knuth.graph.depth_first_search(graph, 'A')
        assert len(visited) == 2
        
        # Test string
        matches = knuth.string.knuth_morris_pratt("abcabc", "abc")
        assert len(matches) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
