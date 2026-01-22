"""
Tests for Foundational Algorithms
Test Sedgewick, Skiena, Aho/Hopcroft/Ullman, and Bentley algorithms
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from foundational_algorithms import (
        SedgewickDataStructures,
        SkienaAlgorithms,
        AhoHopcroftUllman,
        BentleyAlgorithms,
        FoundationalAlgorithms
    )
    FOUNDATIONAL_AVAILABLE = True
except ImportError:
    FOUNDATIONAL_AVAILABLE = False
    pytestmark = pytest.mark.skip("Foundational algorithms not available")


class TestSedgewickDataStructures:
    """Tests for Sedgewick data structures"""
    
    def test_red_black_tree(self):
        """Test Red-Black Tree"""
        rbt = SedgewickDataStructures.RedBlackTree()
        rbt.insert(5, 'five')
        rbt.insert(3, 'three')
        rbt.insert(7, 'seven')
        
        assert rbt.search(5) == 'five'
        assert rbt.search(3) == 'three'
        assert rbt.search(10) is None
    
    def test_skip_list(self):
        """Test Skip List"""
        sl = SedgewickDataStructures.SkipList()
        sl.insert(5, 'five')
        sl.insert(3, 'three')
        sl.insert(7, 'seven')
        
        assert sl.search(5) == 'five'
        assert sl.search(3) == 'three'
        assert sl.search(10) is None
    
    def test_max_flow_min_cut(self):
        """Test Max Flow / Min Cut"""
        graph = {
            0: [(1, 16), (2, 13)],
            1: [(2, 10), (3, 12)],
            2: [(1, 4), (3, 14)],
            3: []
        }
        max_flow = SedgewickDataStructures.max_flow_min_cut(graph, 0, 3)
        assert max_flow > 0


class TestSkienaAlgorithms:
    """Tests for Skiena algorithms"""
    
    def test_backtracking_framework(self):
        """Test backtracking framework"""
        def candidates(state):
            return [1, 2, 3] if len(state) < 2 else []
        
        def is_valid(state, move):
            return move not in state.values()
        
        def is_complete(state):
            return len(state) == 2
        
        def make_move(state, move):
            new_state = state.copy()
            new_state[len(state)] = move
            return new_state
        
        def unmake_move(state, move):
            new_state = state.copy()
            if new_state:
                new_state.pop(max(new_state.keys()))
            return new_state
        
        solutions = SkienaAlgorithms.backtracking_framework(
            candidates, is_valid, is_complete, make_move, unmake_move
        )
        assert len(solutions) > 0
    
    def test_greedy_approximation(self):
        """Test greedy approximation"""
        items = [1, 2, 3, 4, 5]
        value_func = lambda x: x
        constraint_func = lambda selected, item: sum(selected) + item <= 10
        selection_func = lambda items: max(items, key=value_func)
        
        selected = SkienaAlgorithms.greedy_approximation(
            items, value_func, constraint_func, selection_func
        )
        assert len(selected) > 0
    
    def test_monte_carlo(self):
        """Test Monte Carlo algorithm"""
        def problem_func():
            return np.random.rand()
        
        result = SkienaAlgorithms.monte_carlo_algorithm(problem_func, n_trials=100)
        assert 'mean' in result
        assert 'std' in result


class TestAhoHopcroftUllman:
    """Tests for Aho/Hopcroft/Ullman algorithms"""
    
    def test_avl_tree(self):
        """Test AVL Tree"""
        avl = AhoHopcroftUllman.AVLTree()
        avl.insert(5, 'five')
        avl.insert(3, 'three')
        avl.insert(7, 'seven')
        avl.insert(1, 'one')
        
        assert avl.search(5) == 'five'
        assert avl.search(3) == 'three'
        assert avl.search(10) is None


class TestBentleyAlgorithms:
    """Tests for Bentley algorithms"""
    
    def test_maximum_subarray(self):
        """Test maximum subarray (Kadane's)"""
        arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
        max_sum, start, end = BentleyAlgorithms.maximum_subarray(arr)
        assert max_sum == 6
        assert start <= end
    
    def test_two_sum(self):
        """Test two sum"""
        arr = [2, 7, 11, 15]
        result = BentleyAlgorithms.two_sum(arr, 9)
        assert result is not None
        assert arr[result[0]] + arr[result[1]] == 9
    
    def test_rotate_array(self):
        """Test array rotation"""
        arr = [1, 2, 3, 4, 5]
        rotated = BentleyAlgorithms.rotate_array(arr, 2)
        assert rotated == [3, 4, 5, 1, 2]
    
    def test_search_rotated_array(self):
        """Test search in rotated array"""
        arr = [4, 5, 6, 7, 0, 1, 2]
        idx = BentleyAlgorithms.search_rotated_array(arr, 0)
        assert idx == 4
    
    def test_bit_manipulation_tricks(self):
        """Test bit manipulation tricks"""
        tricks = BentleyAlgorithms.bit_manipulation_tricks()
        assert tricks['is_power_of_two'](8)
        assert not tricks['is_power_of_two'](7)
        assert tricks['count_set_bits'](7) == 3


class TestFoundationalAlgorithms:
    """Test unified interface"""
    
    def test_unified_interface(self):
        """Test FoundationalAlgorithms"""
        algo = FoundationalAlgorithms()
        assert algo.sedgewick is not None
        assert algo.skiena is not None
        assert algo.aho_hopcroft_ullman is not None
        assert algo.bentley is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
