"""
Donald Knuth's "The Art of Computer Programming" Algorithms
Implementation of key algorithms from TAOCP for ML Toolbox

Algorithms from:
- Vol. 1: Fundamental Algorithms (Graph algorithms)
- Vol. 2: Seminumerical Algorithms (Random numbers)
- Vol. 3: Sorting and Searching
- Vol. 4: Combinatorial Algorithms
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Iterator
import numpy as np
from collections import deque
import warnings

sys.path.insert(0, str(Path(__file__).parent))


class KnuthRandom:
    """
    Random Number Generation (TAOCP Vol. 2)
    
    High-quality random number generators
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed
        """
        self.seed = seed
        self.state = seed
    
    def linear_congruential_generator(
        self,
        n: int,
        a: int = 1664525,
        c: int = 1013904223,
        m: int = 2**32
    ) -> List[int]:
        """
        Linear Congruential Generator (LCG) - Algorithm A (Vol. 2)
        
        Fast, reproducible random number generation
        
        Args:
            n: Number of random numbers to generate
            a: Multiplier
            c: Increment
            m: Modulus
            
        Returns:
            List of random integers
        """
        numbers = []
        x = self.state
        
        for _ in range(n):
            x = (a * x + c) % m
            numbers.append(x)
        
        self.state = x
        return numbers
    
    def lagged_fibonacci_generator(
        self,
        n: int,
        j: int = 24,
        k: int = 55
    ) -> List[float]:
        """
        Lagged Fibonacci Generator (Vol. 2)
        
        Better quality than LCG
        
        Args:
            n: Number of random numbers to generate
            j, k: Lag parameters (typically 24, 55)
            
        Returns:
            List of random floats in [0, 1)
        """
        # Initialize with LCG
        lcg = self.linear_congruential_generator(k)
        state = [x / (2**32) for x in lcg]
        
        numbers = []
        for i in range(n):
            # Fibonacci-like: x[n] = (x[n-j] + x[n-k]) mod 1
            new_val = (state[-j] + state[-k]) % 1.0
            state.append(new_val)
            numbers.append(new_val)
            state.pop(0)  # Keep only last k values
        
        return numbers
    
    def fisher_yates_shuffle(self, arr: List[Any]) -> List[Any]:
        """
        Fisher-Yates Shuffle - Algorithm P (Vol. 2)
        
        Proper random shuffling
        
        Args:
            arr: List to shuffle
            
        Returns:
            Shuffled list
        """
        arr = arr.copy()
        n = len(arr)
        
        # Generate random indices using LCG
        for i in range(n - 1, 0, -1):
            # Generate random index in [0, i]
            random_numbers = self.linear_congruential_generator(1)
            j = random_numbers[0] % (i + 1)
            
            # Swap
            arr[i], arr[j] = arr[j], arr[i]
        
        return arr
    
    def random_sample_without_replacement(
        self,
        population: List[Any],
        k: int
    ) -> List[Any]:
        """
        Random sample without replacement (Vol. 2)
        
        Efficient sampling using Fisher-Yates
        
        Args:
            population: Population to sample from
            k: Sample size
            
        Returns:
            Random sample
        """
        if k > len(population):
            raise ValueError("Sample size cannot exceed population size")
        
        # Use reservoir sampling for large populations
        if k < len(population) // 2:
            # Partial Fisher-Yates
            arr = population.copy()
            for i in range(k):
                random_numbers = self.linear_congruential_generator(1)
                j = i + (random_numbers[0] % (len(arr) - i))
                arr[i], arr[j] = arr[j], arr[i]
            return arr[:k]
        else:
            # Full shuffle and take first k
            shuffled = self.fisher_yates_shuffle(population)
            return shuffled[:k]


class KnuthSorting:
    """
    Sorting Algorithms (TAOCP Vol. 3)
    """
    
    @staticmethod
    def heapsort(arr: List[Any], key: Optional[callable] = None) -> List[Any]:
        """
        Heapsort - Algorithm H (Vol. 3)
        
        O(n log n) worst-case, in-place sorting
        
        Args:
            arr: List to sort
            key: Optional key function
            
        Returns:
            Sorted list
        """
        if key is None:
            key = lambda x: x
        
        arr = arr.copy()
        n = len(arr)
        
        def heapify(arr, n, i):
            """Heapify subtree rooted at i"""
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and key(arr[left]) > key(arr[largest]):
                largest = left
            
            if right < n and key(arr[right]) > key(arr[largest]):
                largest = right
            
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)
        
        # Build heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
        
        # Extract elements
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            heapify(arr, i, 0)
        
        return arr
    
    @staticmethod
    def quicksort_median_of_three(
        arr: List[Any],
        key: Optional[callable] = None
    ) -> List[Any]:
        """
        Quicksort with median-of-three pivot (Vol. 3)
        
        Optimized quicksort variant
        
        Args:
            arr: List to sort
            key: Optional key function
            
        Returns:
            Sorted list
        """
        if key is None:
            key = lambda x: x
        
        if len(arr) <= 1:
            return arr.copy()
        
        # Median-of-three pivot selection
        first, middle, last = 0, len(arr) // 2, len(arr) - 1
        pivot_idx = sorted(
            [first, middle, last],
            key=lambda i: key(arr[i])
        )[1]  # Median
        
        pivot = arr[pivot_idx]
        arr[0], arr[pivot_idx] = arr[pivot_idx], arr[0]
        
        # Partition
        left = [x for x in arr[1:] if key(x) <= key(pivot)]
        right = [x for x in arr[1:] if key(x) > key(pivot)]
        
        # Recursive sort
        return (
            KnuthSorting.quicksort_median_of_three(left, key) +
            [pivot] +
            KnuthSorting.quicksort_median_of_three(right, key)
        )


class KnuthSearching:
    """
    Searching Algorithms (TAOCP Vol. 3)
    """
    
    @staticmethod
    def binary_search(
        arr: List[Any],
        target: Any,
        key: Optional[callable] = None
    ) -> Optional[int]:
        """
        Binary Search - Algorithm B (Vol. 3)
        
        O(log n) search in sorted array
        
        Args:
            arr: Sorted list
            target: Target value
            key: Optional key function
            
        Returns:
            Index of target or None
        """
        if key is None:
            key = lambda x: x
        
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            mid_val = key(arr[mid])
            target_val = key(target)
            
            if mid_val == target_val:
                return mid
            elif mid_val < target_val:
                left = mid + 1
            else:
                right = mid - 1
        
        return None
    
    @staticmethod
    def interpolation_search(
        arr: List[float],
        target: float
    ) -> Optional[int]:
        """
        Interpolation Search (Vol. 3)
        
        O(log log n) average case for uniform distributions
        
        Args:
            arr: Sorted list of numbers
            target: Target value
            
        Returns:
            Index of target or None
        """
        left, right = 0, len(arr) - 1
        
        while left <= right and arr[left] <= target <= arr[right]:
            if left == right:
                return left if arr[left] == target else None
            
            # Interpolation formula
            pos = left + int(
                ((target - arr[left]) * (right - left)) / (arr[right] - arr[left])
            )
            
            if arr[pos] == target:
                return pos
            elif arr[pos] < target:
                left = pos + 1
            else:
                right = pos - 1
        
        return None


class KnuthCombinatorial:
    """
    Combinatorial Algorithms (TAOCP Vol. 4)
    
    Essential for feature selection and hyperparameter search
    """
    
    @staticmethod
    def generate_subsets_lexicographic(items: List[Any], k: Optional[int] = None) -> Iterator[List[Any]]:
        """
        Generate all subsets - Algorithm L (Lexicographic, Vol. 4)
        
        Generates subsets in lexicographic order
        
        Args:
            items: List of items
            k: Optional subset size (None for all sizes)
            
        Yields:
            Subsets
        """
        n = len(items)
        
        if k is None:
            # Generate all subsets (2^n)
            for i in range(2**n):
                subset = [items[j] for j in range(n) if (i >> j) & 1]
                yield subset
        else:
            # Generate k-subsets only
            if k > n:
                return
            
            # Use bit manipulation for k-subsets
            for i in range(2**n):
                subset = [items[j] for j in range(n) if (i >> j) & 1]
                if len(subset) == k:
                    yield subset
    
    @staticmethod
    def generate_permutations_lexicographic(items: List[Any]) -> Iterator[List[Any]]:
        """
        Generate all permutations - Algorithm L (Vol. 4)
        
        Generates permutations in lexicographic order
        
        Args:
            items: List of items
            
        Yields:
            Permutations
        """
        items = sorted(items)  # Start with sorted
        n = len(items)
        
        yield items.copy()
        
        while True:
            # Find largest j such that a[j] < a[j+1]
            j = n - 2
            while j >= 0 and items[j] >= items[j + 1]:
                j -= 1
            
            if j < 0:
                break  # No more permutations
            
            # Find largest l such that a[j] < a[l]
            l = n - 1
            while items[j] >= items[l]:
                l -= 1
            
            # Swap a[j] and a[l]
            items[j], items[l] = items[l], items[j]
            
            # Reverse a[j+1..n-1]
            items[j+1:] = list(reversed(items[j+1:]))
            
            yield items.copy()
    
    @staticmethod
    def generate_combinations_lexicographic(
        items: List[Any],
        k: int
    ) -> Iterator[List[Any]]:
        """
        Generate k-combinations - Algorithm T (Vol. 4)
        
        Generates combinations in lexicographic order
        
        Args:
            items: List of items
            k: Combination size
            
        Yields:
            Combinations
        """
        n = len(items)
        if k > n:
            return
        
        # Initialize: first combination is [0, 1, ..., k-1]
        c = list(range(k))
        yield [items[i] for i in c]
        
        while True:
            # Find largest j such that c[j] < n - k + j
            j = k - 1
            while j >= 0 and c[j] >= n - k + j:
                j -= 1
            
            if j < 0:
                break  # No more combinations
            
            # Increment c[j]
            c[j] += 1
            
            # Set c[j+1..k-1] to consecutive values
            for i in range(j + 1, k):
                c[i] = c[i-1] + 1
            
            yield [items[i] for i in c]
    
    @staticmethod
    def backtracking_search(
        constraints: List[callable],
        domain: List[Any],
        max_solutions: int = 1
    ) -> List[List[Any]]:
        """
        Backtracking search for constraint satisfaction (Vol. 4)
        
        Finds solutions satisfying all constraints
        
        Args:
            constraints: List of constraint functions (return True if satisfied)
            domain: Domain of values
            max_solutions: Maximum number of solutions to find
            
        Returns:
            List of solutions
        """
        solutions = []
        
        def backtrack(assignment: List[Any], depth: int):
            if len(solutions) >= max_solutions:
                return
            
            # Check if all constraints satisfied
            if depth == len(constraints):
                if all(constraint(assignment) for constraint in constraints):
                    solutions.append(assignment.copy())
                return
            
            # Try each value in domain
            for value in domain:
                assignment.append(value)
                # Prune if constraint violated
                if constraints[depth](assignment):
                    backtrack(assignment, depth + 1)
                assignment.pop()
        
        backtrack([], 0)
        return solutions


class KnuthGraph:
    """
    Graph Algorithms (TAOCP Vol. 1, 4)
    """
    
    @staticmethod
    def depth_first_search(
        graph: Dict[Any, List[Any]],
        start: Any
    ) -> List[Any]:
        """
        Depth-First Search (Vol. 1)
        
        Graph traversal
        
        Args:
            graph: Adjacency list representation
            start: Starting node
            
        Returns:
            List of visited nodes in DFS order
        """
        visited = []
        stack = [start]
        seen = {start}
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.append(node)
                
                # Add neighbors to stack
                for neighbor in graph.get(node, []):
                    if neighbor not in seen:
                        stack.append(neighbor)
                        seen.add(neighbor)
        
        return visited
    
    @staticmethod
    def breadth_first_search(
        graph: Dict[Any, List[Any]],
        start: Any
    ) -> List[Any]:
        """
        Breadth-First Search (Vol. 1)
        
        Level-order traversal
        
        Args:
            graph: Adjacency list representation
            start: Starting node
            
        Returns:
            List of visited nodes in BFS order
        """
        visited = []
        queue = deque([start])
        seen = {start}
        
        while queue:
            node = queue.popleft()
            visited.append(node)
            
            # Add neighbors to queue
            for neighbor in graph.get(node, []):
                if neighbor not in seen:
                    queue.append(neighbor)
                    seen.add(neighbor)
        
        return visited
    
    @staticmethod
    def topological_sort(graph: Dict[Any, List[Any]]) -> List[Any]:
        """
        Topological Sort (Vol. 1)
        
        Order nodes such that all dependencies come before dependents
        
        Args:
            graph: Directed acyclic graph (adjacency list)
            
        Returns:
            Topologically sorted nodes
        """
        # Calculate in-degrees
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                if neighbor not in in_degree:
                    in_degree[neighbor] = 0
                in_degree[neighbor] += 1
        
        # Find nodes with no incoming edges
        queue = deque([node for node in in_degree if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # Reduce in-degree of neighbors
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(result) != len(in_degree):
            raise ValueError("Graph contains cycles")
        
        return result
    
    @staticmethod
    def shortest_path_dijkstra(
        graph: Dict[Any, List[Tuple[Any, float]]],
        start: Any,
        end: Optional[Any] = None
    ) -> Dict[Any, float]:
        """
        Dijkstra's Shortest Path (Vol. 1)
        
        Find shortest paths from start to all nodes
        
        Args:
            graph: Weighted graph (adjacency list with weights)
            start: Starting node
            end: Optional end node (stops early if provided)
            
        Returns:
            Dictionary of {node: shortest_distance}
        """
        import heapq
        
        distances = {start: 0}
        pq = [(0, start)]
        visited = set()
        
        while pq:
            dist, node = heapq.heappop(pq)
            
            if node in visited:
                continue
            
            visited.add(node)
            
            # Early termination if end node reached
            if end and node == end:
                break
            
            # Update distances to neighbors
            for neighbor, weight in graph.get(node, []):
                if neighbor not in visited:
                    new_dist = dist + weight
                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))
        
        return distances


class KnuthString:
    """
    String Algorithms (TAOCP Vol. 3)
    """
    
    @staticmethod
    def knuth_morris_pratt(text: str, pattern: str) -> List[int]:
        """
        Knuth-Morris-Pratt (KMP) Algorithm (Vol. 3)
        
        Efficient pattern matching
        
        Args:
            text: Text to search in
            pattern: Pattern to find
            
        Returns:
            List of indices where pattern found
        """
        if not pattern:
            return []
        
        # Build failure function (partial match table)
        def build_failure_function(p):
            m = len(p)
            fail = [0] * m
            j = 0
            
            for i in range(1, m):
                while j > 0 and p[i] != p[j]:
                    j = fail[j - 1]
                if p[i] == p[j]:
                    j += 1
                fail[i] = j
            
            return fail
        
        fail = build_failure_function(pattern)
        matches = []
        n, m = len(text), len(pattern)
        j = 0
        
        for i in range(n):
            while j > 0 and text[i] != pattern[j]:
                j = fail[j - 1]
            if text[i] == pattern[j]:
                j += 1
            if j == m:
                matches.append(i - m + 1)
                j = fail[j - 1]
        
        return matches
    
    @staticmethod
    def edit_distance(s1: str, s2: str) -> int:
        """
        Edit Distance (Levenshtein) - Dynamic Programming (Vol. 3)
        
        Minimum number of edits to transform s1 to s2
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Delete
                        dp[i][j-1],      # Insert
                        dp[i-1][j-1]     # Replace
                    )
        
        return dp[m][n]


class KnuthAlgorithms:
    """
    Unified interface for all Knuth algorithms
    """
    
    def __init__(self, seed: int = 42):
        self.random = KnuthRandom(seed)
        self.sorting = KnuthSorting()
        self.searching = KnuthSearching()
        self.combinatorial = KnuthCombinatorial()
        self.graph = KnuthGraph()
        self.string = KnuthString()
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies for Knuth algorithms"""
        return {
            'numpy': 'numpy>=1.26.0',
            'python': 'Python 3.8+'
        }
