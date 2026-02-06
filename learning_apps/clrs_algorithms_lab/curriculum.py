"""
Curriculum: Introduction to Algorithms (CLRS) — Cormen, Leiserson, Rivest, Stein.
DP, Greedy, Graph, Data Structures.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "clrs_dp", "name": "CLRS Dynamic Programming", "short": "Ch 15 DP", "color": "#2563eb"},
    {"id": "clrs_greedy", "name": "CLRS Greedy", "short": "Ch 16 Greedy", "color": "#059669"},
    {"id": "clrs_graph", "name": "CLRS Graph Algorithms", "short": "Ch 23–26 Graph", "color": "#7c3aed"},
]

CURRICULUM: List[Dict[str, Any]] = [
    {"id": "clrs_obst", "book_id": "clrs_dp", "level": "advanced", "title": "Optimal BST (Ch 15.5)",
     "learn": "Build binary search tree with minimum expected search cost given key frequencies. DP on root choices for each subrange.",
     "try_code": "from clrs_complete_algorithms import CLRSDynamicProgramming\nkeys=['A','B','C'];freq=[0.5,0.3,0.2]\ncost,root=CLRSDynamicProgramming.optimal_binary_search_tree(keys,freq)",
     "try_demo": "clrs_obst",},
    {"id": "clrs_lis", "book_id": "clrs_dp", "level": "intermediate", "title": "Longest Increasing Subsequence",
     "learn": "DP: dp[i] = length of LIS ending at i. Reconstruct via parent pointers. O(n^2); can do O(n log n) with binary search.",
     "try_code": "from clrs_complete_algorithms import CLRSDynamicProgramming\nlength,indices=CLRSDynamicProgramming.longest_increasing_subsequence([10,22,9,33,21,50,41,60])",
     "try_demo": "clrs_lis",},
    {"id": "clrs_coin", "book_id": "clrs_dp", "level": "basics", "title": "Coin Change (Min Coins)",
     "learn": "DP: min coins to make amount. dp[i] = min over coins of 1 + dp[i-coin]. Reconstruct combination.",
     "try_code": "from clrs_complete_algorithms import CLRSDynamicProgramming\nn,combo=CLRSDynamicProgramming.coin_change_min_coins([1,3,4],6)",
     "try_demo": "clrs_coin",},
    {"id": "clrs_rod", "book_id": "clrs_dp", "level": "intermediate", "title": "Rod Cutting",
     "learn": "Maximize profit by cutting rod. prices[i] = price for length i+1. DP with cut choices.",
     "try_code": "from clrs_complete_algorithms import CLRSDynamicProgramming\nprofit,cuts=CLRSDynamicProgramming.rod_cutting([1,5,8,9,10,17,17,20],8)",
     "try_demo": "clrs_rod",},
    {"id": "clrs_prim", "book_id": "clrs_greedy", "level": "intermediate", "title": "Prim's MST",
     "learn": "Grow minimum spanning tree from a start vertex. Use min-heap for next edge. O(E log V).",
     "try_code": "from clrs_complete_algorithms import CLRSGreedyAlgorithms\n# Prim's MST on weighted graph",
     "try_demo": None,},
    {"id": "clrs_activity", "book_id": "clrs_greedy", "level": "basics", "title": "Activity Selection",
     "learn": "Schedule maximum number of non-overlapping activities. Greedy: pick earliest finish, then next compatible.",
     "try_code": "from clrs_complete_algorithms import CLRSGreedyAlgorithms\n# activity selection (start, end) pairs",
     "try_demo": None,},
    {"id": "clrs_bellman", "book_id": "clrs_graph", "level": "advanced", "title": "Bellman-Ford",
     "learn": "Single-source shortest paths with negative edge weights. Relax all edges V-1 times. Detects negative cycles.",
     "try_code": "from clrs_complete_algorithms import CLRSGraphAlgorithms\n# BellmanFord(n, edges, source)",
     "try_demo": None,},
]


def get_curriculum() -> List[Dict[str, Any]]:
    return list(CURRICULUM)


def get_books() -> List[Dict[str, Any]]:
    return list(BOOKS)


def get_levels() -> List[str]:
    return list(LEVELS)


def get_by_book(book_id: str) -> List[Dict[str, Any]]:
    return [c for c in CURRICULUM if c["book_id"] == book_id]


def get_by_level(level: str) -> List[Dict[str, Any]]:
    return [c for c in CURRICULUM if c["level"] == level]


def get_item(item_id: str) -> Dict[str, Any] | None:
    for c in CURRICULUM:
        if c["id"] == item_id:
            return c
    return None
