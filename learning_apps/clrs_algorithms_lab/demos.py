"""Demos for CLRS Algorithms Lab."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def clrs_obst():
    from clrs_complete_algorithms import CLRSDynamicProgramming
    keys, freq = ["A", "B", "C"], [0.5, 0.3, 0.2]
    cost, root = CLRSDynamicProgramming.optimal_binary_search_tree(keys, freq)
    return {"ok": True, "output": f"Optimal BST cost = {cost}, root table (0..n) = {root[0]}"}


def clrs_lis():
    from clrs_complete_algorithms import CLRSDynamicProgramming
    arr = [10, 22, 9, 33, 21, 50, 41, 60]
    length, indices = CLRSDynamicProgramming.longest_increasing_subsequence(arr)
    sub = [arr[i] for i in indices]
    return {"ok": True, "output": f"LIS length = {length}, indices = {indices}, subsequence = {sub}"}


def clrs_coin():
    from clrs_complete_algorithms import CLRSDynamicProgramming
    n, combo = CLRSDynamicProgramming.coin_change_min_coins([1, 3, 4], 6)
    return {"ok": True, "output": f"Min coins for 6: {n}, combination: {combo}"}


def clrs_rod():
    from clrs_complete_algorithms import CLRSDynamicProgramming
    profit, cuts = CLRSDynamicProgramming.rod_cutting([1, 5, 8, 9, 10, 17, 17, 20], 8)
    return {"ok": True, "output": f"Max profit for length 8: {profit}, cuts: {cuts}"}


DEMO_HANDLERS = {"clrs_obst": clrs_obst, "clrs_lis": clrs_lis, "clrs_coin": clrs_coin, "clrs_rod": clrs_rod}


def run_demo(demo_id: str):
    if demo_id not in DEMO_HANDLERS:
        return {"ok": False, "output": "", "error": f"Unknown demo: {demo_id}"}
    try:
        return DEMO_HANDLERS[demo_id]()
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}
