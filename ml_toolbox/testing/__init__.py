"""
ML Toolbox Testing Module
Comprehensive testing and benchmarking infrastructure
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from .comprehensive_test_suite import ComprehensiveMLTestSuite
    from .benchmark_suite import MLBenchmarkSuite
    __all__ = ['ComprehensiveMLTestSuite', 'MLBenchmarkSuite']
except ImportError as e:
    __all__ = []
    import warnings
    warnings.warn(f"Testing module imports failed: {e}")
