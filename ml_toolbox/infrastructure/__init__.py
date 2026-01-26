"""
ML Toolbox Infrastructure Module
Infrastructure components and utilities
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ml_toolbox.infrastructure.performance_metrics import (
        PerformanceMetric, PerformanceAudit, PerformanceMonitor, get_performance_monitor
    )
    __all__ = [
        'PerformanceMetric',
        'PerformanceAudit',
        'PerformanceMonitor',
        'get_performance_monitor'
    ]
except ImportError as e:
    __all__ = []
    import warnings
    warnings.warn(f"Performance metrics not available: {e}")

# Neural Lace (Sci-Fi)
try:
    from ml_toolbox.infrastructure.neural_lace import (
        NeuralThread, NeuralLace, DirectNeuralInterface
    )
    __all__.extend(['NeuralThread', 'NeuralLace', 'DirectNeuralInterface'])
except ImportError:
    pass
except ImportError as e:
    __all__ = []
    import warnings
    warnings.warn(f"Infrastructure module imports failed: {e}")
