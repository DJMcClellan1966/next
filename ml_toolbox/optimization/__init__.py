"""
ML Toolbox Optimization Module
Model compression, calibration, and optimization
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from .model_compression import ModelCompression
    from .model_calibration import ModelCalibration
    __all__ = ['ModelCompression', 'ModelCalibration']
except ImportError as e:
    __all__ = []
    import warnings
    warnings.warn(f"Optimization module imports failed: {e}")
