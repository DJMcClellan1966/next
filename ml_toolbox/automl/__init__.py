"""
ML Toolbox AutoML Module
Automated machine learning framework
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ml_toolbox.automl.automl_framework import AutoMLFramework
    __all__ = ['AutoMLFramework']
except ImportError as e:
    __all__ = []
    import warnings
    warnings.warn(f"AutoML module imports failed: {e}")
