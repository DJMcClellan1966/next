"""
ML Toolbox Optimization Module
Model compression, calibration, and optimization
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

__all__ = []

try:
    from ml_toolbox.optimization.model_compression import ModelCompressor as ModelCompression
    from ml_toolbox.optimization.model_calibration import ModelCalibrator as ModelCalibration
    __all__.extend(['ModelCompression', 'ModelCalibration'])
except ImportError as e:
    import warnings
    warnings.warn(f"Optimization module imports failed: {e}")

# Evolutionary Algorithms (Darwin)
try:
    from ml_toolbox.optimization.evolutionary_algorithms import (
        GeneticAlgorithm, DifferentialEvolution, evolutionary_feature_selection
    )
    __all__.extend(['GeneticAlgorithm', 'DifferentialEvolution', 'evolutionary_feature_selection'])
except ImportError:
    pass

# Control Theory (Wiener)
try:
    from ml_toolbox.optimization.control_theory import (
        PIDController, AdaptiveLearningRateController, TrainingStabilityMonitor,
        AdaptiveHyperparameterTuner
    )
    __all__.extend(['PIDController', 'AdaptiveLearningRateController', 
                    'TrainingStabilityMonitor', 'AdaptiveHyperparameterTuner'])
except ImportError:
    pass

# Bounded Rationality (Simon)
try:
    from ml_toolbox.optimization.bounded_rationality import (
        SatisficingOptimizer, AdaptiveAspirationLevel, HeuristicModelSelector,
        fast_approximate_inference
    )
    __all__.extend(['SatisficingOptimizer', 'AdaptiveAspirationLevel', 
                    'HeuristicModelSelector', 'fast_approximate_inference'])
except ImportError:
    pass

# Systems Theory (Bateson)
try:
    from ml_toolbox.optimization.systems_theory import (
        MultiObjectiveOptimizer, DoubleBindResolver, SystemHierarchy, MetaCommunication
    )
    __all__.extend(['MultiObjectiveOptimizer', 'DoubleBindResolver', 
                    'SystemHierarchy', 'MetaCommunication'])
except ImportError:
    pass

# Multiverse Processing (Sci-Fi)
try:
    from ml_toolbox.optimization.multiverse import (
        ParallelUniverse, MultiverseProcessor
    )
    __all__.extend(['ParallelUniverse', 'MultiverseProcessor'])
except ImportError:
    pass
