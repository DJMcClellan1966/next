"""
Machine Learning Toolbox
Organized into four compartments:
1. Data: Preprocessing, validation, transformation
2. Infrastructure: Kernels, AI components, LLM
3. Algorithms: Models, evaluation, tuning, ensembles
4. MLOps: Production deployment, monitoring, A/B testing, experiment tracking

Also includes Advanced ML Toolbox for big data and advanced features
"""
from .compartment1_data import DataCompartment
from .compartment2_infrastructure import InfrastructureCompartment
from .compartment3_algorithms import AlgorithmsCompartment

# Try to import MLOps compartment
try:
    from .compartment4_mlops import MLOpsCompartment
    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False
    MLOpsCompartment = None

# Import advanced toolbox
try:
    from .advanced import AdvancedMLToolbox
    __all__ = [
        'DataCompartment',
        'InfrastructureCompartment',
        'AlgorithmsCompartment',
        'MLToolbox',
        'AdvancedMLToolbox'
    ]
except ImportError:
    __all__ = [
        'DataCompartment',
        'InfrastructureCompartment',
        'AlgorithmsCompartment',
        'MLToolbox'
    ]


class MLToolbox:
    """
    Complete Machine Learning Toolbox
    
    Four compartments:
    1. Data: Preprocessing, validation, transformation
    2. Infrastructure: Kernels, AI components, LLM
    3. Algorithms: Models, evaluation, tuning, ensembles
    4. MLOps: Production deployment, monitoring, A/B testing, experiment tracking
    """
    
    def __init__(self, include_mlops: bool = True):
        self.data = DataCompartment()
        self.infrastructure = InfrastructureCompartment()
        self.algorithms = AlgorithmsCompartment()
        
        # MLOps compartment (optional)
        if include_mlops and MLOPS_AVAILABLE:
            self.mlops = MLOpsCompartment()
        else:
            self.mlops = None
    
    def __repr__(self):
        mlops_info = f", mlops={len(self.mlops.components)}" if self.mlops else ""
        return f"MLToolbox(data={len(self.data.components)}, infrastructure={len(self.infrastructure.components)}, algorithms={len(self.algorithms.components)}{mlops_info})"
