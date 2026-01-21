"""
Machine Learning Toolbox
Organized into three compartments:
1. Data: Preprocessing, validation, transformation
2. Infrastructure: Kernels, AI components, LLM
3. Algorithms: Models, evaluation, tuning, ensembles

Also includes Advanced ML Toolbox for big data and advanced features
"""
from .compartment1_data import DataCompartment
from .compartment2_infrastructure import InfrastructureCompartment
from .compartment3_algorithms import AlgorithmsCompartment

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
    
    Three compartments:
    1. Data: Preprocessing, validation, transformation
    2. Infrastructure: Kernels, AI components, LLM
    3. Algorithms: Models, evaluation, tuning, ensembles
    """
    
    def __init__(self):
        self.data = DataCompartment()
        self.infrastructure = InfrastructureCompartment()
        self.algorithms = AlgorithmsCompartment()
    
    def __repr__(self):
        return f"MLToolbox(data={len(self.data.components)}, infrastructure={len(self.infrastructure.components)}, algorithms={len(self.algorithms.components)})"
