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
    
    Also includes:
    - Medulla Oblongata System: Automatic resource regulation
    - Virtual Quantum Computer: CPU-based quantum simulation (optional)
    """
    
    def __init__(self, include_mlops: bool = True, auto_start_medulla: bool = True):
        """
        Initialize ML Toolbox
        
        Args:
            include_mlops: Include MLOps compartment
            auto_start_medulla: Automatically start Medulla resource regulation system
        """
        # Initialize Medulla system (automatic resource regulation)
        self.medulla = None
        if auto_start_medulla:
            try:
                from medulla_oblongata_system import MedullaOblongataSystem
                self.medulla = MedullaOblongataSystem(
                    max_cpu_percent=80.0,
                    max_memory_percent=75.0,
                    min_cpu_reserve=20.0,
                    min_memory_reserve_mb=1024.0
                )
                self.medulla.start_regulation()
                print("[MLToolbox] Medulla Oblongata System started (automatic resource regulation)")
            except ImportError as e:
                print(f"[MLToolbox] Warning: Medulla system not available: {e}")
            except Exception as e:
                print(f"[MLToolbox] Warning: Could not start Medulla system: {e}")
        
        # Initialize compartments (pass medulla to infrastructure)
        self.data = DataCompartment()
        self.infrastructure = InfrastructureCompartment(medulla=self.medulla)
        self.algorithms = AlgorithmsCompartment()
        
        # MLOps compartment (optional)
        if include_mlops and MLOPS_AVAILABLE:
            self.mlops = MLOpsCompartment()
        else:
            self.mlops = None
    
    def __repr__(self):
        mlops_info = f", mlops={len(self.mlops.components)}" if self.mlops else ""
        medulla_info = ", medulla=active" if self.medulla and self.medulla.regulation_running else ""
        return f"MLToolbox(data={len(self.data.components)}, infrastructure={len(self.infrastructure.components)}, algorithms={len(self.algorithms.components)}{mlops_info}{medulla_info})"
    
    def __enter__(self):
        """Context manager entry - Medulla already started"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - Stop Medulla if running"""
        if self.medulla and self.medulla.regulation_running:
            self.medulla.stop_regulation()
    
    def get_system_status(self):
        """Get Medulla system status"""
        if self.medulla:
            return self.medulla.get_system_status()
        return {"status": "medulla_not_available"}
    
    def get_quantum_computer(self, num_qubits: int = 8, use_architecture_optimizations: bool = True):
        """Get a Virtual Quantum Computer instance (uses Medulla if available)"""
        try:
            from virtual_quantum_computer import VirtualQuantumComputer
            return VirtualQuantumComputer(
                num_qubits=num_qubits,
                medulla=self.medulla,
                use_architecture_optimizations=use_architecture_optimizations
            )
        except ImportError:
            raise ImportError("Virtual Quantum Computer not available. Install required dependencies.")
