"""
Virtual Quantum Computer
Uses CPU cores and threads to simulate quantum computing operations

Leverages architecture-specific optimizations for maximum performance
"""
import sys
from pathlib import Path
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available")

try:
    from medulla_oblongata_system import MedullaOblongataSystem
    MEDULLA_AVAILABLE = True
except ImportError:
    MEDULLA_AVAILABLE = False
    warnings.warn("Medulla system not available")

try:
    from architecture_optimizer import get_architecture_optimizer
    ARCHITECTURE_OPTIMIZER_AVAILABLE = True
except ImportError:
    ARCHITECTURE_OPTIMIZER_AVAILABLE = False
    warnings.warn("Architecture optimizer not available")


class QuantumGate:
    """Quantum gate operations"""
    
    @staticmethod
    def pauli_x(qubit_state: np.ndarray) -> np.ndarray:
        """Pauli-X (NOT) gate"""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        return np.dot(X, qubit_state)
    
    @staticmethod
    def pauli_y(qubit_state: np.ndarray) -> np.ndarray:
        """Pauli-Y gate"""
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        return np.dot(Y, qubit_state)
    
    @staticmethod
    def pauli_z(qubit_state: np.ndarray) -> np.ndarray:
        """Pauli-Z gate"""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return np.dot(Z, qubit_state)
    
    @staticmethod
    def hadamard(qubit_state: np.ndarray) -> np.ndarray:
        """Hadamard gate"""
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        return np.dot(H, qubit_state)
    
    @staticmethod
    def cnot(control: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CNOT gate (controlled-NOT)"""
        # Simplified: if control is |1>, flip target
        if abs(control[1]) > abs(control[0]):
            target = QuantumGate.pauli_x(target)
        return control, target


class VirtualQuantumComputer:
    """
    Virtual Quantum Computer
    
    Simulates quantum computing using CPU cores and threads
    Leverages architecture-specific optimizations for performance
    """
    
    def __init__(
        self,
        num_qubits: int = 8,
        medulla: Optional[MedullaOblongataSystem] = None,
        use_architecture_optimizations: bool = True
    ):
        """
        Initialize Virtual Quantum Computer
        
        Args:
            num_qubits: Number of qubits to simulate
            medulla: Medulla system for resource regulation
            use_architecture_optimizations: Use architecture-specific optimizations
        """
        self.num_qubits = num_qubits
        self.medulla = medulla
        self.use_architecture_optimizations = use_architecture_optimizations
        
        # Architecture optimizer
        if use_architecture_optimizations and ARCHITECTURE_OPTIMIZER_AVAILABLE:
            self.arch_optimizer = get_architecture_optimizer()
            self.optimal_threads = self.arch_optimizer.get_optimal_thread_count()
        else:
            self.arch_optimizer = None
            self.optimal_threads = 4
        
        # Quantum state (2^num_qubits complex amplitudes)
        self.state_size = 2 ** num_qubits
        self.quantum_state = np.zeros(self.state_size, dtype=complex)
        self.quantum_state[0] = 1.0  # Initialize to |0...0>
        
        # Performance metrics
        self.metrics = {
            'operations_performed': 0,
            'total_compute_time': 0.0,
            'parallel_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def initialize_state(self, state: Optional[np.ndarray] = None):
        """Initialize quantum state"""
        if state is not None:
            if len(state) == self.state_size:
                self.quantum_state = state.copy()
            else:
                raise ValueError(f"State size mismatch: {len(state)} != {self.state_size}")
        else:
            self.quantum_state = np.zeros(self.state_size, dtype=complex)
            self.quantum_state[0] = 1.0
    
    def apply_gate(self, gate: str, qubit: int, parallel: bool = True):
        """Apply a quantum gate to a qubit"""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required for quantum operations")
        
        start_time = time.time()
        
        # Architecture-optimize state
        if self.arch_optimizer:
            self.quantum_state = self.arch_optimizer.optimize_array_operations(self.quantum_state)
        
        # Apply gate
        if gate == 'X':
            self._apply_pauli_x(qubit, parallel)
        elif gate == 'Y':
            self._apply_pauli_y(qubit, parallel)
        elif gate == 'Z':
            self._apply_pauli_z(qubit, parallel)
        elif gate == 'H':
            self._apply_hadamard(qubit, parallel)
        else:
            raise ValueError(f"Unknown gate: {gate}")
        
        elapsed = time.time() - start_time
        self.metrics['operations_performed'] += 1
        self.metrics['total_compute_time'] += elapsed
        
        if parallel:
            self.metrics['parallel_operations'] += 1
    
    def _apply_pauli_x(self, qubit: int, parallel: bool):
        """Apply Pauli-X gate using optimized operations"""
        # Vectorized operation
        mask = 1 << qubit
        indices = np.arange(self.state_size)
        
        # Flip qubit: swap |0> and |1> states
        swap_indices = indices ^ mask
        
        # Vectorized swap
        temp = self.quantum_state.copy()
        self.quantum_state = temp[swap_indices]
    
    def _apply_pauli_y(self, qubit: int, parallel: bool):
        """Apply Pauli-Y gate"""
        mask = 1 << qubit
        indices = np.arange(self.state_size)
        
        # Y gate: i|1><0| - i|0><1|
        swap_indices = indices ^ mask
        temp = self.quantum_state.copy()
        
        # Apply phase and swap
        phase = np.where((indices & mask) != 0, -1j, 1j)
        self.quantum_state = phase * temp[swap_indices]
    
    def _apply_pauli_z(self, qubit: int, parallel: bool):
        """Apply Pauli-Z gate"""
        mask = 1 << qubit
        indices = np.arange(self.state_size)
        
        # Z gate: phase flip
        phase = np.where((indices & mask) != 0, -1, 1)
        self.quantum_state *= phase
    
    def _apply_hadamard(self, qubit: int, parallel: bool):
        """Apply Hadamard gate"""
        mask = 1 << qubit
        indices = np.arange(self.state_size)
        
        # Hadamard: superposition
        temp = self.quantum_state.copy()
        
        # Split into |0> and |1> components
        zero_mask = (indices & mask) == 0
        one_mask = ~zero_mask
        
        # Apply Hadamard transformation
        self.quantum_state = np.zeros_like(temp)
        self.quantum_state[zero_mask] = (temp[zero_mask] + temp[zero_mask ^ mask]) / np.sqrt(2)
        self.quantum_state[one_mask] = (temp[one_mask ^ mask] - temp[one_mask]) / np.sqrt(2)
    
    def measure(self, qubit: int) -> int:
        """Measure a qubit (returns 0 or 1)"""
        mask = 1 << qubit
        indices = np.arange(self.state_size)
        
        # Calculate probabilities
        probs = np.abs(self.quantum_state) ** 2
        
        # Probability of |0> and |1>
        prob_zero = np.sum(probs[zero_mask := (indices & mask) == 0])
        prob_one = np.sum(probs[one_mask := (indices & mask) != 0])
        
        # Normalize
        total = prob_zero + prob_one
        if total > 0:
            prob_zero /= total
            prob_one /= total
        
        # Sample
        result = 1 if np.random.random() < prob_one else 0
        
        # Collapse state
        if result == 0:
            self.quantum_state[one_mask] = 0
        else:
            self.quantum_state[zero_mask] = 0
        
        # Renormalize
        norm = np.linalg.norm(self.quantum_state)
        if norm > 0:
            self.quantum_state /= norm
        
        return result
    
    def parallel_quantum_operation(
        self,
        operations: List[Tuple[str, int]],
        max_workers: Optional[int] = None
    ) -> float:
        """Perform multiple quantum operations in parallel"""
        if max_workers is None:
            if self.medulla:
                resources = self.medulla.get_available_resources()
                max_workers = min(resources['cores_allocated'], self.optimal_threads)
            else:
                max_workers = self.optimal_threads
        
        start_time = time.time()
        
        # Split operations across threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for gate, qubit in operations:
                future = executor.submit(self.apply_gate, gate, qubit, parallel=True)
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()
        
        elapsed = time.time() - start_time
        return elapsed
    
    def get_state_vector(self) -> np.ndarray:
        """Get current quantum state vector"""
        return self.quantum_state.copy()
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities"""
        return np.abs(self.quantum_state) ** 2
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            'num_qubits': self.num_qubits,
            'state_size': self.state_size,
            'optimal_threads': self.optimal_threads,
            'avg_operation_time': (
                self.metrics['total_compute_time'] / max(1, self.metrics['operations_performed'])
            )
        }
    
    def reset(self):
        """Reset quantum computer to initial state"""
        self.initialize_state()
        self.metrics = {
            'operations_performed': 0,
            'total_compute_time': 0.0,
            'parallel_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }


# Example usage
if __name__ == '__main__':
    print("Virtual Quantum Computer")
    print("="*80)
    
    # Create Medulla system
    if MEDULLA_AVAILABLE:
        medulla = MedullaOblongataSystem()
        medulla.start_regulation()
    else:
        medulla = None
    
    # Create virtual quantum computer
    qc = VirtualQuantumComputer(num_qubits=8, medulla=medulla)
    
    print(f"\nInitialized {qc.num_qubits}-qubit quantum computer")
    print(f"State size: {qc.state_size}")
    print(f"Optimal threads: {qc.optimal_threads}")
    
    # Perform quantum operations
    print("\nPerforming quantum operations...")
    
    # Create superposition
    qc.apply_gate('H', 0)
    qc.apply_gate('H', 1)
    
    # Apply CNOT (entanglement)
    qc.apply_gate('X', 1)  # Simplified CNOT
    
    # Measure
    result = qc.measure(0)
    print(f"\nMeasurement result: {result}")
    
    # Get probabilities
    probs = qc.get_probabilities()
    print(f"\nProbabilities (first 10 states):")
    for i in range(min(10, len(probs))):
        if probs[i] > 0.001:
            print(f"  |{i:08b}>: {probs[i]:.4f}")
    
    # Performance metrics
    metrics = qc.get_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Operations: {metrics['operations_performed']}")
    print(f"  Total time: {metrics['total_compute_time']:.4f}s")
    print(f"  Avg time/op: {metrics['avg_operation_time']:.6f}s")
    
    if medulla:
        medulla.stop_regulation()
