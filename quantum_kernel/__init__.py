"""
Quantum Kernel - Universal Processing Layer
Reusable kernel for any application requiring semantic understanding,
similarity computation, and relationship discovery.
"""
from .kernel import QuantumKernel, KernelConfig, get_kernel, reset_kernel

__all__ = ['QuantumKernel', 'KernelConfig', 'get_kernel', 'reset_kernel']
__version__ = '1.0.0'
