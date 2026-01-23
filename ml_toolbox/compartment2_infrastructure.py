"""
Compartment 2: Infrastructure
Kernels, AI components, LLM, and supporting infrastructure
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class InfrastructureCompartment:
    """
    Compartment 2: Infrastructure
    
    Components for AI infrastructure:
    - Quantum Kernel: Semantic understanding and embeddings
    - PocketFence Kernel: Safety filtering (external service)
    - AI Components: Understanding, knowledge graph, search, reasoning
    - LLM: Quantum-inspired language models
    - Adaptive Neuron: Neural-like learning components
    """
    
    def __init__(self, medulla=None):
        """
        Initialize Infrastructure Compartment
        
        Args:
            medulla: Optional Medulla Oblongata System for resource regulation
        """
        self.components = {}
        self.medulla = medulla
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all infrastructure compartment components"""
        
        # Quantum Kernel
        try:
            from quantum_kernel import QuantumKernel, KernelConfig, get_kernel
            self.components['QuantumKernel'] = QuantumKernel
            self.components['KernelConfig'] = KernelConfig
            self.components['get_kernel'] = get_kernel
        except ImportError as e:
            print(f"Warning: Could not import Quantum Kernel: {e}")
        
        # AI Components
        try:
            from ai.core import CompleteAISystem
            from ai.components import (
                SemanticUnderstandingEngine,
                KnowledgeGraphBuilder,
                IntelligentSearch,
                ReasoningEngine,
                LearningSystem,
                ConversationalAI
            )
            self.components['CompleteAISystem'] = CompleteAISystem
            self.components['SemanticUnderstandingEngine'] = SemanticUnderstandingEngine
            self.components['KnowledgeGraphBuilder'] = KnowledgeGraphBuilder
            self.components['IntelligentSearch'] = IntelligentSearch
            self.components['ReasoningEngine'] = ReasoningEngine
            self.components['LearningSystem'] = LearningSystem
            self.components['ConversationalAI'] = ConversationalAI
        except ImportError as e:
            print(f"Warning: Could not import AI components: {e}")
        
        # LLM
        try:
            from llm.quantum_llm_standalone import StandaloneQuantumLLM
            self.components['StandaloneQuantumLLM'] = StandaloneQuantumLLM
        except ImportError as e:
            print(f"Warning: Could not import LLM: {e}")
        
        # Adaptive Neuron
        try:
            from ai.adaptive_neuron import AdaptiveNeuron, AdaptiveNeuralNetwork
            self.components['AdaptiveNeuron'] = AdaptiveNeuron
            self.components['AdaptiveNeuralNetwork'] = AdaptiveNeuralNetwork
        except ImportError as e:
            print(f"Warning: Could not import Adaptive Neuron: {e}")
        
        # Medulla Oblongata System (resource regulation)
        try:
            from medulla_oblongata_system import MedullaOblongataSystem
            self.components['MedullaOblongataSystem'] = MedullaOblongataSystem
            if self.medulla:
                self.components['medulla'] = self.medulla  # Instance if available
        except ImportError as e:
            print(f"Warning: Could not import Medulla Oblongata System: {e}")
        
        # Virtual Quantum Computer
        try:
            from virtual_quantum_computer import VirtualQuantumComputer
            self.components['VirtualQuantumComputer'] = VirtualQuantumComputer
        except ImportError as e:
            print(f"Warning: Could not import Virtual Quantum Computer: {e}")
        
        # Knuth Knowledge Graph (for knowledge graph operations)
        try:
            from knuth_ml_integrations import KnuthKnowledgeGraph
            self.components['KnuthKnowledgeGraph'] = KnuthKnowledgeGraph
        except ImportError as e:
            print(f"Warning: Could not import Knuth knowledge graph: {e}")
        
        # Add component descriptions
        self.component_descriptions = {
            'QuantumKernel': {
                'description': 'Universal processing layer for semantic understanding',
                'features': [
                    'Semantic embeddings',
                    'Similarity computation',
                    'Relationship discovery',
                    'Theme discovery',
                    'Parallel processing',
                    'Caching'
                ],
                'location': 'quantum_kernel/kernel.py',
                'category': 'Kernel'
            },
            'CompleteAISystem': {
                'description': 'Complete AI system integrating all components',
                'features': [
                    'Semantic understanding',
                    'Knowledge graph building',
                    'Intelligent search',
                    'Reasoning',
                    'Learning',
                    'Conversation'
                ],
                'location': 'ai/core.py',
                'category': 'AI System'
            },
            'StandaloneQuantumLLM': {
                'description': 'Quantum-inspired language model',
                'features': [
                    'Text generation',
                    'Grounded generation',
                    'Progressive learning',
                    'Bias detection',
                    'Quantum sampling'
                ],
                'location': 'llm/quantum_llm_standalone.py',
                'category': 'LLM'
            },
            'AdaptiveNeuron': {
                'description': 'Neural-like component with adaptive learning',
                'features': [
                    'Semantic learning',
                    'Adaptive weights',
                    'Reinforcement learning',
                    'Concept associations'
                ],
                'location': 'ai/adaptive_neuron.py',
                'category': 'Neural Component'
            },
            'AdaptiveNeuralNetwork': {
                'description': 'Network of specialized adaptive neurons',
                'features': [
                    'Multiple specialized neurons',
                    'Network coordination',
                    'Distributed learning'
                ],
                'location': 'ai/adaptive_neuron.py',
                'category': 'Neural Network'
            }
        }
    
    def get_kernel(self, config=None):
        """
        Get a Quantum Kernel instance
        
        Args:
            config: Optional KernelConfig, uses defaults if None
            
        Returns:
            QuantumKernel instance
        """
        if 'get_kernel' in self.components:
            return self.components['get_kernel'](config)
        else:
            raise ImportError("Quantum Kernel not available")
    
    def get_ai_system(self, use_llm: bool = False):
        """
        Get a Complete AI System instance
        
        Args:
            use_llm: Whether to enable LLM integration
            
        Returns:
            CompleteAISystem instance
        """
        if 'CompleteAISystem' in self.components:
            return self.components['CompleteAISystem'](use_llm=use_llm)
        else:
            raise ImportError("AI System not available")
    
    def list_components(self):
        """List all available components in this compartment"""
        print("="*80)
        print("COMPARTMENT 2: INFRASTRUCTURE")
        print("="*80)
        print("\nComponents:")
        for name, component in self.components.items():
            desc = self.component_descriptions.get(name, {})
            print(f"\n{name}:")
            print(f"  Description: {desc.get('description', 'N/A')}")
            print(f"  Location: {desc.get('location', 'N/A')}")
            print(f"  Category: {desc.get('category', 'N/A')}")
            if 'features' in desc:
                print(f"  Features:")
                for feature in desc['features']:
                    print(f"    - {feature}")
        print("\n" + "="*80)
    
    def get_info(self):
        """Get information about this compartment"""
        return {
            'name': 'Infrastructure Compartment',
            'description': 'Kernels, AI components, LLM, and supporting infrastructure',
            'components': list(self.components.keys()),
            'component_count': len(self.components)
        }
