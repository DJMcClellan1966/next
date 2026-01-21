# Quantum-Inspired AI Platform

**Personal experiments blending classical ML/LLMs with quantum-inspired kernels for semantic understanding, similarity computation, and intelligent language generation.**

## Current Focus

Exploring quantum kernel methods (QSVM variants, quantum amplitude encoding, quantum interference similarity) alongside LLM prompting and fine-tuning basics to create a unified AI system capable of deep semantic understanding and grounded text generation.

## Components

### 🔬 Quantum Kernel (`quantum_kernel/`)
Universal processing layer for semantic embeddings, similarity computation, and relationship discovery. Features quantum-inspired methods including amplitude encoding, interference-based similarity, and entangled relationship discovery.

### 🤖 AI System (`ai/`)
Complete AI system built around the quantum kernel, including semantic understanding, knowledge graph building, intelligent search, reasoning, and conversational AI capabilities.

### 💬 LLM (`llm/`)
Quantum-inspired large language models with grounded generation, progressive learning, and quantum sampling techniques for more natural text generation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Basic usage
from quantum_kernel import get_kernel, KernelConfig
from ai import CompleteAISystem
from llm.quantum_llm_standalone import StandaloneQuantumLLM

# Initialize kernel with quantum methods
config = KernelConfig(
    use_sentence_transformers=True,
    use_quantum_methods=True,
    similarity_metric='quantum'
)
kernel = get_kernel(config)

# Use with AI system or LLM
ai_system = CompleteAISystem(config=config, use_llm=True)
llm = StandaloneQuantumLLM(kernel=kernel)
```

## Requirements

- Python 3.8+
- numpy, scipy
- torch (for LLM)
- sentence-transformers (optional, for improved embeddings)
- scikit-learn (optional, for theme extraction)

## License

MIT
