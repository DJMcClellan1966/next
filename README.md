# ML Toolbox

**A comprehensive, production-ready machine learning toolbox organized into four compartments: Data, Infrastructure, Algorithms, and MLOps.**

## Overview

ML Toolbox is a complete end-to-end ML platform that combines:
- **Advanced data preprocessing** with quantum-inspired methods
- **AI infrastructure** with semantic understanding and LLM capabilities
- **Comprehensive algorithms** from classical ML to deep learning
- **Production MLOps** with monitoring, deployment, and A/B testing

## Architecture

The toolbox is organized into **four compartments**:

1. **Compartment 1: Data** - Preprocessing, validation, transformation
2. **Compartment 2: Infrastructure** - Quantum Kernel, AI Components, LLM
3. **Compartment 3: Algorithms** - Models, evaluation, tuning, ensembles
4. **Compartment 4: MLOps** - Production deployment, monitoring, A/B testing, experiment tracking

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
from ml_toolbox import MLToolbox

# Initialize toolbox
toolbox = MLToolbox(include_mlops=True)

# Compartment 1: Data Preprocessing
results = toolbox.data.preprocess(
    texts=["text1", "text2", "text3"],
    advanced=True,
    enable_compression=True
)

# Compartment 2: Infrastructure
kernel = toolbox.infrastructure.get_kernel()
ai_system = toolbox.infrastructure.get_ai_system(use_llm=True)

# Compartment 3: Algorithms
evaluator = toolbox.algorithms.get_evaluator()
tuner = toolbox.algorithms.get_tuner()

# Compartment 4: MLOps
monitor = toolbox.mlops.get_model_monitor(model, X_train, y_train)
registry = toolbox.mlops.get_model_registry()
server = toolbox.mlops.get_model_server(registry)
```

## Features

### Compartment 1: Data
- Advanced data preprocessing (quantum + safety filtering)
- Data scrubbing and normalization
- Dimensionality reduction
- Automatic feature creation

### Compartment 2: Infrastructure
- Quantum Kernel (semantic embeddings, similarity)
- Complete AI System (understanding, knowledge graphs, search)
- Quantum LLM (grounded generation, progressive learning)
- Adaptive Neurons

### Compartment 3: Algorithms
- ML evaluation (cross-validation, metrics)
- Hyperparameter tuning
- Ensemble learning
- Statistical learning methods
- Methods from ESL, Bishop, and Deep Learning books
- Interpretability (SHAP, LIME, PDP)
- Fairness and bias detection
- Time series (ARIMA, feature engineering)
- Active learning
- Causal discovery

### Compartment 4: MLOps
- Model monitoring (drift detection, performance tracking)
- Model deployment (REST API, batch/real-time inference)
- A/B testing framework
- Experiment tracking
- Canary deployments

## Requirements

See `requirements.txt` for complete list. Core dependencies:
- Python 3.8+
- numpy>=1.26.0
- scipy>=1.11.0
- scikit-learn>=1.5.0
- torch>=2.3.0
- sentence-transformers>=2.2.0
- fastapi>=0.100.0 (for MLOps)
- uvicorn>=0.23.0 (for MLOps)

## License

MIT
