# Machine Learning Toolbox

## Overview

A comprehensive machine learning toolbox organized into **three compartments**:

1. **Compartment 1: Data** - Preprocessing, validation, transformation
2. **Compartment 2: Infrastructure** - Kernels, AI components, LLM
3. **Compartment 3: Algorithms** - Models, evaluation, tuning, ensembles

---

## Architecture

```
MLToolbox
├── Compartment 1: Data
│   ├── AdvancedDataPreprocessor ⭐ (Quantum + PocketFence)
│   ├── ConventionalPreprocessor
│   └── Data utilities
│
├── Compartment 2: Infrastructure
│   ├── Quantum Kernel
│   ├── AI Components (Understanding, Knowledge Graph, Search, Reasoning)
│   ├── LLM (StandaloneQuantumLLM)
│   └── Adaptive Neuron/Network
│
└── Compartment 3: Algorithms
    ├── ML Evaluation (MLEvaluator)
    ├── Hyperparameter Tuning (HyperparameterTuner)
    └── Ensemble Learning (EnsembleLearner)
```

---

## Quick Start

### Basic Usage

```python
from ml_toolbox import MLToolbox

# Initialize toolbox
toolbox = MLToolbox()

# Access compartments
data = toolbox.data
infrastructure = toolbox.infrastructure
algorithms = toolbox.algorithms
```

### Preprocess Data (Compartment 1)

```python
# Use AdvancedDataPreprocessor from Data compartment
texts = ["text1", "text2", "text3"]

# Preprocess
results = data.preprocess(
    texts,
    advanced=True,  # Use AdvancedDataPreprocessor
    dedup_threshold=0.85,
    enable_compression=True
)

# Get preprocessed features
X = results['compressed_embeddings']
```

### Use Infrastructure (Compartment 2)

```python
# Get Quantum Kernel
kernel = infrastructure.get_kernel()

# Get AI System
ai_system = infrastructure.get_ai_system(use_llm=True)

# Use for semantic understanding
embedding = kernel.embed("Python programming")
similarity = kernel.similarity("Python", "programming")
```

### Evaluate Models (Compartment 3)

```python
# Get evaluator
evaluator = algorithms.get_evaluator()

# Evaluate model
results = evaluator.evaluate_model(
    model=your_model,
    X=X_train,
    y=y_train,
    cv=5
)

# Get tuner
tuner = algorithms.get_tuner()

# Tune hyperparameters
best_params = tuner.tune(
    model=your_model,
    X=X_train,
    y=y_train,
    param_grid={'C': [0.1, 1, 10]}
)
```

---

## Compartment 1: Data

### Components

- **AdvancedDataPreprocessor** ⭐
  - Quantum Kernel + PocketFence Kernel
  - Safety filtering, semantic deduplication, categorization
  - Quality scoring, dimensionality reduction
  - Automatic feature creation

- **ConventionalPreprocessor**
  - Basic preprocessing
  - Exact duplicate removal
  - Keyword-based categorization

### Usage

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Preprocess data
results = toolbox.data.preprocess(
    texts,
    advanced=True,
    dedup_threshold=0.85,
    enable_compression=True,
    compression_ratio=0.5
)

# Access results
X = results['compressed_embeddings']  # Features
categories = results['categorized']     # Categories
quality = results['quality_scores']   # Quality metrics
```

### Why AdvancedDataPreprocessor is in Data Compartment

**AdvancedDataPreprocessor belongs in Compartment 1 (Data)** because it:
- ✅ **Preprocesses raw data** before it enters models
- ✅ **Transforms text** into numerical features
- ✅ **Cleans and validates** data quality
- ✅ **Creates features** automatically
- ✅ **Prepares data** for machine learning

It's a **data preprocessing tool**, not an algorithm or infrastructure component.

---

## Compartment 2: Infrastructure

### Components

- **Quantum Kernel**
  - Semantic embeddings
  - Similarity computation
  - Relationship discovery

- **Quantum AI** ⭐
  - CompleteAISystem (main AI system)
  - SemanticUnderstandingEngine (understand intent)
  - KnowledgeGraphBuilder (build knowledge graphs)
  - IntelligentSearch (semantic search)
  - ReasoningEngine (logical reasoning)
  - LearningSystem (continuous learning)
  - ConversationalAI (conversational interface)

- **LLM** ⭐
  - StandaloneQuantumLLM
  - Text generation
  - Grounded generation
  - Progressive learning

- **Adaptive Neuron/Network**
  - AdaptiveNeuron
  - AdaptiveNeuralNetwork

### Usage

```python
# Get kernel
kernel = toolbox.infrastructure.get_kernel()

# Get Quantum AI system
ai = toolbox.infrastructure.get_ai_system(use_llm=True)

# Use Quantum AI for understanding
understanding = ai.understanding.understand_intent("What is Python?")
search_results = ai.search.search("Python programming", corpus)

# Use LLM for text generation
from llm.quantum_llm_standalone import StandaloneQuantumLLM
llm = StandaloneQuantumLLM(kernel=kernel)
generated = llm.generate_grounded("Explain machine learning", max_length=100)

# Use for semantic operations
embedding = kernel.embed("text")
similarity = kernel.similarity("text1", "text2")
```

### Why Quantum AI and LLM are in Infrastructure

**Quantum AI** and **LLM** belong in **Compartment 2: Infrastructure** because:
- ✅ They provide **AI services** (understanding, search, generation)
- ✅ They use **Quantum Kernel** as infrastructure
- ✅ They're **not data preprocessing** (that's Compartment 1)
- ✅ They're **not ML algorithms** (that's Compartment 3)
- ✅ They provide **infrastructure** for AI operations

See `QUANTUM_AI_LLM_PLACEMENT.md` for detailed explanation.

---

## Compartment 3: Algorithms

### Components

- **MLEvaluator**
  - Cross-validation
  - Multiple metrics
  - Overfitting detection

- **HyperparameterTuner**
  - Grid search
  - Random search

- **EnsembleLearner**
  - Voting, bagging, boosting, stacking

### Usage

```python
# Evaluate model
evaluator = toolbox.algorithms.get_evaluator()
results = evaluator.evaluate_model(model, X, y)

# Tune hyperparameters
tuner = toolbox.algorithms.get_tuner()
best_params = tuner.tune(model, X, y, param_grid)

# Create ensemble
ensemble = toolbox.algorithms.get_ensemble()
ensemble.fit(X, y)
```

---

## Complete Workflow Example

```python
from ml_toolbox import MLToolbox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize toolbox
toolbox = MLToolbox()

# Step 1: Preprocess data (Compartment 1)
texts = ["text1", "text2", ...]
labels = [0, 1, ...]

results = toolbox.data.preprocess(
    texts,
    advanced=True,
    enable_compression=True
)

X = results['compressed_embeddings']
y = labels[:len(X)]

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 3: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Evaluate (Compartment 3)
evaluator = toolbox.algorithms.get_evaluator()
evaluation_results = evaluator.evaluate_model(
    model=model,
    X=X_train,
    y=y_train,
    cv=5
)

print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
print(f"Precision: {evaluation_results['precision']:.4f}")
print(f"Recall: {evaluation_results['recall']:.4f}")

# Step 5: Tune hyperparameters (Compartment 3)
tuner = toolbox.algorithms.get_tuner()
best_params = tuner.tune(
    model=RandomForestClassifier(),
    X=X_train,
    y=y_train,
    param_grid={
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None]
    }
)

print(f"Best parameters: {best_params}")
```

---

## Component Placement

### ✅ AdvancedDataPreprocessor → Compartment 1: Data

**Why:**
- Preprocesses raw data
- Transforms text to features
- Cleans and validates data
- Prepares data for ML models

**Not in:**
- ❌ Compartment 2 (Infrastructure) - It's not infrastructure, it's data processing
- ❌ Compartment 3 (Algorithms) - It's not an algorithm, it's preprocessing

### ✅ Quantum Kernel → Compartment 2: Infrastructure

**Why:**
- Provides semantic understanding infrastructure
- Used by other components
- Core infrastructure component

### ✅ ML Evaluation → Compartment 3: Algorithms

**Why:**
- Evaluates ML algorithms
- Part of algorithm workflow
- Algorithm-related functionality

---

## Benefits of Compartment Organization

### ✅ **Clear Separation of Concerns**

- **Data**: Preprocessing and transformation
- **Infrastructure**: Core components and services
- **Algorithms**: Models and evaluation

### ✅ **Easy to Navigate**

- Know where to find components
- Understand component relationships
- Clear workflow

### ✅ **Modular Design**

- Each compartment is independent
- Can use compartments separately
- Easy to extend

### ✅ **Logical Grouping**

- Related components together
- Clear purpose for each compartment
- Intuitive organization

---

## Summary

### Compartment 1: Data
- **AdvancedDataPreprocessor** ⭐ (main component)
- ConventionalPreprocessor
- Data preprocessing and transformation

### Compartment 2: Infrastructure
- Quantum Kernel
- AI Components
- LLM
- Adaptive Neuron/Network

### Compartment 3: Algorithms
- ML Evaluation
- Hyperparameter Tuning
- Ensemble Learning

### Key Point

**AdvancedDataPreprocessor is correctly placed in Compartment 1 (Data)** because it preprocesses and transforms data before it enters machine learning models.

---

## Usage Tips

1. **Start with Data Compartment** - Preprocess your data first
2. **Use Infrastructure as Needed** - For semantic operations
3. **Apply Algorithms** - Evaluate and tune your models
4. **Combine Compartments** - Use all three for complete workflow

---

## Documentation

- **Data Compartment**: See `AUTOMATIC_FEATURE_CREATION_GUIDE.md`
- **Infrastructure**: See `quantum_kernel/README.md`, `ai/README.md`
- **Algorithms**: See `ML_EVALUATION_GUIDE.md`, `ENSEMBLE_LEARNING_GUIDE.md`
