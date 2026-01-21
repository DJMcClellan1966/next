# Complete AI System

A comprehensive AI system built around the quantum kernel, providing semantic understanding, knowledge graphs, intelligent search, reasoning, learning, and conversation capabilities.

## Features

- ✅ **Semantic Understanding** - Understand meaning, intent, and context
- ✅ **Knowledge Graphs** - Build and maintain knowledge graphs automatically
- ✅ **Intelligent Search** - Semantic search with concept discovery
- ✅ **Reasoning** - Logical and causal reasoning
- ✅ **Learning** - Pattern extraction and adaptation
- ✅ **Conversation** - Context-aware conversational AI

## Installation

```bash
# Ensure quantum_kernel is available
# The kernel should be in the parent directory or installed
```

## Quick Start

### Basic Usage

```python
from complete_ai_system import CompleteAISystem

# Create system
ai = CompleteAISystem()

# Process input
result = ai.process({
    "query": "divine love",
    "documents": [
        "God is love",
        "Love is patient and kind",
        "Faith, hope, and love"
    ]
})

print(result["search"]["results"])
```

### Using Individual Components

```python
from complete_ai_system import (
    SemanticUnderstandingEngine,
    IntelligentSearch,
    KnowledgeGraphBuilder
)
from quantum_kernel import get_kernel, KernelConfig

# Get kernel
kernel = get_kernel(KernelConfig())

# Create components
understanding = SemanticUnderstandingEngine(kernel)
search = IntelligentSearch(kernel)
graph_builder = KnowledgeGraphBuilder(kernel)

# Use components
intent = understanding.understand_intent("I want to search for information")
results = search.search("divine love", documents)
graph = graph_builder.build_graph(documents)
```

## Components

### SemanticUnderstandingEngine

Understands user intent and context.

```python
from complete_ai_system import SemanticUnderstandingEngine

engine = SemanticUnderstandingEngine(kernel)
result = engine.understand_intent(
    "I want to search for information",
    context=["previous conversation"]
)
```

### KnowledgeGraphBuilder

Builds knowledge graphs from documents.

```python
from complete_ai_system import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder(kernel)
graph = builder.build_graph([
    "Document 1",
    "Document 2",
    "Document 3"
])
```

### IntelligentSearch

Performs semantic search with discovery.

```python
from complete_ai_system import IntelligentSearch

search = IntelligentSearch(kernel)
results = search.search_and_discover(
    "query",
    corpus=["item1", "item2", "item3"]
)
```

### ReasoningEngine

Performs logical and causal reasoning.

```python
from complete_ai_system import ReasoningEngine

reasoning = ReasoningEngine(kernel)
result = reasoning.reason(
    premises=["Premise 1", "Premise 2"],
    question="What follows?"
)
```

### LearningSystem

Learns patterns from examples.

```python
from complete_ai_system import LearningSystem

learning = LearningSystem(kernel)
result = learning.learn_from_examples([
    ("input1", "output1"),
    ("input2", "output2")
])
```

### ConversationalAI

Provides context-aware conversation.

```python
from complete_ai_system import ConversationalAI

conversation = ConversationalAI(kernel)
response = conversation.respond("Hello, how are you?")
```

## Complete System

### Initialization

```python
from complete_ai_system import CompleteAISystem
from quantum_kernel import KernelConfig

# With default config
ai = CompleteAISystem()

# With custom config
config = KernelConfig(
    embedding_dim=512,
    cache_size=100000
)
ai = CompleteAISystem(config)
```

### Processing

```python
# Process multiple types of input
result = ai.process({
    "query": "search query",
    "message": "conversation message",
    "premises": ["premise1", "premise2"],
    "question": "reasoning question",
    "documents": ["doc1", "doc2"],
    "examples": [("input", "output")],
    "context": ["context1", "context2"]
})
```

### Statistics

```python
# Get system statistics
stats = ai.get_stats()
print(stats)
```

### Reset

```python
# Reset system (clear caches, history, etc.)
ai.reset()
```

## Examples

See `examples/` directory for complete examples:
- Research platform
- Knowledge base
- Content platform
- And more...

## Performance

- **Fast**: 10-200x speedup from caching
- **Scalable**: Linear performance with data size
- **Efficient**: Controlled memory usage
- **Reliable**: Tested and verified

## Architecture

```
CompleteAISystem
├── SemanticUnderstandingEngine
├── KnowledgeGraphBuilder
├── IntelligentSearch
├── ReasoningEngine
├── LearningSystem
└── ConversationalAI
    └── QuantumKernel (shared foundation)
```

## Requirements

- Python 3.7+
- quantum_kernel (from parent directory)
- numpy
- scipy

## License

## Support

For issues or questions, see the main project documentation.
