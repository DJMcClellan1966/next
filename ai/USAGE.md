# Complete AI System - Usage Guide

## Quick Start

### 1. Import the System

```python
from complete_ai_system import CompleteAISystem
```

### 2. Create an Instance

```python
# With defaults
ai = CompleteAISystem()

# With custom configuration
from quantum_kernel import KernelConfig

config = KernelConfig(
    embedding_dim=512,
    cache_size=100000,
    enable_caching=True
)
ai = CompleteAISystem(config)
```

### 3. Use the System

```python
# Process various types of input
result = ai.process({
    "query": "your search query",
    "documents": ["document1", "document2"],
    "message": "conversation message",
    "premises": ["premise1", "premise2"],
    "question": "reasoning question"
})
```

## Common Use Cases

### Semantic Search

```python
result = ai.process({
    "query": "divine love",
    "documents": [
        "God is love",
        "Love is patient",
        "Faith, hope, and love"
    ]
})

# Access results
for item in result["search"]["results"]:
    print(f"{item['text']}: {item['similarity']:.3f}")
```

### Understanding Intent

```python
intent = ai.understanding.understand_intent(
    "I want to search for information",
    context=["previous conversation"]
)

print(f"Intent: {intent['intent']}")
print(f"Confidence: {intent['confidence']:.3f}")
```

### Building Knowledge Graphs

```python
graph = ai.knowledge_graph.build_graph([
    "Document 1",
    "Document 2",
    "Document 3"
])

print(f"Nodes: {len(graph['nodes'])}")
print(f"Edges: {len(graph['edges'])}")
print(f"Themes: {len(graph['themes'])}")
```

### Reasoning

```python
reasoning = ai.reasoning.reason(
    premises=["Premise 1", "Premise 2"],
    question="What follows?"
)

print(f"Connections: {len(reasoning['connections'])}")
print(f"Confidence: {reasoning['confidence']:.3f}")
```

### Learning from Examples

```python
learning = ai.learning.learn_from_examples([
    ("input1", "output1"),
    ("input2", "output2")
])

print(f"Patterns learned: {learning['patterns_learned']}")
```

### Conversation

```python
response = ai.conversation.respond("Hello, how are you?")
print(response)
```

## Advanced Usage

### Using Individual Components

You can use components independently:

```python
from complete_ai_system import (
    SemanticUnderstandingEngine,
    IntelligentSearch,
    KnowledgeGraphBuilder
)
from quantum_kernel import get_kernel, KernelConfig

kernel = get_kernel(KernelConfig())

understanding = SemanticUnderstandingEngine(kernel)
search = IntelligentSearch(kernel)
graph_builder = KnowledgeGraphBuilder(kernel)
```

### Customizing Components

```python
# Add custom intents
ai.understanding.add_intent("custom intent")

# Add custom response templates
ai.conversation.add_response_template("custom intent", "Custom response")

# Get learned patterns
patterns = ai.learning.get_patterns()
```

### System Statistics

```python
# Get system statistics
stats = ai.get_stats()
print(stats)

# Get kernel statistics
kernel_stats = ai.kernel.get_stats()
print(kernel_stats)
```

### Resetting the System

```python
# Reset everything (caches, history, patterns)
ai.reset()
```

## Best Practices

1. **Reuse the same instance** - The system caches embeddings, so reuse improves performance
2. **Use appropriate cache size** - Set cache_size based on your data volume
3. **Clear history when needed** - Use `conversation.clear_history()` for new conversations
4. **Monitor statistics** - Check `get_stats()` to monitor performance

## Performance Tips

- **Enable caching** - Always enable caching for better performance
- **Batch operations** - Process multiple items together when possible
- **Reuse embeddings** - The system automatically caches embeddings
- **Monitor cache hits** - Check statistics to see cache effectiveness

## Examples

See `examples.py` for complete working examples.
