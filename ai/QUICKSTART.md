# Complete AI System - Quick Start

## Installation

The system is ready to use! Just ensure `quantum_kernel` is available.

## Basic Usage

```python
from complete_ai_system import CompleteAISystem

# Create system
ai = CompleteAISystem()

# Use it
result = ai.process({
    "query": "divine love",
    "documents": ["God is love", "Love is patient"]
})

print(result["search"]["results"])
```

## Using Individual Components

```python
from complete_ai_system import IntelligentSearch
from quantum_kernel import get_kernel, KernelConfig

kernel = get_kernel(KernelConfig())
search = IntelligentSearch(kernel)

results = search.search("query", ["item1", "item2"])
```

## Examples

Run examples:
```bash
python -m complete_ai_system.examples
```

## Tests

Run tests:
```bash
python -m complete_ai_system.test_system
```

## Documentation

- `README.md` - Full documentation
- `USAGE.md` - Detailed usage guide
- `examples.py` - Working examples
- `test_system.py` - Test suite

## Features

✅ Semantic Understanding
✅ Knowledge Graphs
✅ Intelligent Search
✅ Reasoning
✅ Learning
✅ Conversation

All built on the quantum kernel foundation!
