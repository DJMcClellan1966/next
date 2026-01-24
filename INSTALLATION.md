# Installation Guide

## Installing ML Toolbox as a Package

### Option 1: Install in Development Mode (Recommended for Development)

This allows you to edit the code and see changes immediately:

```bash
# From the project root directory
pip install -e .
```

### Option 2: Install from Source

```bash
# From the project root directory
pip install .
```

### Option 3: Install with Optional Dependencies

```bash
# Install with all optional dependencies
pip install -e ".[full]"

# Or install specific optional dependencies
pip install -e ".[advanced,deep-learning,nlp]"
```

### Option 4: Install with Requirements File

```bash
# Install core dependencies
pip install -r requirements.txt

# Then install package in development mode
pip install -e .
```

## Verifying Installation

After installation, verify it works:

```python
# Test basic import
from ml_toolbox import MLToolbox

# Create instance
toolbox = MLToolbox()

# Test basic functionality
print("✅ ML Toolbox installed successfully!")
```

## Troubleshooting

### Import Errors

If you get import errors for modules in the parent directory (like `dependency_manager`, `error_handler`), these are optional dependencies. The toolbox will work without them, but some features may be limited.

### Development Mode

When developing, use `pip install -e .` so changes to the code are immediately available without reinstalling.

### Package Structure

The package is organized as:
```
ml_toolbox/
├── __init__.py          # Main MLToolbox class
├── compartment1_data/   # Data preprocessing
├── compartment2_infrastructure/  # Infrastructure
├── compartment3_algorithms/      # Algorithms
├── compartment4_mlops.py         # MLOps
├── ai_agent/            # Super Power Agent
├── ai_agents/           # LLM+RAG+KG Agents
├── agentic_systems/     # Agentic Systems
├── multi_agent_design/   # Multi-Agent Design
└── ...                  # Other modules
```

## Next Steps

After installation, see:
- `examples/complete_workflow_example.py` - Working examples
- `WHERE_TO_GO_FROM_HERE.md` - Next steps guide
- `PRACTICAL_NEXT_STEPS_ROADMAP.md` - Detailed roadmap
