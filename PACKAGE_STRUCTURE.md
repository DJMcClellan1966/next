# ML Toolbox Package Structure

## Package Organization

The ML Toolbox is now organized as a proper Python package that can be installed and imported.

## Installation

```bash
# Install in development mode (recommended)
pip install -e .

# Or install normally
pip install .
```

## Package Structure

```
ml_toolbox/
├── __init__.py                    # Main package file with MLToolbox class
├── compartment1_data/             # Data preprocessing compartment
├── compartment2_infrastructure.py # Infrastructure compartment
├── compartment3_algorithms.py     # Algorithms compartment
├── compartment4_mlops.py         # MLOps compartment
├── advanced/                      # Advanced ML features
├── ai_agent/                      # Super Power Agent
├── ai_agents/                     # LLM+RAG+KG Agents
├── agentic_systems/               # Agentic Systems
├── multi_agent_design/            # Multi-Agent Design
├── agent_brain/                   # Brain-like features
├── agent_enhancements/            # Agent enhancements
├── agent_fundamentals/            # Agent fundamentals
├── agent_pipelines/               # Agent pipelines
├── ai_concepts/                   # AI concepts
├── core_models/                   # Core ML models
├── math_foundations/              # Mathematical foundations
├── textbook_concepts/             # Textbook concepts
├── generative_ai_patterns/        # Generative AI patterns
├── framework_integration/         # Framework integration
├── llm_engineering/               # LLM engineering
├── optimization_kernels/          # Optimization kernels
├── computational_kernels/         # Computational kernels
├── automl/                        # AutoML
├── deployment/                    # Deployment
├── infrastructure/                # Infrastructure
├── models/                        # Models
├── security/                      # Security
├── testing/                       # Testing
└── ui/                            # UI components
```

## Usage

### Basic Import

```python
from ml_toolbox import MLToolbox

# Create instance
toolbox = MLToolbox()

# Use it
result = toolbox.fit(X, y)
predictions = toolbox.predict(result['model'], X_test)
```

### Natural Language ML

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()
response = toolbox.chat("Classify this data", X, y)
```

### Advanced Features

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Access compartments
data_comp = toolbox.data
infra_comp = toolbox.infrastructure
algo_comp = toolbox.algorithms
mlops_comp = toolbox.mlops

# Access agents
super_agent = toolbox.super_power_agent
llm_agent = toolbox.llm_rag_kg_agent
```

## Optional Dependencies

The package works with core dependencies (numpy, scikit-learn, pandas) but has optional dependencies for advanced features:

```bash
# Install with all optional dependencies
pip install -e ".[full]"

# Or install specific groups
pip install -e ".[advanced,deep-learning,nlp]"
```

## Testing Installation

Run the test script:

```bash
python test_package_installation.py
```

This will verify:
- ✅ Package can be imported
- ✅ MLToolbox can be instantiated
- ✅ Basic ML functionality works

## Development

When developing, use:

```bash
pip install -e .
```

This installs in "editable" mode, so changes to the code are immediately available without reinstalling.

## Package Files

- `setup.py` - Setuptools configuration
- `pyproject.toml` - Modern Python packaging configuration
- `MANIFEST.in` - Files to include in package
- `requirements.txt` - Core dependencies
- `INSTALLATION.md` - Detailed installation guide

## Troubleshooting

### Import Errors

If you get import errors for modules in the parent directory (like `dependency_manager`, `error_handler`, `ml_math_optimizer`, `medulla_toolbox_optimizer`), these are optional dependencies. The toolbox will work without them, but some features may be limited.

### Module Not Found

Make sure you've installed the package:
```bash
pip install -e .
```

### Development Mode

Always use `pip install -e .` when developing so changes are immediately available.
