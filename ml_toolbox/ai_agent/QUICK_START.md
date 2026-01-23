# AI Agent Quick Start Guide

## 🚀 **Get Started in 5 Minutes**

### **Basic Usage**

```python
from ml_toolbox.ai_agent import MLCodeAgent

# Initialize agent
agent = MLCodeAgent()

# Build ML solution from natural language
result = agent.build("Classify iris flowers into 3 classes")

# Check results
if result['success']:
    print("✅ Code generated successfully!")
    print("\nGenerated Code:")
    print(result['code'])
    print("\nOutput:")
    print(result.get('output', ''))
else:
    print("❌ Generation failed:")
    print(result.get('error', 'Unknown error'))
```

---

## 📋 **Example Tasks**

### **1. Simple Classification**

```python
agent = MLCodeAgent()
result = agent.build("Create a classifier for binary classification")
print(result['code'])
```

### **2. Regression Task**

```python
result = agent.build("Build a regression model to predict house prices")
print(result['code'])
```

### **3. With Context**

```python
context = {
    'data_shape': (1000, 10),
    'target_type': 'classification',
    'classes': 3
}
result = agent.build("Train a model", context=context)
print(result['code'])
```

---

## 🔧 **Components**

### **1. Knowledge Base**

```python
from ml_toolbox.ai_agent import ToolboxKnowledgeBase

kb = ToolboxKnowledgeBase()

# Get capabilities
capabilities = kb.get_capabilities()

# Find solutions
solutions = kb.find_solution("classification")

# Get API docs
api_docs = kb.get_api_docs('MLToolbox')
```

### **2. Code Generator**

```python
from ml_toolbox.ai_agent import CodeGenerator

generator = CodeGenerator()

# Generate code
result = generator.generate("Train a classifier")
print(result['code'])
```

### **3. Code Sandbox**

```python
from ml_toolbox.ai_agent import CodeSandbox

sandbox = CodeSandbox()

# Execute code
result = sandbox.execute("""
from ml_toolbox import MLToolbox
import numpy as np

toolbox = MLToolbox()
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)
result = toolbox.fit(X, y)
print(f"Accuracy: {result['accuracy']:.2%}")
""")

print(f"Success: {result['success']}")
print(f"Output: {result.get('output', '')}")
```

---

## 🎯 **How It Works**

1. **Task Understanding** - Agent understands your task
2. **Knowledge Lookup** - Finds relevant toolbox capabilities
3. **Code Generation** - Generates Python code using LLM or templates
4. **Syntax Validation** - Validates code syntax
5. **Execution** - Executes code in safe sandbox
6. **Error Fixing** - Automatically fixes errors (up to 3 iterations)

---

## 📊 **Agent History**

```python
agent = MLCodeAgent()

# Build multiple solutions
agent.build("Classify data")
agent.build("Predict prices")

# Get history
history = agent.get_history()
for item in history:
    print(f"Task: {item['task']}")
    print(f"Success: {item['success']}")
    print(f"Iterations: {item['iterations']}")
```

---

## ⚙️ **Configuration**

### **With LLM (Default)**

```python
# Uses StandaloneQuantumLLM for code generation
agent = MLCodeAgent(use_llm=True)
```

### **Template-Based (Fallback)**

```python
# Uses code templates if LLM not available
agent = MLCodeAgent(use_llm=False)
```

### **Custom Max Iterations**

```python
# Allow more error-fixing iterations
agent = MLCodeAgent(max_iterations=5)
```

---

## 🎯 **Next Steps**

1. **Try Basic Tasks** - Start with simple classification/regression
2. **Add More Patterns** - Extend knowledge base with your patterns
3. **Improve Error Handling** - Add custom error fixers
4. **Add Task Planning** - Break down complex tasks

---

## 📚 **See Also**

- `AI_AGENT_ROADMAP.md` - Complete roadmap
- `knowledge_base.py` - Add your own patterns
- `code_generator.py` - Customize generation
- `code_sandbox.py` - Configure execution

---

**Ready to build ML code automatically!**
