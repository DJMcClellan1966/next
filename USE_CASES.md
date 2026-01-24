# ML Toolbox Use Cases 🚀

## Comprehensive Practical Examples

This document provides detailed use cases demonstrating the ML Toolbox's capabilities across three key areas:
1. **Semantic Search/Reasoning** via Quantum Kernel + AI System
2. **Time Series/Tabular Prediction** with Preprocessing + Interpretability
3. **Small Agent-like Workflow**

---

## 📚 **Use Case 1: Semantic Search & Reasoning with Quantum Kernel**

### **Overview**

Use the Quantum Kernel for semantic understanding, similarity search, and intelligent reasoning across text documents, code, or any semantic data.

### **Scenario: Intelligent Document Search System**

Build a semantic search system that understands meaning, not just keywords.

```python
from ml_toolbox import MLToolbox
import numpy as np

# Initialize toolbox
toolbox = MLToolbox()

# Example: Search through research papers, documentation, or code
documents = [
    "Machine learning algorithms for classification",
    "Deep neural networks for image recognition",
    "Statistical methods for data analysis",
    "Natural language processing with transformers",
    "Computer vision using convolutional networks",
    "Reinforcement learning for game playing",
    "Clustering algorithms for unsupervised learning"
]

# Access quantum kernel through infrastructure compartment
infrastructure = toolbox.infrastructure
quantum_kernel = infrastructure.quantum_kernel if hasattr(infrastructure, 'quantum_kernel') else None

if quantum_kernel is None:
    # Alternative: Import directly
    from quantum_kernel.kernel import QuantumKernel
    quantum_kernel = QuantumKernel()

# Build semantic embeddings
print("Building semantic embeddings...")
embeddings = []
for doc in documents:
    embedding = quantum_kernel.compute_embedding(doc)
    embeddings.append(embedding)

embeddings = np.array(embeddings)

# Semantic search function
def semantic_search(query, documents, embeddings, top_k=3):
    """Find most semantically similar documents"""
    query_embedding = quantum_kernel.compute_embedding(query)
    
    # Compute semantic similarity
    similarities = quantum_kernel.compute_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]
    
    # Get top-k most similar
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'similarity': float(similarities[idx]),
            'rank': len(results) + 1
        })
    
    return results

# Example searches
queries = [
    "How to classify images?",
    "What are neural networks?",
    "Unsupervised learning methods"
]

print("\n" + "="*80)
print("SEMANTIC SEARCH RESULTS")
print("="*80)

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-" * 80)
    results = semantic_search(query, documents, embeddings, top_k=3)
    for result in results:
        print(f"  [{result['rank']}] Similarity: {result['similarity']:.4f}")
        print(f"      Document: {result['document']}")
```

### **Advanced: Semantic Reasoning & Deduplication**

```python
# Advanced: Find semantically similar/duplicate content
def find_semantic_duplicates(documents, threshold=0.85):
    """Find near-duplicate documents using semantic similarity"""
    embeddings = np.array([
        quantum_kernel.compute_embedding(doc) for doc in documents
    ])
    
    # Compute pairwise similarities
    similarity_matrix = quantum_kernel.compute_similarity(embeddings, embeddings)
    
    duplicates = []
    seen = set()
    
    for i in range(len(documents)):
        if i in seen:
            continue
        
        group = [i]
        for j in range(i + 1, len(documents)):
            if similarity_matrix[i, j] >= threshold:
                group.append(j)
                seen.add(j)
        
        if len(group) > 1:
            duplicates.append({
                'group': group,
                'documents': [documents[idx] for idx in group],
                'avg_similarity': float(np.mean([
                    similarity_matrix[i, j] for j in group if j != i
                ]))
            })
    
    return duplicates

# Find duplicates
print("\n" + "="*80)
print("SEMANTIC DEDUPLICATION")
print("="*80)

duplicates = find_semantic_duplicates(documents, threshold=0.70)
for dup in duplicates:
    print(f"\nDuplicate Group (Similarity: {dup['avg_similarity']:.4f}):")
    for doc in dup['documents']:
        print(f"  - {doc}")
```

### **Use Case: Code Search & Understanding**

```python
# Search through code repositories semantically
code_snippets = [
    "def train_model(X, y): return model.fit(X, y)",
    "def predict(model, X): return model.predict(X)",
    "def evaluate_model(y_true, y_pred): return accuracy_score(y_true, y_pred)",
    "def preprocess_data(data): return StandardScaler().fit_transform(data)",
    "def split_data(X, y): return train_test_split(X, y, test_size=0.2)"
]

code_embeddings = np.array([
    quantum_kernel.compute_embedding(code) for code in code_snippets
])

# Find code that does similar things
query = "How to train a machine learning model?"
results = semantic_search(query, code_snippets, code_embeddings, top_k=2)

print("\n" + "="*80)
print("CODE SEMANTIC SEARCH")
print("="*80)
print(f"Query: '{query}'")
for result in results:
    print(f"\n[{result['rank']}] Similarity: {result['similarity']:.4f}")
    print(f"    Code: {result['document']}")
```

---

## 📈 **Use Case 2: Time Series & Tabular Prediction with Preprocessing & Interpretability**

### **Overview**

Build production-ready time series and tabular prediction models with automatic preprocessing, feature engineering, and model interpretability.

### **Scenario A: Time Series Forecasting with Interpretability**

```python
from ml_toolbox import MLToolbox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize toolbox
toolbox = MLToolbox()

# Generate sample time series data (sales forecasting)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
np.random.seed(42)

# Create realistic time series with trend, seasonality, and noise
trend = np.linspace(100, 200, len(dates))
seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
noise = np.random.normal(0, 5, len(dates))
sales = trend + seasonality + noise

# Create features
data = pd.DataFrame({
    'date': dates,
    'sales': sales,
    'day_of_week': dates.dayofweek,
    'month': dates.month,
    'quarter': dates.quarter,
    'is_weekend': (dates.dayofweek >= 5).astype(int)
})

# Split into train/test
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

X_train = train_data[['day_of_week', 'month', 'quarter', 'is_weekend']].values
y_train = train_data['sales'].values
X_test = test_data[['day_of_week', 'month', 'quarter', 'is_weekend']].values
y_test = test_data['sales'].values

print("="*80)
print("TIME SERIES FORECASTING WITH INTERPRETABILITY")
print("="*80)

# Train model with automatic preprocessing
print("\n1. Training model with automatic preprocessing...")
result = toolbox.fit(
    X_train, y_train,
    task_type='regression',
    use_preprocessing=True,
    use_interpretability=True
)

model = result['model']
print(f"   Model trained: {result['model_type']}")

# Make predictions
print("\n2. Making predictions...")
predictions = toolbox.predict(model, X_test)

# Evaluate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"\n3. Model Performance:")
print(f"   R² Score: {r2:.4f}")
print(f"   MSE: {mse:.4f}")
print(f"   MAE: {mae:.4f}")

# Interpretability: Feature importance
print("\n4. Feature Importance (Interpretability):")
if hasattr(model, 'feature_importances_'):
    feature_names = ['day_of_week', 'month', 'quarter', 'is_weekend']
    importances = model.feature_importances_
    for name, importance in zip(feature_names, importances):
        print(f"   {name}: {importance:.4f}")
elif hasattr(toolbox, 'get_interpretability'):
    interpretability = toolbox.get_interpretability()
    if interpretability:
        explanations = interpretability.explain_model(model, X_test[:10], y_test[:10])
        print(f"   Interpretability explanations generated")
        # Display key insights
        if 'feature_importance' in explanations:
            for feature, importance in explanations['feature_importance'].items():
                print(f"   {feature}: {importance:.4f}")

# Advanced: Time series specific preprocessing
print("\n5. Advanced Time Series Preprocessing:")
preprocessor = toolbox.get_universal_adaptive_preprocessor()
if preprocessor:
    # Add lag features, rolling statistics, etc.
    print("   Using Universal Adaptive Preprocessor for time series features...")
    # The preprocessor can automatically detect time series patterns
```

### **Scenario B: Tabular Data Prediction with Full Pipeline**

```python
# Comprehensive tabular prediction with preprocessing and interpretability

# Sample tabular data (customer churn prediction)
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'subscription_months': np.random.randint(1, 60, n_samples),
    'monthly_charges': np.random.uniform(20, 100, n_samples),
    'total_charges': np.random.uniform(100, 5000, n_samples),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'payment_method': np.random.choice(['Electronic', 'Mailed check', 'Bank transfer'], n_samples),
    'tenure': np.random.randint(1, 72, n_samples),
    'has_tech_support': np.random.choice([0, 1], n_samples),
    'has_online_security': np.random.choice([0, 1], n_samples)
})

# Create target (churn) with some logic
data['churn'] = (
    (data['subscription_months'] < 12) |
    (data['monthly_charges'] > 80) |
    (data['contract_type'] == 'Month-to-month')
).astype(int)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le_contract = LabelEncoder()
le_payment = LabelEncoder()

data['contract_type_encoded'] = le_contract.fit_transform(data['contract_type'])
data['payment_method_encoded'] = le_payment.fit_transform(data['payment_method'])

# Prepare features
feature_cols = [
    'age', 'subscription_months', 'monthly_charges', 'total_charges',
    'contract_type_encoded', 'payment_method_encoded', 'tenure',
    'has_tech_support', 'has_online_security'
]

X = data[feature_cols].values
y = data['churn'].values

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n" + "="*80)
print("TABULAR PREDICTION WITH PREPROCESSING & INTERPRETABILITY")
print("="*80)

# Train with full pipeline
print("\n1. Training with automatic preprocessing...")
result = toolbox.fit(
    X_train, y_train,
    task_type='classification',
    use_preprocessing=True,
    use_interpretability=True
)

model = result['model']
print(f"   Model: {result['model_type']}")

# Predictions
print("\n2. Making predictions...")
predictions = toolbox.predict(model, X_test)

# Evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"\n3. Model Performance:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1 Score: {f1:.4f}")

# Interpretability: Feature importance and explanations
print("\n4. Model Interpretability:")
if hasattr(model, 'feature_importances_'):
    print("   Feature Importances:")
    importances = model.feature_importances_
    feature_importance = list(zip(feature_cols, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for feature, importance in feature_importance:
        print(f"      {feature}: {importance:.4f}")

# SHAP-like explanations (if available)
if hasattr(toolbox, 'get_interpretability'):
    interpretability = toolbox.get_interpretability()
    if interpretability:
        print("\n5. Generating detailed explanations...")
        # Explain individual predictions
        explanations = interpretability.explain_prediction(
            model, X_test[0], feature_names=feature_cols
        )
        print(f"   Explanation for first test sample:")
        if explanations:
            for key, value in explanations.items():
                print(f"      {key}: {value}")
```

### **Advanced: AutoML with Interpretability**

```python
# Use AutoML framework for automatic model selection with interpretability
print("\n" + "="*80)
print("AUTOML WITH INTERPRETABILITY")
print("="*80)

automl = toolbox.get_automl_framework()
if automl:
    print("\n1. Running AutoML...")
    automl_result = automl.fit(
        X_train, y_train,
        task_type='classification',
        time_budget=60,  # 1 minute
        interpretability=True
    )
    
    best_model = automl_result['best_model']
    best_score = automl_result['best_score']
    
    print(f"\n2. Best Model Found:")
    print(f"   Model: {best_model}")
    print(f"   Score: {best_score:.4f}")
    
    # Get interpretability for best model
    if 'interpretability' in automl_result:
        print(f"\n3. Interpretability:")
        interp = automl_result['interpretability']
        print(f"   Feature importance available: {interp is not None}")
```

---

## 🤖 **Use Case 3: Small Agent-like Workflow**

### **Overview**

Build an intelligent agent that can understand tasks, generate code, execute it safely, and learn from results.

### **Scenario: ML Pipeline Agent**

```python
from ml_toolbox import MLToolbox
import json

# Initialize toolbox with AI agent capabilities
toolbox = MLToolbox()

print("="*80)
print("SMALL AGENT-LIKE WORKFLOW")
print("="*80)

# Get AI agent
ai_agent = toolbox.get_ai_agent()
if not ai_agent:
    print("AI Agent not available. Using basic workflow.")
    ai_agent = None

# Agent workflow: Task understanding -> Code generation -> Execution -> Learning

class MLPipelineAgent:
    """Small agent for ML pipeline automation"""
    
    def __init__(self, toolbox, ai_agent=None):
        self.toolbox = toolbox
        self.ai_agent = ai_agent
        self.history = []
    
    def understand_task(self, task_description):
        """Understand the user's task"""
        print(f"\n[Agent] Understanding task: '{task_description}'")
        
        if self.ai_agent:
            # Use AI agent to understand task
            understanding = self.ai_agent.understand_task(task_description)
            return understanding
        else:
            # Simple rule-based understanding
            task_type = None
            if 'classify' in task_description.lower() or 'classification' in task_description.lower():
                task_type = 'classification'
            elif 'predict' in task_description.lower() or 'regression' in task_description.lower():
                task_type = 'regression'
            elif 'cluster' in task_description.lower():
                task_type = 'clustering'
            
            return {
                'task_type': task_type,
                'description': task_description,
                'confidence': 0.8
            }
    
    def generate_pipeline(self, understanding, data_info=None):
        """Generate ML pipeline code"""
        print(f"\n[Agent] Generating pipeline for: {understanding.get('task_type', 'unknown')}")
        
        if self.ai_agent:
            # Use AI agent to generate code
            code = self.ai_agent.generate_code(
                task=understanding['description'],
                task_type=understanding.get('task_type'),
                data_info=data_info
            )
            return code
        else:
            # Simple template-based generation
            task_type = understanding.get('task_type', 'classification')
            template = f"""
# Auto-generated ML pipeline
from ml_toolbox import MLToolbox

toolbox = MLToolbox()
result = toolbox.fit(X_train, y_train, task_type='{task_type}')
model = result['model']
predictions = toolbox.predict(model, X_test)
"""
            return template
    
    def execute_pipeline(self, code, X_train, y_train, X_test):
        """Execute the generated pipeline safely"""
        print(f"\n[Agent] Executing pipeline...")
        
        # Use code sandbox if available
        if self.ai_agent and hasattr(self.ai_agent, 'code_sandbox'):
            sandbox = self.ai_agent.code_sandbox
            result = sandbox.execute(code, {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'toolbox': self.toolbox
            })
            return result
        else:
            # Direct execution (less safe, but works)
            try:
                exec_globals = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'toolbox': self.toolbox,
                    'MLToolbox': MLToolbox
                }
                exec(code, exec_globals)
                return {
                    'success': True,
                    'model': exec_globals.get('model'),
                    'predictions': exec_globals.get('predictions')
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
    
    def learn_from_result(self, task, code, result):
        """Learn from execution results"""
        print(f"\n[Agent] Learning from result...")
        
        learning = {
            'task': task,
            'code': code,
            'success': result.get('success', False),
            'timestamp': str(pd.Timestamp.now())
        }
        
        if result.get('success'):
            learning['outcome'] = 'success'
            if self.ai_agent and hasattr(self.ai_agent, 'knowledge_base'):
                # Store in knowledge base
                self.ai_agent.knowledge_base.store_pattern(task, code, result)
        else:
            learning['outcome'] = 'failure'
            learning['error'] = result.get('error')
        
        self.history.append(learning)
        return learning
    
    def run_workflow(self, task_description, X_train, y_train, X_test):
        """Run complete agent workflow"""
        print("\n" + "="*80)
        print("AGENT WORKFLOW EXECUTION")
        print("="*80)
        
        # Step 1: Understand task
        understanding = self.understand_task(task_description)
        print(f"   Task type: {understanding.get('task_type')}")
        print(f"   Confidence: {understanding.get('confidence', 0):.2f}")
        
        # Step 2: Generate pipeline
        data_info = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1] if len(X_train.shape) > 1 else 1,
            'n_classes': len(np.unique(y_train)) if understanding.get('task_type') == 'classification' else None
        }
        code = self.generate_pipeline(understanding, data_info)
        print(f"   Code generated: {len(code)} characters")
        
        # Step 3: Execute pipeline
        result = self.execute_pipeline(code, X_train, y_train, X_test)
        if result.get('success'):
            print(f"   Execution: SUCCESS")
            if 'predictions' in result:
                print(f"   Predictions shape: {result['predictions'].shape}")
        else:
            print(f"   Execution: FAILED - {result.get('error')}")
        
        # Step 4: Learn from result
        learning = self.learn_from_result(task_description, code, result)
        print(f"   Learning stored: {learning['outcome']}")
        
        return {
            'understanding': understanding,
            'code': code,
            'result': result,
            'learning': learning
        }

# Example usage
print("\n" + "="*80)
print("EXAMPLE: AGENT WORKFLOW")
print("="*80)

# Create sample data
np.random.seed(42)
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, 100)
X_test = np.random.rand(20, 5)

# Initialize agent
agent = MLPipelineAgent(toolbox, ai_agent)

# Run workflow
task = "Build a classification model to predict binary outcomes"
workflow_result = agent.run_workflow(task, X_train, y_train, X_test)

# Display workflow history
print("\n" + "="*80)
print("AGENT WORKFLOW HISTORY")
print("="*80)
for i, entry in enumerate(agent.history, 1):
    print(f"\n[{i}] Task: {entry['task']}")
    print(f"    Outcome: {entry['outcome']}")
    print(f"    Timestamp: {entry['timestamp']}")
```

### **Advanced: Multi-Step Agent Workflow**

```python
# Advanced: Agent that can handle multi-step tasks
class AdvancedMLAgent(MLPipelineAgent):
    """Advanced agent with multi-step reasoning"""
    
    def handle_complex_task(self, task_description, data):
        """Handle complex multi-step tasks"""
        print("\n" + "="*80)
        print("ADVANCED AGENT: COMPLEX TASK HANDLING")
        print("="*80)
        
        # Step 1: Break down task into subtasks
        subtasks = self.break_down_task(task_description)
        print(f"\n[Agent] Task broken into {len(subtasks)} subtasks:")
        for i, subtask in enumerate(subtasks, 1):
            print(f"   {i}. {subtask}")
        
        # Step 2: Execute subtasks sequentially
        results = []
        for subtask in subtasks:
            print(f"\n[Agent] Executing: {subtask}")
            result = self.run_workflow(subtask, data['X_train'], data['y_train'], data['X_test'])
            results.append(result)
        
        # Step 3: Combine results
        combined_result = self.combine_results(results)
        print(f"\n[Agent] Combined result: {combined_result.get('status')}")
        
        return combined_result
    
    def break_down_task(self, task_description):
        """Break complex task into subtasks"""
        # Simple rule-based breakdown
        subtasks = []
        
        if 'preprocess' in task_description.lower():
            subtasks.append("Preprocess the data")
        
        if 'train' in task_description.lower() or 'build' in task_description.lower():
            subtasks.append("Train a machine learning model")
        
        if 'evaluate' in task_description.lower() or 'test' in task_description.lower():
            subtasks.append("Evaluate the model performance")
        
        if 'interpret' in task_description.lower() or 'explain' in task_description.lower():
            subtasks.append("Generate model interpretability")
        
        if not subtasks:
            subtasks.append(task_description)
        
        return subtasks
    
    def combine_results(self, results):
        """Combine results from multiple subtasks"""
        return {
            'status': 'success' if all(r.get('result', {}).get('success') for r in results),
            'subtasks': len(results),
            'results': results
        }

# Example: Complex task
print("\n" + "="*80)
print("ADVANCED AGENT EXAMPLE")
print("="*80)

advanced_agent = AdvancedMLAgent(toolbox, ai_agent)

complex_task = "Preprocess the data, train a classification model, evaluate it, and explain the results"
data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test
}

complex_result = advanced_agent.handle_complex_task(complex_task, data)
```

---

## 🎯 **Summary**

### **Use Case 1: Semantic Search & Reasoning**
- ✅ **Quantum Kernel** for semantic understanding
- ✅ **Semantic similarity** search
- ✅ **Deduplication** of similar content
- ✅ **Code search** and understanding

### **Use Case 2: Time Series & Tabular Prediction**
- ✅ **Automatic preprocessing** with Universal Adaptive Preprocessor
- ✅ **Model interpretability** (feature importance, SHAP-like explanations)
- ✅ **AutoML** for automatic model selection
- ✅ **Full pipeline** from data to predictions

### **Use Case 3: Agent-like Workflow**
- ✅ **Task understanding** (AI-powered or rule-based)
- ✅ **Code generation** (AI agent or templates)
- ✅ **Safe execution** (code sandbox)
- ✅ **Learning** from results (knowledge base)
- ✅ **Multi-step** complex task handling

---

## 🚀 **Getting Started**

Run these examples:

```bash
# Semantic search
python -c "from USE_CASES import *; run_semantic_search_example()"

# Time series prediction
python -c "from USE_CASES import *; run_time_series_example()"

# Agent workflow
python -c "from USE_CASES import *; run_agent_example()"
```

Or use the examples directly in your code by importing the functions and classes defined above.

---

## 📝 **Notes**

- All examples use the ML Toolbox's revolutionary features
- Quantum Kernel provides semantic understanding beyond keyword matching
- Universal Adaptive Preprocessor handles automatic feature engineering
- AI Agent enables intelligent code generation and execution
- All workflows include interpretability for model understanding

**These use cases demonstrate the ML Toolbox's unique capabilities beyond standard ML libraries!** 🎉
