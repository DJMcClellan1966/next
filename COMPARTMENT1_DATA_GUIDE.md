# Compartment 1: Data - How It Works

## 🎯 **Overview**

**Compartment 1: Data** is the first compartment of the ML Toolbox, responsible for all data preprocessing, validation, transformation, and data management tasks. It's designed to prepare your data for machine learning algorithms.

---

## 🏗️ **Architecture**

```
Compartment 1: Data
├── Core Preprocessors
│   ├── AdvancedDataPreprocessor (Quantum + PocketFence)
│   └── ConventionalPreprocessor (Basic)
│
├── Kuhn/Johnson Methods
│   ├── ModelSpecificPreprocessor
│   ├── MissingDataHandler
│   ├── ClassImbalanceHandler
│   ├── HighCardinalityHandler
│   └── VarianceCorrelationFilter
│
├── Advanced Features
│   ├── TimeSeriesFeatureEngineer
│   └── KnuthDataSampling/Preprocessing
│
└── Optimizations
    ├── LRU Caching
    ├── Performance Monitoring
    └── Big O Optimizations
```

---

## 🔧 **How It Works**

### **1. Initialization**

When you create a `DataCompartment`, it:

1. **Loads all available components** (lazy loading)
2. **Sets up caching** for performance
3. **Initializes monitoring** (if available)
4. **Registers component descriptions**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()
data_compartment = toolbox.data  # Compartment 1

# Or directly
from ml_toolbox.compartment1_data import DataCompartment
data = DataCompartment()
```

### **2. Component System**

The compartment uses a **component registry** pattern:

- **Components are loaded on-demand** (lazy loading)
- **Components are cached** for performance
- **Components can be accessed by name**
- **Components have descriptions** for documentation

```python
# List all components
data.list_components()

# Get component info
info = data.get_info()
print(info['components'])  # List of all component names
```

---

## 📦 **Available Components**

### **Core Preprocessors**

#### **1. AdvancedDataPreprocessor**
- **Purpose:** Advanced preprocessing with Quantum Kernel + PocketFence Kernel
- **Features:**
  - Safety filtering (PocketFence)
  - Semantic deduplication (Quantum)
  - Intelligent categorization (Quantum)
  - Quality scoring (Quantum)
  - Dimensionality reduction (PCA/SVD)
  - Automatic feature creation

**Usage:**
```python
# Get preprocessor
preprocessor = data.get_preprocessor(advanced=True)

# Or directly
from data_preprocessor import AdvancedDataPreprocessor
preprocessor = AdvancedDataPreprocessor()

# Preprocess text data
texts = ["text1", "text2", "text3"]
results = preprocessor.preprocess(texts, verbose=True)
```

#### **2. ConventionalPreprocessor**
- **Purpose:** Basic preprocessing with exact matching
- **Features:**
  - Basic safety filtering
  - Exact duplicate removal
  - Keyword-based categorization
  - Simple quality scoring

**Usage:**
```python
preprocessor = data.get_preprocessor(advanced=False)
results = preprocessor.preprocess(texts)
```

---

### **Kuhn/Johnson Methods**

#### **3. ModelSpecificPreprocessor**
- **Purpose:** Model-specific preprocessing (different preprocessing per model type)
- **Features:**
  - Tree models: Skip scaling (not needed)
  - Linear models: Centering/scaling
  - Distance-based: Spatial sign transformation
  - Box-Cox/Yeo-Johnson transformations

**Usage:**
```python
# Get model-specific preprocessor
preprocessor = data.get_model_specific_preprocessor(model_type='random_forest')

# For tree-based models (no scaling needed)
X_processed = preprocessor.fit_transform(X)

# For linear models (scaling required)
preprocessor = data.get_model_specific_preprocessor(model_type='linear')
X_processed = preprocessor.fit_transform(X)
```

**Model Types:**
- `'tree'` - Random Forest, XGBoost, etc. (no scaling)
- `'linear'` - Logistic Regression, Linear Regression (scaling)
- `'knn'` - k-Nearest Neighbors (spatial sign)
- `'svm'` - Support Vector Machines (robust scaling)
- `'neural_net'` - Neural Networks (min-max scaling)
- `'auto'` - Auto-detect from model object

#### **4. MissingDataHandler**
- **Purpose:** Systematic missing data handling
- **Features:**
  - Multiple imputation strategies (mean, median, KNN, iterative)
  - Missing indicator variables
  - Pattern detection (MCAR, MAR, MNAR)
  - CV-aware imputation

**Usage:**
```python
handler = data.get_missing_data_handler(strategy='knn', add_indicator=True)
X_imputed = handler.fit_transform(X, y)
```

#### **5. ClassImbalanceHandler**
- **Purpose:** Handle class imbalance
- **Features:**
  - SMOTE (Synthetic Minority Oversampling)
  - ADASYN, BorderlineSMOTE
  - Random undersampling
  - Cost-sensitive learning
  - Threshold tuning

**Usage:**
```python
handler = data.get_class_imbalance_handler(method='smote')
X_resampled, y_resampled = handler.fit_resample(X, y)
```

#### **6. HighCardinalityHandler**
- **Purpose:** Handle high-cardinality categorical variables
- **Features:**
  - Target encoding (mean encoding)
  - Feature hashing
  - Frequency encoding
  - Rare category grouping

**Usage:**
```python
handler = data.get_high_cardinality_handler(method='target_encoding')
X_encoded = handler.fit_transform(X, y)
```

#### **7. VarianceCorrelationFilter**
- **Purpose:** Filter uninformative and correlated features
- **Features:**
  - Near-zero variance detection
  - High correlation filtering
  - Percent unique values
  - Frequency ratio analysis

**Usage:**
```python
filter = data.get_variance_correlation_filter(
    remove_nzv=True,
    remove_high_correlation=True
)
X_filtered = filter.fit_transform(X)
```

---

## 💻 **Usage Examples**

### **Example 1: Basic Preprocessing**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()
data = toolbox.data

# Preprocess text data
texts = [
    "Python is a programming language",
    "Machine learning is a subset of AI",
    "Data science involves statistics"
]

# Use advanced preprocessor
results = data.preprocess(
    texts,
    advanced=True,
    dedup_threshold=0.85,
    verbose=True
)

# Get preprocessed data
processed_texts = results['processed_data']
embeddings = results.get('embeddings', [])
```

### **Example 2: Model-Specific Preprocessing**

```python
from ml_toolbox import MLToolbox
import numpy as np
import pandas as pd

toolbox = MLToolbox()
data = toolbox.data

# Create sample data
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

# For Random Forest (tree-based - no scaling needed)
preprocessor = data.get_model_specific_preprocessor(model_type='tree')
X_processed = preprocessor.fit_transform(X)

# For Logistic Regression (linear - scaling required)
preprocessor = data.get_model_specific_preprocessor(model_type='linear')
X_processed = preprocessor.fit_transform(X)
```

### **Example 3: Handling Missing Data**

```python
import pandas as pd
import numpy as np

# Create data with missing values
df = pd.DataFrame({
    'feature1': [1, 2, np.nan, 4, 5],
    'feature2': [10, np.nan, 30, 40, 50],
    'target': [0, 1, 0, 1, 0]
})

X = df.drop(columns=['target'])
y = df['target']

# Handle missing data
handler = data.get_missing_data_handler(strategy='knn', add_indicator=True)
X_imputed = handler.fit_transform(X, y)
```

### **Example 4: Complete Preprocessing Pipeline**

```python
from ml_toolbox import MLToolbox
import numpy as np

toolbox = MLToolbox()
data = toolbox.data

# Step 1: Handle missing data
handler = data.get_missing_data_handler(strategy='knn')
X = handler.fit_transform(X, y)

# Step 2: Filter uninformative features
filter = data.get_variance_correlation_filter()
X = filter.fit_transform(X)

# Step 3: Handle class imbalance
imbalance_handler = data.get_class_imbalance_handler(method='smote')
X, y = imbalance_handler.fit_resample(X, y)

# Step 4: Model-specific preprocessing
preprocessor = data.get_model_specific_preprocessor(model_type='random_forest')
X = preprocessor.fit_transform(X)

# Now X is ready for training!
```

---

## ⚡ **Performance Optimizations**

### **1. LRU Caching**

Preprocessor instances are cached using LRU (Least Recently Used) cache:

```python
# First call - creates instance
preprocessor1 = data.get_preprocessor(advanced=True)

# Second call - returns cached instance (fast!)
preprocessor2 = data.get_preprocessor(advanced=True)
```

**Big O Complexity:**
- Cache hit: O(1)
- Cache miss: O(1) - instantiation is fast

### **2. Lazy Loading**

Components are loaded only when needed:

```python
# Component not loaded yet
data = DataCompartment()

# Component loaded on first use
preprocessor = data.get_preprocessor()  # Now loaded
```

### **3. Performance Monitoring**

If optimizations are available, performance is monitored:

```python
# Monitoring happens automatically
# Check performance metrics if available
if data._monitor:
    metrics = data._monitor.get_metrics()
```

---

## 🔍 **How Components Are Loaded**

### **Component Initialization Flow**

```
1. DataCompartment.__init__()
   └── _initialize_components()
       ├── Try to import AdvancedDataPreprocessor
       ├── Try to import ModelSpecificPreprocessor
       ├── Try to import MissingDataHandler
       ├── Try to import ClassImbalanceHandler
       ├── Try to import HighCardinalityHandler
       ├── Try to import VarianceCorrelationFilter
       └── Store in self.components dictionary
```

### **Component Access**

```python
# Direct access to component class
preprocessor_class = data.components['AdvancedDataPreprocessor']

# Create instance
preprocessor = preprocessor_class()

# Or use helper method
preprocessor = data.get_preprocessor(advanced=True)
```

---

## 📊 **Component Descriptions**

Each component has metadata:

```python
# Get component description
desc = data.component_descriptions['AdvancedDataPreprocessor']

print(desc['description'])
print(desc['features'])
print(desc['location'])
print(desc['category'])
```

**Available Information:**
- `description`: What the component does
- `features`: List of features
- `location`: Source file
- `category`: Component category

---

## 🎯 **Key Features**

### **1. Unified Interface**

All preprocessing goes through the compartment:

```python
# Simple interface
results = data.preprocess(texts, advanced=True)
```

### **2. Model-Specific Processing**

Different preprocessing for different models:

```python
# Tree-based: No scaling
preprocessor = data.get_model_specific_preprocessor('tree')

# Linear: Scaling required
preprocessor = data.get_model_specific_preprocessor('linear')
```

### **3. Comprehensive Data Handling**

Handles all common data issues:

- Missing data
- Class imbalance
- High-cardinality categoricals
- Uninformative features
- Correlated features

### **4. Performance Optimized**

- LRU caching
- Lazy loading
- Performance monitoring
- Big O optimizations

---

## 🔗 **Integration with Other Compartments**

### **With Compartment 2 (Infrastructure)**

```python
toolbox = MLToolbox()

# Use Quantum Kernel from Infrastructure
kernel = toolbox.infrastructure.get_kernel()

# AdvancedDataPreprocessor uses Quantum Kernel internally
preprocessor = toolbox.data.get_preprocessor(advanced=True)
```

### **With Compartment 3 (Algorithms)**

```python
# Preprocess data
X = data.preprocess(texts)

# Train model
model = algorithms.get_simple_ml_tasks().train_classifier(X, y)
```

---

## 📝 **Best Practices**

### **1. Use Model-Specific Preprocessing**

```python
# Know your model type
preprocessor = data.get_model_specific_preprocessor(model_type='random_forest')
X = preprocessor.fit_transform(X)
```

### **2. Handle Data Issues Early**

```python
# Handle missing data first
X = data.get_missing_data_handler().fit_transform(X, y)

# Then filter features
X = data.get_variance_correlation_filter().fit_transform(X)

# Then handle imbalance
X, y = data.get_class_imbalance_handler().fit_resample(X, y)
```

### **3. Use Caching**

```python
# Reuse preprocessor instances (cached)
preprocessor = data.get_preprocessor(advanced=True)
# Use multiple times - cached!
```

### **4. Check Component Availability**

```python
# Check if component is available
if 'ModelSpecificPreprocessor' in data.components:
    preprocessor = data.get_model_specific_preprocessor()
else:
    print("Component not available")
```

---

## 🐛 **Troubleshooting**

### **"Component not available"**

Components are loaded lazily. If a component isn't available:

1. Check if the module exists
2. Check if dependencies are installed
3. Check import errors in console

### **"Preprocessor not working"**

1. Check if data format is correct
2. Check if dependencies are installed (sklearn, etc.)
3. Check component initialization

### **"Performance issues"**

1. Use caching (automatic)
2. Use model-specific preprocessing (faster)
3. Check monitoring metrics

---

## 📚 **Summary**

**Compartment 1: Data** provides:

✅ **Unified interface** for all data preprocessing  
✅ **Model-specific preprocessing** (Kuhn/Johnson methods)  
✅ **Comprehensive data handling** (missing, imbalance, etc.)  
✅ **Performance optimizations** (caching, lazy loading)  
✅ **Easy integration** with other compartments  

**Key Principle:** Prepare your data efficiently and appropriately for your specific model type.

---

**For more details, see:**
- `ml_toolbox/compartment1_data.py` - Source code
- `KUHN_JOHNSON_BOTTLENECK_SUMMARY.md` - Preprocessing optimization guide
- `DATA_SCRUBBING_GUIDE.md` - Data cleaning guide
