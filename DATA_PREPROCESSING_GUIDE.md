# Data Preprocessing Guide 📊

## Overview

The ML Toolbox provides multiple data preprocessing approaches, from simple conventional preprocessing to advanced AI-powered preprocessing. This guide explains how data preprocessing works.

---

## 🏗️ **Preprocessing Architecture**

### **Compartment 1: Data Preprocessing**

The ML Toolbox organizes preprocessing into **Compartment 1 (Data)**, which includes:

1. **Universal Adaptive Preprocessor** - AI-powered, automatic preprocessing
2. **Advanced Data Preprocessor** - Quantum kernel + semantic deduplication
3. **Conventional Preprocessor** - Standard scaling, encoding, normalization
4. **Model-Specific Preprocessor** - Kuhn/Johnson methods for specific models
5. **Data Cleaning Utilities** - Missing values, outliers, standardization

---

## 🔄 **Preprocessing Workflow**

### **Standard Workflow:**

```
Raw Data
    ↓
[1. Data Validation]
    ↓
[2. Missing Value Handling]
    ↓
[3. Outlier Detection & Treatment]
    ↓
[4. Feature Engineering]
    ↓
[5. Encoding (Categorical → Numerical)]
    ↓
[6. Scaling/Normalization]
    ↓
[7. Dimensionality Reduction (Optional)]
    ↓
[8. Quality Assessment]
    ↓
Preprocessed Data
```

---

## 🎯 **Preprocessing Methods**

### **1. Universal Adaptive Preprocessor (Recommended)**

**AI-powered preprocessing that automatically adapts to your data.**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Access Universal Adaptive Preprocessor
preprocessor = toolbox.universal_preprocessor

# Automatic preprocessing
X_processed = preprocessor.fit_transform(X, y)

# What it does automatically:
# - Detects data types (numerical, categorical, text, time series)
# - Chooses appropriate preprocessing for each type
# - Handles missing values intelligently
# - Detects and handles outliers
# - Encodes categorical variables
# - Scales numerical features
# - Reduces dimensionality if needed
# - Scores data quality
```

**Features:**
- ✅ **Automatic detection** - Detects data types automatically
- ✅ **Adaptive** - Adapts preprocessing to data characteristics
- ✅ **AI-powered** - Uses ML to optimize preprocessing
- ✅ **Comprehensive** - Handles all preprocessing steps
- ✅ **Quality scoring** - Assesses data quality automatically

**When to Use:**
- Quick prototyping
- Unknown data characteristics
- Want automatic preprocessing
- Need best preprocessing without manual tuning

---

### **2. Advanced Data Preprocessor**

**Quantum kernel + semantic deduplication for advanced preprocessing.**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Access Advanced Data Preprocessor
advanced_preprocessor = toolbox.data.get_advanced_preprocessor()

# Advanced preprocessing with quantum kernel
result = advanced_preprocessor.preprocess(
    data=X,
    use_quantum_kernel=True,      # Semantic understanding
    dedup_threshold=0.85,         # Remove semantic duplicates
    enable_compression=True,       # Dimensionality reduction
    compression_ratio=0.5,         # Compress to 50% of original
    quality_scoring=True          # Assess quality
)

# Returns:
# {
#     'processed_data': preprocessed array,
#     'compressed_embeddings': compressed features,
#     'quality_score': 0.95,
#     'deduplication_info': {...},
#     'metadata': {...}
# }
```

**Features:**
- ✅ **Quantum Kernel** - Semantic understanding of data
- ✅ **Semantic Deduplication** - Finds and removes near-duplicates
- ✅ **PocketFence Kernel** - Content filtering and safety
- ✅ **Quality Scoring** - Automatic quality assessment
- ✅ **Intelligent Categorization** - Automatic categorization

**When to Use:**
- Text data with semantic meaning
- Need to find duplicate/similar records
- Want semantic understanding
- Need content filtering

**Key Steps:**
1. **Semantic Embedding** - Convert data to semantic embeddings
2. **Deduplication** - Find and remove semantic duplicates
3. **Quality Filtering** - Filter low-quality data
4. **Compression** - Reduce dimensionality while preserving meaning
5. **Quality Scoring** - Assess final data quality

---

### **3. Conventional Preprocessor**

**Standard preprocessing: scaling, encoding, normalization.**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Access Conventional Preprocessor
conventional = toolbox.data.get_conventional_preprocessor()

# Standard preprocessing
X_processed = conventional.fit_transform(X, y)

# What it does:
# - Standard scaling (mean=0, std=1)
# - Min-max normalization (0-1 range)
# - One-hot encoding (categorical)
# - Label encoding (ordinal)
# - Missing value imputation (mean/median/mode)
```

**Features:**
- ✅ **Standard methods** - Industry-standard preprocessing
- ✅ **Fast** - Optimized implementations
- ✅ **Reliable** - Battle-tested methods
- ✅ **Simple** - Easy to understand and use

**When to Use:**
- Standard ML tasks
- Known data characteristics
- Need fast preprocessing
- Want simple, reliable preprocessing

**Key Steps:**
1. **Missing Value Imputation** - Fill missing values
2. **Categorical Encoding** - Convert categories to numbers
3. **Scaling** - Standardize numerical features
4. **Normalization** - Normalize to 0-1 range

---

### **4. Model-Specific Preprocessor (Kuhn/Johnson)**

**Preprocessing optimized for specific ML models.**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Access Model-Specific Preprocessor
model_preprocessor = toolbox.data.get_model_specific_preprocessor()

# Preprocess for specific model
X_processed = model_preprocessor.fit_transform(
    X, y,
    model_type='random_forest'  # or 'svm', 'neural_network', etc.
)

# What it does:
# - Chooses preprocessing based on model requirements
# - Optimizes features for model type
# - Applies model-specific transformations
```

**Features:**
- ✅ **Model-optimized** - Preprocessing tailored to model
- ✅ **Kuhn/Johnson methods** - Based on Applied Predictive Modeling
- ✅ **Feature engineering** - Model-specific features
- ✅ **Optimized** - Best preprocessing for each model

**When to Use:**
- Know which model you'll use
- Want best preprocessing for specific model
- Need model-specific feature engineering

**Supported Models:**
- Random Forest
- SVM
- Neural Networks
- Linear Models
- Tree-based models

---

## 📊 **Preprocessing Steps in Detail**

### **Step 1: Data Validation**

```python
# Automatic data validation
validation_result = toolbox.data.validate_data(X)

# Checks:
# - Data types
# - Missing values
# - Outliers
# - Data quality
# - Feature distributions
```

**What it checks:**
- ✅ Data types (numerical, categorical, text)
- ✅ Missing value patterns
- ✅ Outlier detection
- ✅ Data quality metrics
- ✅ Feature distributions

---

### **Step 2: Missing Value Handling**

```python
# Automatic missing value handling
from ml_toolbox.compartment1_data.data_cleaning_utilities import MissingValueHandler

handler = MissingValueHandler()
X_imputed = handler.handle_missing_values(X, strategy='auto')

# Strategies:
# - 'mean' - Mean imputation (numerical)
# - 'median' - Median imputation (numerical)
# - 'mode' - Mode imputation (categorical)
# - 'knn' - K-Nearest Neighbors imputation
# - 'auto' - Automatically choose best strategy
```

**Strategies:**
- **Mean/Median** - For numerical features
- **Mode** - For categorical features
- **KNN Imputation** - More sophisticated
- **Forward/Backward Fill** - For time series
- **Auto** - Automatically choose best

---

### **Step 3: Outlier Detection & Treatment**

```python
# Automatic outlier detection and treatment
from ml_toolbox.compartment1_data.data_cleaning_utilities import OutlierHandler

handler = OutlierHandler()
X_cleaned = handler.handle_outliers(X, method='auto')

# Methods:
# - 'iqr' - Interquartile Range
# - 'zscore' - Z-score method
# - 'isolation_forest' - Isolation Forest
# - 'auto' - Automatically choose best
```

**Methods:**
- **IQR** - Interquartile Range (robust)
- **Z-Score** - Statistical method
- **Isolation Forest** - ML-based detection
- **Auto** - Automatically choose best

---

### **Step 4: Feature Engineering**

```python
# Automatic feature engineering
from ml_toolbox.compartment1_data import FeatureEngineer

engineer = FeatureEngineer()
X_engineered = engineer.create_features(X)

# Creates:
# - Polynomial features
# - Interaction features
# - Time-based features (if time series)
# - Statistical features
# - Domain-specific features
```

**Feature Types:**
- **Polynomial** - x², x³, etc.
- **Interaction** - x₁ × x₂
- **Time-based** - Lag, rolling stats
- **Statistical** - Mean, std, min, max
- **Domain-specific** - Custom features

---

### **Step 5: Categorical Encoding**

```python
# Automatic categorical encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label encoding (ordinal)
label_encoder = LabelEncoder()
X_encoded = label_encoder.fit_transform(X_categorical)

# One-hot encoding (nominal)
onehot_encoder = OneHotEncoder()
X_onehot = onehot_encoder.fit_transform(X_categorical)
```

**Encoding Methods:**
- **Label Encoding** - For ordinal categories
- **One-Hot Encoding** - For nominal categories
- **Target Encoding** - Mean target per category
- **Frequency Encoding** - Category frequency
- **Auto** - Automatically choose best

---

### **Step 6: Scaling/Normalization**

```python
# Automatic scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standard scaling (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-max normalization (0-1 range)
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X)
```

**Scaling Methods:**
- **Standard Scaling** - Mean=0, Std=1
- **Min-Max Scaling** - 0-1 range
- **Robust Scaling** - Median and IQR (robust to outliers)
- **Auto** - Automatically choose best

---

### **Step 7: Dimensionality Reduction (Optional)**

```python
# Automatic dimensionality reduction
from sklearn.decomposition import PCA

# PCA reduction
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X)

# Or use quantum kernel compression
result = advanced_preprocessor.preprocess(
    X,
    enable_compression=True,
    compression_ratio=0.5  # Compress to 50%
)
```

**Reduction Methods:**
- **PCA** - Principal Component Analysis
- **t-SNE** - t-Distributed Stochastic Neighbor Embedding
- **UMAP** - Uniform Manifold Approximation
- **Quantum Compression** - Semantic compression
- **Auto** - Automatically choose best

---

### **Step 8: Quality Assessment**

```python
# Automatic quality assessment
quality_score = toolbox.data.assess_quality(X_processed)

# Assesses:
# - Data completeness
# - Data consistency
# - Feature quality
# - Overall data quality
```

**Quality Metrics:**
- **Completeness** - Percentage of non-missing values
- **Consistency** - Data consistency checks
- **Feature Quality** - Individual feature quality
- **Overall Score** - Composite quality score

---

## 🚀 **Using Preprocessing in Practice**

### **Method 1: Using Universal Adaptive Preprocessor (Easiest)**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Automatic preprocessing
preprocessor = toolbox.universal_preprocessor
X_processed = preprocessor.fit_transform(X, y)

# That's it! Everything is automatic.
```

---

### **Method 2: Using Data Kernel (Unified Interface)**

```python
from ml_toolbox.compartment_kernels import DataKernel

# Create data kernel
data_kernel = DataKernel(toolbox.data)

# Process data
result = data_kernel.fit(X_train).transform(X_test)

# Or use process() for complete pipeline
result = data_kernel.process(X_train)
# Returns: {
#     'processed_data': preprocessed array,
#     'quality_score': 0.95,
#     'metadata': {...}
# }
```

---

### **Method 3: Using Compartment Directly**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Access data compartment
data_compartment = toolbox.data

# Use specific preprocessor
preprocessor = data_compartment.get_advanced_preprocessor()
X_processed = preprocessor.fit_transform(X, y)
```

---

### **Method 4: Custom Preprocessing Pipeline**

```python
from ml_toolbox import MLToolbox
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

toolbox = MLToolbox()

# Build custom pipeline
pipeline = Pipeline([
    ('imputation', toolbox.data.get_missing_value_handler()),
    ('encoding', OneHotEncoder()),
    ('scaling', StandardScaler()),
    ('quality_check', toolbox.data.get_quality_scorer())
])

# Use pipeline
X_processed = pipeline.fit_transform(X, y)
```

---

## 🎯 **Preprocessing Examples**

### **Example 1: Simple Preprocessing**

```python
from ml_toolbox import MLToolbox
import numpy as np

toolbox = MLToolbox()

# Sample data with missing values
X = np.array([
    [1, 2, np.nan],
    [4, 5, 6],
    [7, np.nan, 9]
])

# Automatic preprocessing
preprocessor = toolbox.universal_preprocessor
X_processed = preprocessor.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Processed shape: {X_processed.shape}")
print(f"Missing values: {np.isnan(X_processed).sum()}")
```

---

### **Example 2: Text Data Preprocessing**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Text data
texts = [
    "Machine learning is amazing",
    "Deep learning uses neural networks",
    "ML and AI are transforming industries"
]

# Advanced preprocessing with semantic understanding
advanced = toolbox.data.get_advanced_preprocessor()
result = advanced.preprocess(
    texts,
    use_quantum_kernel=True,
    dedup_threshold=0.85,
    enable_compression=True
)

print(f"Processed embeddings shape: {result['compressed_embeddings'].shape}")
print(f"Quality score: {result['quality_score']}")
print(f"Deduplication: {result['deduplication_info']}")
```

---

### **Example 3: Complete ML Pipeline with Preprocessing**

```python
from ml_toolbox import MLToolbox
from sklearn.model_selection import train_test_split
import numpy as np

toolbox = MLToolbox()

# Generate sample data
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess using data kernel
from ml_toolbox.compartment_kernels import DataKernel

data_kernel = DataKernel(toolbox.data)
data_result = data_kernel.fit(X_train).transform(X_test)

# Get preprocessed data
X_train_processed = data_kernel.transform(X_train)
X_test_processed = data_result['processed_data']

# Train model on preprocessed data
result = toolbox.fit(X_train_processed, y_train, task_type='classification')
predictions = toolbox.predict(result['model'], X_test_processed)

print(f"Quality score: {data_result['quality_score']:.2f}")
print(f"Accuracy: {toolbox.algorithms.evaluate_model(result['model'], X_test_processed, y_test)['accuracy']:.2f}")
```

---

## 🔍 **Preprocessing Details**

### **What Happens Internally:**

1. **Data Type Detection**
   ```python
   # Automatically detects:
   # - Numerical features (int, float)
   # - Categorical features (object, string)
   # - Text features (long strings)
   # - Time series features (datetime)
   ```

2. **Missing Value Analysis**
   ```python
   # Analyzes missing value patterns:
   # - Percentage missing per feature
   # - Missing value patterns (MCAR, MAR, MNAR)
   # - Best imputation strategy
   ```

3. **Outlier Detection**
   ```python
   # Detects outliers using:
   # - Statistical methods (IQR, Z-score)
   # - ML methods (Isolation Forest)
   # - Domain knowledge
   ```

4. **Feature Engineering**
   ```python
   # Creates features:
   # - Polynomial features
   # - Interaction features
   # - Time-based features
   # - Statistical features
   ```

5. **Encoding**
   ```python
   # Encodes categorical:
   # - Label encoding (ordinal)
   # - One-hot encoding (nominal)
   # - Target encoding (supervised)
   ```

6. **Scaling**
   ```python
   # Scales numerical:
   # - Standard scaling (mean=0, std=1)
   # - Min-max scaling (0-1)
   # - Robust scaling (median, IQR)
   ```

7. **Quality Assessment**
   ```python
   # Assesses quality:
   # - Completeness score
   # - Consistency score
   # - Feature quality scores
   # - Overall quality score
   ```

---

## 📊 **Preprocessing Comparison**

| Method | Speed | Quality | Automation | Best For |
|--------|-------|---------|------------|----------|
| **Universal Adaptive** | Fast | High | Full | Quick prototyping, unknown data |
| **Advanced** | Medium | Very High | High | Text data, semantic understanding |
| **Conventional** | Very Fast | Good | Medium | Standard ML, known data |
| **Model-Specific** | Fast | High | Medium | Specific models, optimization |

---

## 🎯 **Best Practices**

### **1. Use Universal Adaptive for Quick Start**
```python
# Easiest and most automatic
preprocessor = toolbox.universal_preprocessor
X_processed = preprocessor.fit_transform(X, y)
```

### **2. Use Advanced for Text/Semantic Data**
```python
# Best for text and semantic understanding
advanced = toolbox.data.get_advanced_preprocessor()
result = advanced.preprocess(texts, use_quantum_kernel=True)
```

### **3. Use Data Kernel for Unified Interface**
```python
# Consistent interface, easy to use
data_kernel = DataKernel(toolbox.data)
result = data_kernel.process(X)
```

### **4. Always Check Quality Score**
```python
# Assess data quality
result = data_kernel.process(X)
if result['quality_score'] < 0.7:
    print("Warning: Low data quality!")
```

---

## 📝 **Summary**

**Data preprocessing in ML Toolbox:**

1. **Multiple Methods** - Universal, Advanced, Conventional, Model-Specific
2. **Automatic** - Detects data types and chooses preprocessing automatically
3. **Comprehensive** - Handles all preprocessing steps
4. **Quality Assessment** - Scores data quality automatically
5. **Easy to Use** - Simple API, unified interface

**Recommended Approach:**
- **Quick start:** Use Universal Adaptive Preprocessor
- **Text data:** Use Advanced Data Preprocessor
- **Unified interface:** Use Data Kernel
- **Custom needs:** Build custom pipeline

**The ML Toolbox makes data preprocessing automatic, comprehensive, and easy to use!** 🚀
