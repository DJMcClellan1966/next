# Unified ML Pipelines Module

## Overview

This module implements **Option 3: New Pipeline Module** - a professional, industry-standard pipeline architecture for ML systems.

## Architecture

### Four Main Pipelines

1. **Data Collection Pipeline** (`DataCollectionPipeline`) - **NEW!**
   - Extract: From user inputs and NoSQL databases
   - Transform: Clean, validate, structure data
   - Load: Output to Feature Pipeline
   - Uses ETL (Extract, Transform, Load) pattern

2. **Feature Pipeline** (`FeaturePipeline`)
   - Data Ingestion
   - Preprocessing
   - Feature Engineering
   - Feature Selection
   - Feature Store

3. **Training Pipeline** (`TrainingPipeline`)
   - Model Training
   - Hyperparameter Tuning
   - Model Evaluation
   - Model Validation
   - Model Registry

4. **Inference Pipeline** (`InferencePipeline`)
   - Model Serving
   - Batch Inference
   - Real-time Inference
   - A/B Testing
   - Monitoring

### Unified Orchestrator

**UnifiedMLPipeline** orchestrates all pipelines:
- Data Collection Pipeline (ETL) → Feature Pipeline → Training Pipeline → Inference Pipeline
- State management across pipelines
- Feature reuse between training and inference
- Pipeline versioning

## Quick Start

### Basic Usage

```python
from ml_toolbox import MLToolbox, UnifiedMLPipeline

# Initialize
toolbox = MLToolbox()
pipeline = UnifiedMLPipeline(toolbox, enable_data_collection=True)

# Training with ETL (from user input or NoSQL)
result = pipeline.execute(
    X_train,  # Can be dict, list, or array
    y_train,
    mode='train',
    use_data_collection=True,  # Enable ETL
    source_type='user_input',  # or 'nosql'
    feature_name='my_features',
    model_name='my_model'
)

# Inference (reuses features)
predictions = pipeline.execute(X_test, mode='inference',
                               feature_name='my_features',
                               model_name='my_model',
                               reuse_features=True)
```

### Data Collection Pipeline (ETL)

```python
from ml_toolbox.pipelines import DataCollectionPipeline

toolbox = MLToolbox()
data_collection = DataCollectionPipeline(toolbox)

# Extract from user input
collected_data = data_collection.execute(
    user_input,  # dict, list, or array
    source_type='user_input',
    feature_name='collected_features'
)

# Extract from NoSQL
collected_data = data_collection.execute(
    None,
    source_type='nosql',
    nosql_client=mongodb_client,
    nosql_collection='sensors',
    nosql_query={'status': 'active'},
    feature_name='nosql_features'
)
```

### Individual Pipelines

```python
from ml_toolbox.pipelines import FeaturePipeline, TrainingPipeline, InferencePipeline

toolbox = MLToolbox()

# Feature Pipeline
feature_pipeline = FeaturePipeline(toolbox)
X_features = feature_pipeline.execute(X, feature_name='features')

# Training Pipeline
training_pipeline = TrainingPipeline(toolbox)
train_result = training_pipeline.execute(X_features, y, model_name='model')

# Inference Pipeline
inference_pipeline = InferencePipeline(toolbox)
predictions = inference_pipeline.execute(X_test, train_result['model'])
```

## Key Features

### 1. Explicit Pipeline Stages
- Clear data flow through stages
- Easy to debug (know which stage failed)
- Easy to extend (add new stages)

### 2. Feature Store
- Store features for reuse
- Feature versioning
- Automatic feature retrieval

### 3. Pipeline State Management
- Track state across stages
- Pipeline versioning
- Reproducibility

### 4. Integration with ML Toolbox
- Uses existing kernels
- Leverages toolbox features
- Backward compatible

## Components

### Base Classes

- `PipelineStage`: Base class for pipeline stages
- `BasePipeline`: Base class for pipelines
- `PipelineState`: State management
- `FeatureStore`: Feature storage and retrieval

### Pipeline Classes

- `FeaturePipeline`: Feature engineering pipeline
- `TrainingPipeline`: Model training pipeline
- `InferencePipeline`: Model inference pipeline
- `UnifiedMLPipeline`: Orchestrates all pipelines

## Examples

See `examples/unified_pipeline_example.py` for complete examples.

## Benefits

1. **Explicit Structure**: Clear pipeline stages
2. **Feature Reuse**: Store and reuse features
3. **State Management**: Track pipeline state
4. **Industry Standard**: Aligns with Kubeflow, MLflow, TFX
5. **Extensible**: Easy to add new stages
6. **Professional**: Clean, maintainable code

## Comparison

| Aspect | Current `fit()`/`predict()` | Unified Pipeline |
|--------|---------------------------|------------------|
| **Structure** | Implicit | Explicit stages |
| **Feature Reuse** | Manual | Automatic |
| **State Tracking** | None | Full tracking |
| **Versioning** | Model only | Full pipeline |
| **Debugging** | Hard | Easy (stage-level) |
| **Extensibility** | Limited | High |

## Next Steps

1. **Use UnifiedMLPipeline** for production ML systems
2. **Use individual pipelines** for custom workflows
3. **Extend with custom stages** as needed
4. **Integrate with MLOps** tools (Kubeflow, MLflow, etc.)

## Documentation

- See `UNIFIED_ML_ARCHITECTURE_COMPARISON.md` for architecture comparison
- See `UNIFIED_PIPELINE_IMPLEMENTATION_ANALYSIS.md` for implementation details
