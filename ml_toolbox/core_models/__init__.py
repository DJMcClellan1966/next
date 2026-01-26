"""
Core ML Models - Regression, Classification, Neural Networks, Modern Architectures, Evaluation

Implements:
- Linear Regression
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVMs)
- Neural Networks with SGD, Dropout, Batch Normalization
- Transformer, BERT, GPT architectures
- Evaluation Metrics (Precision, Recall, F1, ROC)
"""
try:
    from .regression_classification import (
        LinearRegression, LogisticRegression, DecisionTree, SVM
    )
    from .neural_networks import (
        NeuralNetwork, SGD, Dropout, BatchNormalization
    )
    from .modern_architectures import (
        Transformer, BERT, GPT, MultiHeadAttention, TransformerBlock
    )
    from .evaluation_metrics import (
        precision_score, recall_score, f1_score,
        roc_curve, roc_auc_score, auc_score, classification_report,
        cross_entropy_loss, kl_divergence_score, model_comparison_kl
    )
    __all__ = [
        'LinearRegression',
        'LogisticRegression',
        'DecisionTree',
        'SVM',
        'NeuralNetwork',
        'SGD',
        'Dropout',
        'BatchNormalization',
        'Transformer',
        'BERT',
        'GPT',
        'MultiHeadAttention',
        'TransformerBlock',
        'precision_score',
        'recall_score',
        'f1_score',
        'roc_curve',
        'roc_auc_score',
        'auc_score',
        'classification_report',
        'cross_entropy_loss',
        'kl_divergence_score',
        'model_comparison_kl'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Some core models not available: {e}")
    __all__ = []
