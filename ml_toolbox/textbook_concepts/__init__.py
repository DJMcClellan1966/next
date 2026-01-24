"""
Textbook Concepts - Best Practices from Leading AI/ML Textbooks

Implements concepts from:
- Artificial Intelligence: A Modern Approach (Russell & Norvig)
- Hands-On Machine Learning (Géron)
- Deep Learning (Goodfellow et al.)
- Mathematics for Machine Learning (Deisenroth et al.)
- The Hundred-Page Machine Learning Book (Burkov)
- Pattern Recognition and Machine Learning (Bishop)
"""
try:
    from .knowledge_representation import KnowledgeBase, RuleBasedSystem, ExpertSystem
    from .practical_ml import (
        FeatureEngineering, ModelSelection, HyperparameterTuning,
        EnsembleMethods, CrossValidation, ProductionML
    )
    from .advanced_dl import (
        RegularizationTechniques, AdvancedOptimization, GenerativeModels,
        AttentionMechanisms, TransferLearning
    )
    from .information_theory import (
        Entropy, MutualInformation, KLDivergence, InformationGain
    )
    from .probabilistic_ml import (
        EMAlgorithm, VariationalInference, BayesianLearning, GraphicalModels
    )
    from .dimensionality_reduction import (
        PCA, LDA, tSNE, UMAP, Autoencoder
    )
    __all__ = [
        'KnowledgeBase',
        'RuleBasedSystem',
        'ExpertSystem',
        'FeatureEngineering',
        'ModelSelection',
        'HyperparameterTuning',
        'EnsembleMethods',
        'CrossValidation',
        'ProductionML',
        'RegularizationTechniques',
        'AdvancedOptimization',
        'GenerativeModels',
        'AttentionMechanisms',
        'TransferLearning',
        'Entropy',
        'MutualInformation',
        'KLDivergence',
        'InformationGain',
        'EMAlgorithm',
        'VariationalInference',
        'BayesianLearning',
        'GraphicalModels',
        'PCA',
        'LDA',
        'tSNE',
        'UMAP',
        'Autoencoder'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Some textbook concepts not available: {e}")
    __all__ = []
