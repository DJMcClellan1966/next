"""
Information Theory Integration Examples

Demonstrates:
1. MI-based Feature Selection in Feature Pipeline
2. Enhanced DecisionTree with Information Theory
3. Information-Theoretic Evaluation Metrics
4. Data Quality Assessment Tools
"""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_toolbox import MLToolbox
from ml_toolbox.pipelines import FeaturePipeline, FeatureStore
from ml_toolbox.core_models import (
    DecisionTree, cross_entropy_loss, kl_divergence_score, model_comparison_kl
)
from ml_toolbox.textbook_concepts import (
    DataQualityAssessor, feature_informativeness, feature_redundancy
)

print("=" * 80)
print("Information Theory Integration Examples")
print("=" * 80)

# Initialize toolbox
toolbox = MLToolbox()
print("\n[OK] ML Toolbox initialized")

# ============================================================================
# Example 1: MI-based Feature Selection in Feature Pipeline
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: MI-based Feature Selection in Feature Pipeline")
print("=" * 80)

# Generate synthetic data with informative and non-informative features
np.random.seed(42)
n_samples = 200
n_features = 10

# Create informative features (correlated with target)
X = np.random.randn(n_samples, n_features)
y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

# Add some noise features
X_noise = np.random.randn(n_samples, 5)
X = np.column_stack([X, X_noise])

print(f"\nOriginal data shape: {X.shape}")
print(f"Target distribution: {np.bincount(y)}")

# Create feature pipeline with MI-based selection
feature_store = FeatureStore()
feature_pipeline = FeaturePipeline(toolbox=toolbox, feature_store=feature_store)

# Run with variance-based selection (old method)
print("\n--- Variance-based Feature Selection ---")
result_variance = feature_pipeline.execute(
    X,
    feature_name='variance_features',
    max_features=5,
    selection_method='variance',
    target=y
)
print(f"Selected features (variance): {feature_pipeline.stages[3].metadata.get('selected_indices', [])}")

# Run with MI-based selection (new method)
print("\n--- Mutual Information-based Feature Selection ---")
result_mi = feature_pipeline.execute(
    X,
    feature_name='mi_features',
    max_features=5,
    selection_method='mutual_information',
    target=y
)
print(f"Selected features (MI): {feature_pipeline.stages[3].metadata.get('selected_indices', [])}")
print(f"MI scores: {feature_pipeline.stages[3].metadata.get('mi_scores', [])}")

# ============================================================================
# Example 2: Enhanced DecisionTree with Information Theory
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Enhanced DecisionTree with Information Theory")
print("=" * 80)

# Train DecisionTree with entropy criterion (uses information_theory module)
tree = DecisionTree(max_depth=5, criterion='entropy')
tree.fit(X, y)

# Make predictions
y_pred = tree.predict(X)
accuracy = np.mean(y_pred == y)
print(f"\nDecisionTree Accuracy: {accuracy:.4f}")
print("[OK] DecisionTree successfully uses information_theory module for entropy/information gain")

# ============================================================================
# Example 3: Information-Theoretic Evaluation Metrics
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Information-Theoretic Evaluation Metrics")
print("=" * 80)

# Generate probability predictions
np.random.seed(42)
n_samples = 100
n_classes = 3

# True probabilities (one-hot)
y_true_proba = np.zeros((n_samples, n_classes))
y_true = np.random.randint(0, n_classes, n_samples)
y_true_proba[np.arange(n_samples), y_true] = 1.0

# Predicted probabilities (softmax-like)
y_pred_proba = np.random.rand(n_samples, n_classes)
y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

# Calculate cross-entropy loss
ce_loss = cross_entropy_loss(y_true, y_pred_proba)
print(f"\nCross-Entropy Loss: {ce_loss:.4f}")

# Calculate KL divergence
kl_div = kl_divergence_score(y_true_proba, y_pred_proba)
print(f"KL Divergence: {kl_div:.4f}")

# Compare two models
y_pred_proba_model2 = np.random.rand(n_samples, n_classes)
y_pred_proba_model2 = y_pred_proba_model2 / y_pred_proba_model2.sum(axis=1, keepdims=True)

comparison = model_comparison_kl(y_pred_proba, y_pred_proba_model2, reference_proba=y_true_proba)
print(f"\nModel Comparison:")
print(f"  Model 1 to Reference KL: {comparison['model1_to_reference']:.4f}")
print(f"  Model 2 to Reference KL: {comparison['model2_to_reference']:.4f}")
print(f"  Better Model: {comparison['better_model']}")

# ============================================================================
# Example 4: Data Quality Assessment Tools
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Data Quality Assessment Tools")
print("=" * 80)

# Generate data with varying quality
np.random.seed(42)
n_samples = 150
n_features = 8

# Create informative features
X_quality = np.random.randn(n_samples, n_features)
# Make some features redundant (highly correlated)
X_quality[:, 3] = X_quality[:, 2] + np.random.randn(n_samples) * 0.1
# Make some features low-informativeness (constant-like)
X_quality[:, 5] = np.random.randn(n_samples) * 0.01 + 5.0  # Almost constant

y_quality = (X_quality[:, 0] + X_quality[:, 1] + np.random.randn(n_samples) * 0.2 > 0).astype(int)

# Feature informativeness
print("\n--- Feature Informativeness ---")
informativeness = feature_informativeness(X_quality)
print(f"Informativeness scores: {informativeness}")
print(f"Low informativeness features (< 25th percentile): {np.where(informativeness < np.percentile(informativeness, 25))[0]}")

# Feature redundancy
print("\n--- Feature Redundancy Detection ---")
redundancy_info = feature_redundancy(X_quality, threshold=0.6)
print(f"Redundant feature pairs (MI >= 0.6): {len(redundancy_info['redundant_pairs'])}")
if redundancy_info['redundant_pairs']:
    print(f"  Pairs: {redundancy_info['redundant_pairs'][:3]}...")  # Show first 3
print(f"Highly redundant features: {redundancy_info['highly_redundant_features']}")

# Comprehensive data quality assessment
print("\n--- Comprehensive Data Quality Assessment ---")
assessor = DataQualityAssessor(n_bins=10)
assessment = assessor.assess(X_quality, y=y_quality)

print(f"\nQuality Scores:")
for key, value in assessment['quality_scores'].items():
    print(f"  {key}: {value:.4f}")

print(f"\nRecommendations:")
for i, rec in enumerate(assessment['recommendations'], 1):
    print(f"  {i}. {rec}")

# Missing value impact (simulate missing values)
print("\n--- Missing Value Impact Assessment ---")
missing_mask = np.random.rand(n_samples, n_features) < 0.1  # 10% missing
X_with_missing = X_quality.copy()
X_with_missing[missing_mask] = np.nan

from ml_toolbox.textbook_concepts.data_quality import missing_value_impact
missing_impact = missing_value_impact(X_quality, missing_mask, y=y_quality)
print(f"Features with highest information loss (priority for imputation):")
priority = missing_impact['recommended_imputation_priority'][:5]
for i, feat_idx in enumerate(priority, 1):
    loss = missing_impact['information_loss_per_feature'][feat_idx]
    ratio = missing_impact['feature_missing_ratios'][feat_idx]
    print(f"  {i}. Feature {feat_idx}: loss={loss:.4f}, missing_ratio={ratio:.4f}")

print("\n" + "=" * 80)
print("[OK] All Information Theory Integration Examples Completed!")
print("=" * 80)
