"""
Statistical Learning Methods Demo
Demonstrates uncertainty quantification, statistical validation, Bayesian optimization, and feature selection
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_toolbox import MLToolbox
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print("="*80)
print("STATISTICAL LEARNING METHODS DEMO")
print("="*80)

# Initialize toolbox
toolbox = MLToolbox()

# Generate sample data
print("\n[1] Generating sample data...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Features: {X.shape[1]}")

# 1. Uncertainty Quantification
print("\n[2] Uncertainty Quantification with StatisticalEvaluator")
print("-" * 80)
try:
    stat_evaluator = toolbox.algorithms.get_statistical_evaluator(n_bootstrap=100)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions with confidence intervals
    results = stat_evaluator.predict_with_confidence(
        model=model,
        X=X_test[:10],  # First 10 samples for demo
        confidence_level=0.95,
        method='bootstrap'
    )
    
    if 'error' not in results:
        print(f"  Predictions: {results['predictions'][:5]}")
        print(f"  Confidence intervals (95%):")
        for i in range(5):
            ci = results['confidence_intervals'][i]
            print(f"    Sample {i}: [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  Uncertainty scores (mean): {np.mean(results['uncertainty_scores']):.4f}")
    else:
        print(f"  Error: {results['error']}")
except Exception as e:
    print(f"  Error: {e}")

# 2. Statistical Validation
print("\n[3] Statistical Validation with Permutation Test")
print("-" * 80)
try:
    validator = toolbox.algorithms.get_statistical_validator()
    
    # Compare two models
    model1 = RandomForestClassifier(n_estimators=50, random_state=42)
    model2 = GradientBoostingClassifier(n_estimators=50, random_state=42)
    
    comparison = validator.permutation_test(
        model1=model1,
        model2=model2,
        X=X_train,
        y=y_train,
        metric='accuracy',
        n_permutations=100,  # Reduced for demo
        task_type='classification'
    )
    
    if 'error' not in comparison:
        print(f"  Model 1 score: {comparison['model1_score']:.4f}")
        print(f"  Model 2 score: {comparison['model2_score']:.4f}")
        print(f"  Observed difference: {comparison['statistic']:.4f}")
        print(f"  P-value: {comparison['p_value']:.4f}")
        print(f"  Significant: {comparison['significant']}")
        print(f"  Effect size: {comparison['effect_size']:.4f}")
    else:
        print(f"  Error: {comparison['error']}")
except Exception as e:
    print(f"  Error: {e}")

# 3. Bootstrap Validation
print("\n[4] Bootstrap Validation")
print("-" * 80)
try:
    validator = toolbox.algorithms.get_statistical_validator()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    bootstrap_results = validator.bootstrap_validation(
        model=model,
        X=X_train,
        y=y_train,
        n_bootstrap=50,  # Reduced for demo
        metric='accuracy',
        task_type='classification'
    )
    
    if 'error' not in bootstrap_results:
        print(f"  Mean score: {bootstrap_results['mean_score']:.4f}")
        print(f"  Std score: {bootstrap_results['std_score']:.4f}")
        print(f"  95% Confidence interval: {bootstrap_results['confidence_interval']}")
    else:
        print(f"  Error: {bootstrap_results['error']}")
except Exception as e:
    print(f"  Error: {e}")

# 4. Bayesian Optimization
print("\n[5] Bayesian Optimization")
print("-" * 80)
try:
    bayesian_opt = toolbox.algorithms.get_bayesian_optimizer(n_calls=20)
    
    best_params = bayesian_opt.optimize(
        model_class=RandomForestClassifier,
        X=X_train,
        y=y_train,
        param_space={
            'n_estimators': (50, 200),
            'max_depth': (5, 20)
        },
        n_iterations=20,  # Reduced for demo
        task_type='classification',
        cv=3
    )
    
    if 'error' not in best_params:
        print(f"  Best parameters: {best_params['best_params']}")
        print(f"  Best score: {best_params['best_score']:.4f}")
        print(f"  Method: {best_params['method']}")
    else:
        print(f"  Error: {best_params['error']}")
        if 'fallback' in best_params:
            print(f"  Fallback: {best_params['fallback']}")
except Exception as e:
    print(f"  Error: {e}")

# 5. Statistical Feature Selection
print("\n[6] Statistical Feature Selection")
print("-" * 80)
try:
    feature_selector = toolbox.algorithms.get_statistical_feature_selector()
    
    # Mutual information selection
    mi_results = feature_selector.mutual_information_selection(
        X=X_train,
        y=y_train,
        k=10,
        task_type='classification'
    )
    
    if 'error' not in mi_results:
        print(f"  Selected features (top 10): {mi_results['selected_features']}")
        print(f"  Top scores: {mi_results['top_k_scores'][:5]}")
    
    # F-test selection
    f_results = feature_selector.f_test_selection(
        X=X_train,
        y=y_train,
        alpha=0.05,
        k=10,
        task_type='classification'
    )
    
    if 'error' not in f_results:
        print(f"  Significant features (F-test, alpha=0.05): {len(f_results['significant_features'])}")
        print(f"  Selected features: {f_results['selected_features'][:10]}")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "="*80)
print("DEMO COMPLETE")
print("="*80)
print("\nNote: Some methods require optional dependencies:")
print("  - scipy: pip install scipy (for statistical tests)")
print("  - scikit-optimize: pip install scikit-optimize (for Bayesian optimization)")
