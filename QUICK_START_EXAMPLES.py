"""
ML Toolbox - Quick Start Examples
Run these examples to get started with ML Toolbox
"""

print("="*80)
print("ML TOOLBOX - QUICK START EXAMPLES")
print("="*80)
print()

# Example 1: Basic Classification
print("Example 1: Basic Classification")
print("-" * 80)
try:
    from ml_toolbox import MLToolbox
    import numpy as np
    
    toolbox = MLToolbox()
    
    # Create sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 1, 0, 1, 0, 1])
    
    # Train model
    result = toolbox.fit(X, y, task_type='classification')
    
    # Make predictions
    predictions = toolbox.predict(result['model'], [[7, 8], [8, 9]])
    
    print(f"✅ Success! Predictions: {predictions}")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Example 2: Data Preprocessing
print("Example 2: Data Preprocessing")
print("-" * 80)
try:
    from ml_toolbox import MLToolbox
    
    toolbox = MLToolbox()
    
    # Sample text data
    texts = [
        "Machine learning is amazing",
        "Deep learning uses neural networks",
        "ML and AI are transforming industries",
        "Neural networks are powerful",
        "AI is the future"
    ]
    
    # Preprocess with advanced features
    results = toolbox.data.preprocess(
        texts,
        advanced=True,
        dedup_threshold=0.85,
        enable_compression=True,
        compression_ratio=0.5
    )
    
    X = results['compressed_embeddings']
    print(f"✅ Success! Features shape: {X.shape}")
    print(f"   Original texts: {len(texts)}")
    print(f"   Features: {X.shape[0]} samples, {X.shape[1]} features")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Example 3: Regression
print("Example 3: Regression")
print("-" * 80)
try:
    from ml_toolbox import MLToolbox
    import numpy as np
    
    toolbox = MLToolbox()
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    # Train regressor
    result = toolbox.fit(X, y, task_type='regression')
    
    # Make predictions
    predictions = toolbox.predict(result['model'], X[:5])
    
    print(f"✅ Success! Predictions: {predictions[:3]}...")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Example 4: Model Evaluation
print("Example 4: Model Evaluation")
print("-" * 80)
try:
    from ml_toolbox import MLToolbox
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    toolbox = MLToolbox()
    
    # Generate data
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 2, 200)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    result = toolbox.fit(X_train, y_train, task_type='classification')
    
    # Evaluate
    evaluator = toolbox.algorithms.get_evaluator()
    metrics = evaluator.evaluate_model(
        model=result['model'],
        X=X_test,
        y=y_test
    )
    
    print(f"✅ Success! Evaluation metrics:")
    print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}" if 'accuracy' in metrics else "   Accuracy: N/A")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Example 5: AI Agent Code Generation
print("Example 5: AI Agent Code Generation")
print("-" * 80)
try:
    from ml_toolbox.ai_agent import MLCodeAgent
    
    agent = MLCodeAgent(use_llm=False)  # Use template-based for reliability
    
    # Generate code
    result = agent.build("Create a simple classifier")
    
    if result.get('success'):
        print("✅ Success! Code generated")
        print(f"   Code length: {len(result.get('code', ''))} characters")
    else:
        print(f"⚠️  Generation completed with warnings")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Example 6: Hyperparameter Tuning
print("Example 6: Hyperparameter Tuning")
print("-" * 80)
try:
    from ml_toolbox import MLToolbox
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    toolbox = MLToolbox()
    
    # Generate data
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 2, 200)
    
    # Get tuner
    tuner = toolbox.algorithms.get_tuner()
    
    # Tune hyperparameters
    best_params = tuner.tune(
        model=RandomForestClassifier(n_estimators=10, random_state=42),
        X=X,
        y=y,
        param_grid={
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        },
        cv=3
    )
    
    print(f"✅ Success! Best parameters found")
    print(f"   Parameters: {best_params}")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

print("="*80)
print("QUICK START EXAMPLES COMPLETE")
print("="*80)
print()
print("Next steps:")
print("1. Explore the documentation in the docs/ directory")
print("2. Try the ML Learning App: python ml_learning_app_simple.py")
print("3. Read the full README.md for more examples")
print("4. Check out REVOLUTIONARY_FEATURES_GUIDE.md for advanced features")
