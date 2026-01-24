"""
Run ML Toolbox Use Cases
Demonstrates semantic search, time series prediction, and agent workflows
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from ml_toolbox import MLToolbox
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("ML TOOLBOX USE CASES DEMONSTRATION")
print("="*80)

# Initialize toolbox
toolbox = MLToolbox()

# ============================================================================
# USE CASE 1: Semantic Search & Reasoning
# ============================================================================

def run_semantic_search():
    """Demonstrate semantic search with Quantum Kernel"""
    print("\n" + "="*80)
    print("USE CASE 1: SEMANTIC SEARCH & REASONING")
    print("="*80)
    
    try:
        # Access quantum kernel through infrastructure compartment
        infrastructure = toolbox.infrastructure
        quantum_kernel = infrastructure.quantum_kernel if hasattr(infrastructure, 'quantum_kernel') else None
        
        if quantum_kernel is None:
            print("Quantum Kernel not available through standard API")
            print("Note: Semantic search requires Quantum Kernel from infrastructure compartment")
            return False
        
        documents = [
            "Machine learning algorithms for classification",
            "Deep neural networks for image recognition",
            "Statistical methods for data analysis",
            "Natural language processing with transformers",
            "Computer vision using convolutional networks",
            "Reinforcement learning for game playing",
            "Clustering algorithms for unsupervised learning"
        ]
        
        print("\nBuilding semantic embeddings...")
        embeddings = []
        for doc in documents:
            embedding = quantum_kernel.compute_embedding(doc)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Semantic search
        def semantic_search(query, documents, embeddings, top_k=3):
            query_embedding = quantum_kernel.compute_embedding(query)
            similarities = quantum_kernel.compute_similarity(
                query_embedding.reshape(1, -1),
                embeddings
            )[0]
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    'document': documents[idx],
                    'similarity': float(similarities[idx]),
                    'rank': len(results) + 1
                })
            
            return results
        
        # Test searches
        queries = [
            "How to classify images?",
            "What are neural networks?",
            "Unsupervised learning methods"
        ]
        
        print("\nSemantic Search Results:")
        print("-" * 80)
        for query in queries:
            print(f"\nQuery: '{query}'")
            results = semantic_search(query, documents, embeddings, top_k=3)
            for result in results:
                print(f"  [{result['rank']}] Similarity: {result['similarity']:.4f}")
                print(f"      {result['document']}")
        
        return True
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return False

# ============================================================================
# USE CASE 2: Time Series & Tabular Prediction
# ============================================================================

def run_time_series_prediction():
    """Demonstrate time series prediction with preprocessing and interpretability"""
    print("\n" + "="*80)
    print("USE CASE 2: TIME SERIES PREDICTION WITH INTERPRETABILITY")
    print("="*80)
    
    try:
        # Generate time series data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        trend = np.linspace(100, 200, len(dates))
        seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        noise = np.random.normal(0, 5, len(dates))
        sales = trend + seasonality + noise
        
        data = pd.DataFrame({
            'date': dates,
            'sales': sales,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'quarter': dates.quarter,
            'is_weekend': (dates.dayofweek >= 5).astype(int)
        })
        
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        X_train = train_data[['day_of_week', 'month', 'quarter', 'is_weekend']].values
        y_train = train_data['sales'].values
        X_test = test_data[['day_of_week', 'month', 'quarter', 'is_weekend']].values
        y_test = test_data['sales'].values
        
        print(f"\nTraining data: {len(X_train)} samples")
        print(f"Test data: {len(X_test)} samples")
        
        # Train model
        print("\nTraining model...")
        result = toolbox.fit(
            X_train, y_train,
            task_type='regression'
        )
        
        model = result['model']
        print(f"Model type: {result.get('model_type', 'Unknown')}")
        
        # Predictions
        print("\nMaking predictions...")
        predictions = toolbox.predict(model, X_test)
        
        # Evaluate
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        
        print(f"\nModel Performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MSE: {mse:.4f}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            print(f"\nFeature Importance:")
            feature_names = ['day_of_week', 'month', 'quarter', 'is_weekend']
            importances = model.feature_importances_
            for name, importance in zip(feature_names, importances):
                print(f"  {name}: {importance:.4f}")
        
        return True
    except Exception as e:
        print(f"Error in time series prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_tabular_prediction():
    """Demonstrate tabular prediction with preprocessing and interpretability"""
    print("\n" + "="*80)
    print("USE CASE 2B: TABULAR PREDICTION WITH INTERPRETABILITY")
    print("="*80)
    
    try:
        # Generate sample data
        np.random.seed(42)
        n_samples = 500
        
        data = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'subscription_months': np.random.randint(1, 60, n_samples),
            'monthly_charges': np.random.uniform(20, 100, n_samples),
            'total_charges': np.random.uniform(100, 5000, n_samples),
            'contract_type': np.random.choice([0, 1, 2], n_samples),
            'payment_method': np.random.choice([0, 1, 2], n_samples),
            'tenure': np.random.randint(1, 72, n_samples),
            'has_tech_support': np.random.choice([0, 1], n_samples),
            'has_online_security': np.random.choice([0, 1], n_samples)
        })
        
        # Create target
        data['churn'] = (
            (data['subscription_months'] < 12) |
            (data['monthly_charges'] > 80) |
            (data['contract_type'] == 0)
        ).astype(int)
        
        feature_cols = [
            'age', 'subscription_months', 'monthly_charges', 'total_charges',
            'contract_type', 'payment_method', 'tenure',
            'has_tech_support', 'has_online_security'
        ]
        
        X = data[feature_cols].values
        y = data['churn'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining data: {len(X_train)} samples, {X_train.shape[1]} features")
        print(f"Test data: {len(X_test)} samples")
        
        # Train model
        print("\nTraining model...")
        result = toolbox.fit(
            X_train, y_train,
            task_type='classification'
        )
        
        model = result['model']
        print(f"Model type: {result.get('model_type', 'Unknown')}")
        
        # Predictions
        print("\nMaking predictions...")
        predictions = toolbox.predict(model, X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nModel Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            print(f"\nFeature Importance:")
            importances = model.feature_importances_
            feature_importance = list(zip(feature_cols, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for feature, importance in feature_importance:
                print(f"  {feature}: {importance:.4f}")
        
        return True
    except Exception as e:
        print(f"Error in tabular prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# USE CASE 3: Agent-like Workflow
# ============================================================================

def run_agent_workflow():
    """Demonstrate agent-like workflow"""
    print("\n" + "="*80)
    print("USE CASE 3: AGENT-LIKE WORKFLOW")
    print("="*80)
    
    try:
        # Create sample data
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 5)
        
        print(f"\nData prepared:")
        print(f"  Training: {X_train.shape}")
        print(f"  Test: {X_test.shape}")
        
        # Simple agent workflow
        class SimpleAgent:
            def __init__(self, toolbox):
                self.toolbox = toolbox
                self.history = []
            
            def understand_task(self, task_description):
                task_type = None
                if 'classify' in task_description.lower():
                    task_type = 'classification'
                elif 'predict' in task_description.lower() or 'regression' in task_description.lower():
                    task_type = 'regression'
                
                return {
                    'task_type': task_type,
                    'description': task_description
                }
            
            def execute_task(self, understanding, X_train, y_train, X_test):
                task_type = understanding.get('task_type', 'classification')
                
                print(f"\n[Agent] Executing {task_type} task...")
                
                result = self.toolbox.fit(
                    X_train, y_train,
                    task_type=task_type
                )
                
                model = result['model']
                predictions = self.toolbox.predict(model, X_test)
                
                return {
                    'success': True,
                    'model': model,
                    'predictions': predictions,
                    'model_type': result.get('model_type')
                }
        
        # Run agent
        agent = SimpleAgent(toolbox)
        
        task = "Build a classification model to predict binary outcomes"
        print(f"\n[Agent] Task: {task}")
        
        understanding = agent.understand_task(task)
        print(f"[Agent] Understood as: {understanding['task_type']}")
        
        result = agent.execute_task(understanding, X_train, y_train, X_test)
        
        if result['success']:
            print(f"[Agent] Task completed successfully!")
            print(f"[Agent] Model type: {result['model_type']}")
            print(f"[Agent] Predictions shape: {result['predictions'].shape}")
        else:
            print(f"[Agent] Task failed")
        
        return True
    except Exception as e:
        print(f"Error in agent workflow: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    results = {
        'semantic_search': False,
        'time_series': False,
        'tabular': False,
        'agent': False
    }
    
    # Run use cases
    print("\nRunning use cases...")
    print("="*80)
    
    results['semantic_search'] = run_semantic_search()
    results['time_series'] = run_time_series_prediction()
    results['tabular'] = run_tabular_prediction()
    results['agent'] = run_agent_workflow()
    
    # Summary
    print("\n" + "="*80)
    print("USE CASES SUMMARY")
    print("="*80)
    for use_case, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"{status} {use_case.replace('_', ' ').title()}")
    
    print("\n" + "="*80)
    print("All use cases completed!")
    print("="*80)
