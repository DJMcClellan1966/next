"""
Toolbox Knowledge Base
Structured knowledge of ML Toolbox capabilities, APIs, and patterns
"""
import sys
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import inspect

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ToolboxKnowledgeBase:
    """
    Knowledge Base of ML Toolbox capabilities
    
    Provides:
    - API documentation
    - Capability mapping
    - Code patterns
    - Best practices
    """
    
    def __init__(self):
        """Initialize knowledge base"""
        self.capabilities = self._load_capabilities()
        self.apis = self._load_apis()
        self.patterns = self._load_patterns()
        self.examples = self._load_examples()
    
    def _load_capabilities(self) -> Dict[str, Any]:
        """Load toolbox capabilities"""
        return {
            'data_preprocessing': {
                'description': 'Data preprocessing and transformation',
                'components': [
                    'AdvancedDataPreprocessor',
                    'ConventionalPreprocessor',
                    'CorpusCallosumPreprocessor',
                    'GPUAcceleratedPreprocessor',
                    'ModelSpecificPreprocessor'
                ],
                'use_cases': [
                    'Clean and preprocess data',
                    'Remove duplicates',
                    'Feature engineering',
                    'Dimensionality reduction',
                    'Data compression'
                ]
            },
            'algorithms': {
                'description': 'Machine learning algorithms',
                'components': [
                    'MLEvaluator',
                    'HyperparameterTuner',
                    'EnsembleLearner',
                    'StatisticalEvaluator'
                ],
                'use_cases': [
                    'Classification',
                    'Regression',
                    'Clustering',
                    'Feature selection',
                    'Model evaluation'
                ]
            },
            'optimization': {
                'description': 'Performance optimizations',
                'components': [
                    'MLMathOptimizer',
                    'ModelCache',
                    'MedullaToolboxOptimizer',
                    'ArchitectureOptimizer'
                ],
                'use_cases': [
                    'Faster matrix operations',
                    'Model caching',
                    'Resource optimization',
                    'Architecture-specific optimizations'
                ]
            },
            'mlops': {
                'description': 'MLOps and production features',
                'components': [
                    'ModelRegistry',
                    'ExperimentTracking',
                    'ModelMonitoring'
                ],
                'use_cases': [
                    'Model versioning',
                    'Experiment tracking',
                    'Model deployment',
                    'Performance monitoring'
                ]
            }
        }
    
    def _load_apis(self) -> Dict[str, Any]:
        """Load API documentation"""
        return {
            'MLToolbox': {
                'class': 'ml_toolbox.MLToolbox',
                'methods': {
                    'fit': {
                        'signature': 'fit(X, y, task_type="auto", model_type="auto", use_cache=True, **kwargs)',
                        'description': 'Train a model - unified simple API',
                        'returns': 'Dict with model and metrics',
                        'example': 'result = toolbox.fit(X, y)'
                    },
                    'predict': {
                        'signature': 'predict(model, X, use_cache=True)',
                        'description': 'Make predictions',
                        'returns': 'Predictions array',
                        'example': 'predictions = toolbox.predict(model, X)'
                    },
                    'register_model': {
                        'signature': 'register_model(model, model_name, version=None, metadata=None, stage="dev")',
                        'description': 'Register model in registry',
                        'returns': 'Model ID',
                        'example': 'model_id = toolbox.register_model(model, "my_model")'
                    },
                    'get_ml_math_optimizer': {
                        'signature': 'get_ml_math_optimizer()',
                        'description': 'Get ML Math Optimizer for optimized operations',
                        'returns': 'MLMathOptimizer instance',
                        'example': 'optimizer = toolbox.get_ml_math_optimizer()'
                    }
                }
            },
            'DataCompartment': {
                'class': 'ml_toolbox.compartment1_data.DataCompartment',
                'methods': {
                    'get_preprocessor': {
                        'signature': 'get_preprocessor(preprocessor_type="advanced")',
                        'description': 'Get data preprocessor',
                        'returns': 'Preprocessor instance',
                        'example': 'preprocessor = toolbox.data.get_preprocessor("advanced")'
                    }
                }
            },
            'AlgorithmsCompartment': {
                'class': 'ml_toolbox.compartment3_algorithms.AlgorithmsCompartment',
                'methods': {
                    'get_evaluator': {
                        'signature': 'get_evaluator()',
                        'description': 'Get ML evaluator',
                        'returns': 'MLEvaluator instance',
                        'example': 'evaluator = toolbox.algorithms.get_evaluator()'
                    }
                }
            }
        }
    
    def _load_patterns(self) -> List[Dict[str, Any]]:
        """Load code patterns"""
        return [
            {
                'name': 'simple_classification',
                'description': 'Simple classification task',
                'code': '''from ml_toolbox import MLToolbox
import numpy as np

# Initialize toolbox
toolbox = MLToolbox()

# Generate or load data
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

# Train model
result = toolbox.fit(X, y, task_type='classification')

# Get model and accuracy
model = result['model']
accuracy = result['accuracy']
print(f"Accuracy: {accuracy:.2%}")

# Make predictions
predictions = toolbox.predict(model, X[:10])
print(f"Predictions: {predictions}")'''
            },
            {
                'name': 'simple_regression',
                'description': 'Simple regression task',
                'code': '''from ml_toolbox import MLToolbox
import numpy as np

# Initialize toolbox
toolbox = MLToolbox()

# Generate or load data
X = np.random.randn(100, 10)
y = np.random.randn(100)

# Train model
result = toolbox.fit(X, y, task_type='regression')

# Get model and R² score
model = result['model']
r2 = result['r2_score']
print(f"R² Score: {r2:.3f}")

# Make predictions
predictions = toolbox.predict(model, X[:10])
print(f"Predictions: {predictions}")'''
            },
            {
                'name': 'with_preprocessing',
                'description': 'Classification with data preprocessing',
                'code': '''from ml_toolbox import MLToolbox
import numpy as np

# Initialize toolbox
toolbox = MLToolbox()

# Get preprocessor
preprocessor = toolbox.data.get_preprocessor('advanced')

# Preprocess data
raw_data = ["text1", "text2", "text3"]
processed = preprocessor.preprocess(raw_data)

# Extract features (simplified)
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

# Train model
result = toolbox.fit(X, y)
print(f"Accuracy: {result['accuracy']:.2%}")'''
            },
            {
                'name': 'with_model_registry',
                'description': 'Training and registering model',
                'code': '''from ml_toolbox import MLToolbox
import numpy as np

# Initialize toolbox
toolbox = MLToolbox()

# Train model
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)
result = toolbox.fit(X, y)

# Register model
model_id = toolbox.register_model(
    result['model'],
    model_name='my_classifier',
    metadata={'accuracy': result['accuracy']},
    stage='dev'
)

print(f"Model registered: {model_id}")'''
            },
            {
                'name': 'with_optimizations',
                'description': 'Using ML Math Optimizer',
                'code': '''from ml_toolbox import MLToolbox
import numpy as np

# Initialize toolbox (optimizations auto-enabled)
toolbox = MLToolbox()

# Get ML Math Optimizer
math_optimizer = toolbox.get_ml_math_optimizer()

# Use optimized operations
A = np.random.randn(100, 100)
B = np.random.randn(100, 100)

# Optimized matrix multiplication (15-20% faster)
C = math_optimizer.optimized_matrix_multiply(A, B)

# Optimized SVD (43-48% faster)
U, s, Vh = math_optimizer.optimized_svd(A, full_matrices=False)'''
            }
        ]
    
    def _load_examples(self) -> List[Dict[str, Any]]:
        """Load example use cases"""
        return [
            {
                'task': 'Classify iris flowers',
                'solution': 'simple_classification',
                'description': 'Use toolbox.fit() with classification task'
            },
            {
                'task': 'Predict house prices',
                'solution': 'simple_regression',
                'description': 'Use toolbox.fit() with regression task'
            },
            {
                'task': 'Preprocess text data',
                'solution': 'with_preprocessing',
                'description': 'Use AdvancedDataPreprocessor for text'
            },
            {
                'task': 'Register and version model',
                'solution': 'with_model_registry',
                'description': 'Use model registry for production'
            }
        ]
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get all capabilities"""
        return self.capabilities
    
    def find_solution(self, problem: str) -> List[Dict[str, Any]]:
        """
        Find relevant solutions for a problem
        
        Args:
            problem: Problem description
        
        Returns:
            List of relevant solutions
        """
        problem_lower = problem.lower()
        relevant = []
        
        # Search in examples
        for example in self.examples:
            if any(word in problem_lower for word in example['task'].lower().split()):
                relevant.append({
                    'type': 'example',
                    'task': example['task'],
                    'solution': example['solution'],
                    'description': example['description']
                })
        
        # Search in patterns
        for pattern in self.patterns:
            if any(word in problem_lower for word in pattern['name'].lower().split('_')):
                relevant.append({
                    'type': 'pattern',
                    'name': pattern['name'],
                    'description': pattern['description'],
                    'code': pattern['code']
                })
        
        return relevant
    
    def get_api_docs(self, component: str) -> Optional[Dict[str, Any]]:
        """
        Get API documentation for a component
        
        Args:
            component: Component name (e.g., 'MLToolbox', 'DataCompartment')
        
        Returns:
            API documentation or None
        """
        return self.apis.get(component)
    
    def get_pattern(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a code pattern by name
        
        Args:
            pattern_name: Pattern name
        
        Returns:
            Pattern dictionary or None
        """
        for pattern in self.patterns:
            if pattern['name'] == pattern_name:
                return pattern
        return None
    
    def get_all_patterns(self) -> List[Dict[str, Any]]:
        """Get all code patterns"""
        return self.patterns


# Global knowledge base instance
_global_kb: Optional[ToolboxKnowledgeBase] = None

def get_knowledge_base() -> ToolboxKnowledgeBase:
    """Get global knowledge base instance"""
    global _global_kb
    if _global_kb is None:
        _global_kb = ToolboxKnowledgeBase()
    return _global_kb
