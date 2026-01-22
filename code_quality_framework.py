"""
Code Quality Framework - Steve McConnell "Code Complete" Methods
Systematic code quality practices, metrics, and standards

Methods from:
- Code Complete: Code quality, design patterns, error handling, refactoring
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable, Set
import ast
import inspect
import re
import warnings

sys.path.insert(0, str(Path(__file__).parent))


class CodeQualityMetrics:
    """
    Code Quality Metrics (Code Complete)
    
    Measure code quality using various metrics
    """
    
    @staticmethod
    def cyclomatic_complexity(func: Callable) -> int:
        """
        Calculate Cyclomatic Complexity
        
        Measures code complexity (lower is better)
        
        Args:
            func: Function to analyze
            
        Returns:
            Cyclomatic complexity score
        """
        try:
            source = inspect.getsource(func)
            tree = ast.parse(source)
            
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                elif isinstance(node, ast.Compare):
                    complexity += len(node.ops) - 1
            
            return complexity
        except:
            return 0
    
    @staticmethod
    def maintainability_index(func: Callable) -> float:
        """
        Calculate Maintainability Index (simplified)
        
        Higher is better (0-100 scale)
        
        Args:
            func: Function to analyze
            
        Returns:
            Maintainability index
        """
        try:
            source = inspect.getsource(func)
            lines = len(source.split('\n'))
            
            # Simplified: based on length and complexity
            complexity = CodeQualityMetrics.cyclomatic_complexity(func)
            
            # Maintainability = 171 - 5.2 * ln(avg_lines) - 0.23 * complexity
            # Simplified version
            mi = max(0, 100 - (lines / 10) - (complexity * 2))
            return min(100, mi)
        except:
            return 50.0
    
    @staticmethod
    def code_duplication_ratio(functions: List[Callable]) -> float:
        """
        Calculate Code Duplication Ratio
        
        Measures how much code is duplicated
        
        Args:
            functions: List of functions to analyze
            
        Returns:
            Duplication ratio (0-1)
        """
        if len(functions) < 2:
            return 0.0
        
        signatures = []
        for func in functions:
            try:
                source = inspect.getsource(func)
                # Extract function signature and first few lines
                sig = source.split('\n')[0] if source else ''
                signatures.append(sig)
            except:
                continue
        
        # Count duplicates
        unique = len(set(signatures))
        total = len(signatures)
        
        return 1.0 - (unique / total) if total > 0 else 0.0
    
    @staticmethod
    def function_length(func: Callable) -> int:
        """Get function length in lines"""
        try:
            source = inspect.getsource(func)
            return len([line for line in source.split('\n') if line.strip()])
        except:
            return 0
    
    @staticmethod
    def parameter_count(func: Callable) -> int:
        """Get number of parameters"""
        try:
            sig = inspect.signature(func)
            return len(sig.parameters)
        except:
            return 0
    
    @staticmethod
    def calculate_quality_score(func: Callable) -> Dict[str, Any]:
        """
        Calculate overall quality score for a function
        
        Args:
            func: Function to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        complexity = CodeQualityMetrics.cyclomatic_complexity(func)
        maintainability = CodeQualityMetrics.maintainability_index(func)
        length = CodeQualityMetrics.function_length(func)
        params = CodeQualityMetrics.parameter_count(func)
        
        # Quality score (0-100)
        quality = maintainability
        if complexity > 10:
            quality -= 10
        if length > 50:
            quality -= 10
        if params > 5:
            quality -= 5
        
        return {
            'cyclomatic_complexity': complexity,
            'maintainability_index': maintainability,
            'function_length': length,
            'parameter_count': params,
            'quality_score': max(0, min(100, quality)),
            'recommendations': CodeQualityMetrics._get_recommendations(
                complexity, length, params
            )
        }
    
    @staticmethod
    def _get_recommendations(complexity: int, length: int, params: int) -> List[str]:
        """Get quality recommendations"""
        recommendations = []
        
        if complexity > 10:
            recommendations.append("Consider refactoring to reduce complexity")
        if length > 50:
            recommendations.append("Consider splitting into smaller functions")
        if params > 5:
            recommendations.append("Consider using a configuration object")
        
        return recommendations


class DesignPatterns:
    """
    Design Patterns (Code Complete)
    
    Common design patterns for ML workflows
    """
    
    class ModelFactory:
        """Factory Pattern for Model Creation"""
        
        @staticmethod
        def create_model(model_type: str, **kwargs) -> Any:
            """
            Create model instance using factory pattern
            
            Args:
                model_type: Type of model ('random_forest', 'svm', etc.)
                **kwargs: Model parameters
                
            Returns:
                Model instance
            """
            factories = {
                'random_forest': lambda: DesignPatterns.ModelFactory._create_random_forest(**kwargs),
                'svm': lambda: DesignPatterns.ModelFactory._create_svm(**kwargs),
                'logistic_regression': lambda: DesignPatterns.ModelFactory._create_logistic(**kwargs),
                'neural_network': lambda: DesignPatterns.ModelFactory._create_neural_network(**kwargs)
            }
            
            if model_type not in factories:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return factories[model_type]()
        
        @staticmethod
        def _create_random_forest(**kwargs):
            try:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**kwargs)
            except ImportError:
                return None
        
        @staticmethod
        def _create_svm(**kwargs):
            try:
                from sklearn.svm import SVC
                return SVC(**kwargs)
            except ImportError:
                return None
        
        @staticmethod
        def _create_logistic(**kwargs):
            try:
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(**kwargs)
            except ImportError:
                return None
        
        @staticmethod
        def _create_neural_network(**kwargs):
            try:
                import torch.nn as nn
                return nn.Sequential(
                    nn.Linear(kwargs.get('input_size', 10), kwargs.get('hidden_size', 20)),
                    nn.ReLU(),
                    nn.Linear(kwargs.get('hidden_size', 20), kwargs.get('output_size', 2))
                )
            except ImportError:
                return None
    
    class Strategy:
        """Strategy Pattern for Algorithm Selection"""
        
        def __init__(self, strategy: Callable):
            """
            Args:
                strategy: Strategy function
            """
            self.strategy = strategy
        
        def execute(self, *args, **kwargs) -> Any:
            """Execute strategy"""
            return self.strategy(*args, **kwargs)
    
    class Observer:
        """Observer Pattern for Event Handling"""
        
        def __init__(self):
            self._observers: List[Callable] = []
        
        def attach(self, observer: Callable):
            """Attach observer"""
            self._observers.append(observer)
        
        def detach(self, observer: Callable):
            """Detach observer"""
            if observer in self._observers:
                self._observers.remove(observer)
        
        def notify(self, event: str, data: Any = None):
            """Notify all observers"""
            for observer in self._observers:
                try:
                    observer(event, data)
                except Exception as e:
                    warnings.warn(f"Observer error: {e}")


class AdvancedErrorHandling:
    """
    Advanced Error Handling (Code Complete)
    
    Robust error handling with recovery and classification
    """
    
    class ErrorClassifier:
        """Classify errors by type and severity"""
        
        ERROR_TYPES = {
            'validation': ['ValueError', 'TypeError', 'ValidationError'],
            'resource': ['MemoryError', 'IOError', 'FileNotFoundError'],
            'network': ['ConnectionError', 'TimeoutError'],
            'computation': ['ZeroDivisionError', 'OverflowError'],
            'system': ['OSError', 'PermissionError']
        }
        
        SEVERITY_LEVELS = {
            'critical': ['MemoryError', 'SystemError'],
            'high': ['ValueError', 'TypeError', 'IOError'],
            'medium': ['Warning', 'UserWarning'],
            'low': ['DeprecationWarning']
        }
        
        @staticmethod
        def classify_error(error: Exception) -> Dict[str, str]:
            """
            Classify error by type and severity
            
            Args:
                error: Exception to classify
                
            Returns:
                Dictionary with classification
            """
            error_type = type(error).__name__
            
            # Find error type
            error_category = 'unknown'
            for category, types in AdvancedErrorHandling.ErrorClassifier.ERROR_TYPES.items():
                if error_type in types:
                    error_category = category
                    break
            
            # Find severity
            severity = 'medium'
            for sev, types in AdvancedErrorHandling.ErrorClassifier.SEVERITY_LEVELS.items():
                if error_type in types:
                    severity = sev
                    break
            
            return {
                'error_type': error_type,
                'category': error_category,
                'severity': severity,
                'message': str(error)
            }
    
    class ErrorRecovery:
        """Error recovery strategies"""
        
        @staticmethod
        def retry_with_backoff(
            func: Callable,
            max_retries: int = 3,
            backoff_factor: float = 2.0,
            exceptions: Tuple = (Exception,)
        ) -> Any:
            """
            Retry function with exponential backoff
            
            Args:
                func: Function to retry
                max_retries: Maximum retry attempts
                backoff_factor: Backoff multiplier
                exceptions: Exceptions to catch
                
            Returns:
                Function result
            """
            import time
            
            for attempt in range(max_retries):
                try:
                    return func()
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    time.sleep(wait_time)
            
            raise RuntimeError("Max retries exceeded")
        
        @staticmethod
        def fallback_value(
            func: Callable,
            fallback: Any,
            exceptions: Tuple = (Exception,)
        ) -> Any:
            """
            Execute function with fallback value
            
            Args:
                func: Function to execute
                fallback: Fallback value on error
                exceptions: Exceptions to catch
                
            Returns:
                Function result or fallback
            """
            try:
                return func()
            except exceptions:
                return fallback
        
        @staticmethod
        def graceful_degradation(
            primary: Callable,
            fallback: Callable,
            exceptions: Tuple = (Exception,)
        ) -> Any:
            """
            Try primary, fallback to secondary
            
            Args:
                primary: Primary function
                fallback: Fallback function
                exceptions: Exceptions to catch
                
            Returns:
                Result from primary or fallback
            """
            try:
                return primary()
            except exceptions:
                return fallback()


class CodeSmellDetector:
    """
    Code Smell Detection (Code Complete)
    
    Detect common code quality issues
    """
    
    @staticmethod
    def detect_long_method(func: Callable, threshold: int = 50) -> bool:
        """Detect long methods"""
        length = CodeQualityMetrics.function_length(func)
        return length > threshold
    
    @staticmethod
    def detect_long_parameter_list(func: Callable, threshold: int = 5) -> bool:
        """Detect long parameter lists"""
        params = CodeQualityMetrics.parameter_count(func)
        return params > threshold
    
    @staticmethod
    def detect_high_complexity(func: Callable, threshold: int = 10) -> bool:
        """Detect high cyclomatic complexity"""
        complexity = CodeQualityMetrics.cyclomatic_complexity(func)
        return complexity > threshold
    
    @staticmethod
    def detect_duplicate_code(functions: List[Callable], threshold: float = 0.3) -> bool:
        """Detect code duplication"""
        ratio = CodeQualityMetrics.code_duplication_ratio(functions)
        return ratio > threshold
    
    @staticmethod
    def detect_code_smells(func: Callable) -> List[str]:
        """
        Detect all code smells in a function
        
        Args:
            func: Function to analyze
            
        Returns:
            List of detected code smells
        """
        smells = []
        
        if CodeSmellDetector.detect_long_method(func):
            smells.append("Long method (consider splitting)")
        
        if CodeSmellDetector.detect_long_parameter_list(func):
            smells.append("Long parameter list (consider configuration object)")
        
        if CodeSmellDetector.detect_high_complexity(func):
            smells.append("High complexity (consider refactoring)")
        
        return smells


class RefactoringTools:
    """
    Refactoring Tools (Code Complete)
    
    Tools for safe code refactoring
    """
    
    @staticmethod
    def extract_method_suggestion(func: Callable) -> List[str]:
        """
        Suggest methods to extract
        
        Args:
            func: Function to analyze
            
        Returns:
            List of suggested extractions
        """
        suggestions = []
        
        try:
            source = inspect.getsource(func)
            lines = source.split('\n')
            
            # Find long blocks (potential extractions)
            block_start = None
            block_length = 0
            
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('#'):
                    if block_start is None:
                        block_start = i
                        block_length = 1
                    else:
                        block_length += 1
                else:
                    if block_length > 10 and block_start:
                        suggestions.append(
                            f"Lines {block_start}-{i}: Consider extracting into separate method"
                        )
                        block_start = None
                        block_length = 0
        except:
            pass
        
        return suggestions
    
    @staticmethod
    def rename_variable_suggestion(func: Callable) -> List[str]:
        """
        Suggest variable renames for clarity
        
        Args:
            func: Function to analyze
            
        Returns:
            List of rename suggestions
        """
        suggestions = []
        
        try:
            source = inspect.getsource(func)
            tree = ast.parse(source)
            
            # Find single-letter variables (potential renames)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    if len(node.id) == 1 and node.id.islower():
                        suggestions.append(f"Consider renaming '{node.id}' to more descriptive name")
        except:
            pass
        
        return suggestions


class CodeCompleteFramework:
    """
    Unified Code Complete Framework
    """
    
    def __init__(self):
        self.quality_metrics = CodeQualityMetrics()
        self.design_patterns = DesignPatterns()
        self.error_handling = AdvancedErrorHandling()
        self.smell_detector = CodeSmellDetector()
        self.refactoring = RefactoringTools()
    
    def analyze_function(self, func: Callable) -> Dict[str, Any]:
        """
        Complete code quality analysis
        
        Args:
            func: Function to analyze
            
        Returns:
            Complete analysis report
        """
        quality = self.quality_metrics.calculate_quality_score(func)
        smells = self.smell_detector.detect_code_smells(func)
        refactor_suggestions = self.refactoring.extract_method_suggestion(func)
        
        return {
            'quality_metrics': quality,
            'code_smells': smells,
            'refactoring_suggestions': refactor_suggestions,
            'overall_grade': self._calculate_grade(quality['quality_score'], len(smells))
        }
    
    def _calculate_grade(self, quality_score: float, smell_count: int) -> str:
        """Calculate overall grade"""
        score = quality_score - (smell_count * 10)
        
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'python': 'Python 3.8+'
        }
