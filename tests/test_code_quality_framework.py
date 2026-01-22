"""
Tests for Code Quality Framework
Test Code Complete methods: quality metrics, design patterns, error handling
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from code_quality_framework import (
        CodeQualityMetrics,
        DesignPatterns,
        AdvancedErrorHandling,
        CodeSmellDetector,
        RefactoringTools,
        CodeCompleteFramework
    )
    CODE_QUALITY_AVAILABLE = True
except ImportError:
    CODE_QUALITY_AVAILABLE = False
    pytestmark = pytest.mark.skip("Code quality framework not available")


class TestCodeQualityMetrics:
    """Tests for code quality metrics"""
    
    def simple_function(self):
        """Simple function for testing"""
        return 1 + 1
    
    def complex_function(self):
        """Complex function for testing"""
        if True:
            if False:
                for i in range(10):
                    if i > 5:
                        while i < 10:
                            i += 1
        return 0
    
    def test_cyclomatic_complexity(self):
        """Test cyclomatic complexity"""
        simple = CodeQualityMetrics.cyclomatic_complexity(self.simple_function)
        complex_func = CodeQualityMetrics.cyclomatic_complexity(self.complex_function)
        
        assert simple >= 1
        assert complex_func > simple
    
    def test_maintainability_index(self):
        """Test maintainability index"""
        mi = CodeQualityMetrics.maintainability_index(self.simple_function)
        assert 0 <= mi <= 100
    
    def test_function_length(self):
        """Test function length"""
        length = CodeQualityMetrics.function_length(self.simple_function)
        assert length > 0
    
    def test_parameter_count(self):
        """Test parameter count"""
        def func_with_params(a, b, c):
            return a + b + c
        
        count = CodeQualityMetrics.parameter_count(func_with_params)
        assert count == 3
    
    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        score = CodeQualityMetrics.calculate_quality_score(self.simple_function)
        assert 'quality_score' in score
        assert 'cyclomatic_complexity' in score
        assert 'recommendations' in score


class TestDesignPatterns:
    """Tests for design patterns"""
    
    def test_model_factory(self):
        """Test model factory"""
        try:
            model = DesignPatterns.ModelFactory.create_model('random_forest', n_estimators=10)
            # May return None if sklearn not available
            assert model is None or hasattr(model, 'fit')
        except ValueError:
            # Unknown model type
            pass
    
    def test_strategy_pattern(self):
        """Test strategy pattern"""
        def strategy1(x):
            return x * 2
        
        strategy = DesignPatterns.Strategy(strategy1)
        result = strategy.execute(5)
        assert result == 10
    
    def test_observer_pattern(self):
        """Test observer pattern"""
        events = []
        
        def observer(event, data):
            events.append((event, data))
        
        obs = DesignPatterns.Observer()
        obs.attach(observer)
        obs.notify('test_event', {'data': 123})
        
        assert len(events) == 1
        assert events[0][0] == 'test_event'


class TestAdvancedErrorHandling:
    """Tests for advanced error handling"""
    
    def test_error_classification(self):
        """Test error classification"""
        error = ValueError("Invalid input")
        classification = AdvancedErrorHandling.ErrorClassifier.classify_error(error)
        
        assert 'error_type' in classification
        assert 'category' in classification
        assert 'severity' in classification
    
    def test_retry_with_backoff(self):
        """Test retry with backoff"""
        attempts = []
        
        def failing_func():
            attempts.append(1)
            raise ValueError("Error")
        
        try:
            AdvancedErrorHandling.ErrorRecovery.retry_with_backoff(
                failing_func, max_retries=2, exceptions=(ValueError,)
            )
        except ValueError:
            pass
        
        assert len(attempts) == 2
    
    def test_fallback_value(self):
        """Test fallback value"""
        def failing_func():
            raise ValueError("Error")
        
        result = AdvancedErrorHandling.ErrorRecovery.fallback_value(
            failing_func, fallback=42, exceptions=(ValueError,)
        )
        assert result == 42
    
    def test_graceful_degradation(self):
        """Test graceful degradation"""
        def primary():
            raise ValueError("Error")
        
        def fallback():
            return "fallback_result"
        
        result = AdvancedErrorHandling.ErrorRecovery.graceful_degradation(
            primary, fallback, exceptions=(ValueError,)
        )
        assert result == "fallback_result"


class TestCodeSmellDetector:
    """Tests for code smell detection"""
    
    def short_function(self):
        """Short function"""
        return 1
    
    def long_function(self):
        """Long function for testing"""
        a = 1
        b = 2
        c = 3
        d = 4
        e = 5
        f = 6
        g = 7
        h = 8
        i = 9
        j = 10
        k = 11
        l = 12
        m = 13
        n = 14
        o = 15
        p = 16
        q = 17
        r = 18
        s = 19
        t = 20
        u = 21
        v = 22
        w = 23
        x = 24
        y = 25
        z = 26
        return a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u + v + w + x + y + z
    
    def test_detect_long_method(self):
        """Test long method detection"""
        assert not CodeSmellDetector.detect_long_method(self.short_function)
        assert CodeSmellDetector.detect_long_method(self.long_function, threshold=20)
    
    def test_detect_code_smells(self):
        """Test code smell detection"""
        smells = CodeSmellDetector.detect_code_smells(self.long_function)
        assert isinstance(smells, list)


class TestRefactoringTools:
    """Tests for refactoring tools"""
    
    def test_function(self):
        """Test function for refactoring"""
        a = 1
        b = 2
        c = 3
        d = 4
        e = 5
        f = 6
        g = 7
        h = 8
        i = 9
        j = 10
        return a + b + c + d + e + f + g + h + i + j
    
    def test_extract_method_suggestion(self):
        """Test extract method suggestions"""
        suggestions = RefactoringTools.extract_method_suggestion(self.test_function)
        assert isinstance(suggestions, list)


class TestCodeCompleteFramework:
    """Test unified framework"""
    
    def test_function(self):
        """Test function"""
        return 42
    
    def test_unified_interface(self):
        """Test CodeCompleteFramework"""
        framework = CodeCompleteFramework()
        
        assert framework.quality_metrics is not None
        assert framework.design_patterns is not None
        assert framework.error_handling is not None
    
    def test_analyze_function(self):
        """Test function analysis"""
        framework = CodeCompleteFramework()
        
        def test_func():
            return 1 + 1
        
        analysis = framework.analyze_function(test_func)
        assert 'quality_metrics' in analysis
        assert 'code_smells' in analysis
        assert 'overall_grade' in analysis


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
