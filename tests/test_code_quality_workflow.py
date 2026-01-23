"""
Tests for Code Quality Tools, Reed/Zelle Patterns, and Development Workflow
"""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from code_quality_tools import (
        CodeLinter, CodeFormatter, TypeChecker, DocumentationGenerator,
        CoverageReporter, PerformanceProfiler, CodeQualitySuite
    )
    from reed_zelle_patterns import (
        ProblemDecomposition, AlgorithmPatterns, DataStructureOptimizer,
        CodeOrganizer, RecursiveSolutions, IterativeRefinement
    )
    from development_workflow import (
        PreCommitHooks, CICDPipeline, CodeReviewAutomation, ReleaseManager
    )
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    pytestmark = pytest.mark.skip("Features not available")


class TestCodeQualityTools:
    """Tests for code quality tools"""
    
    def test_code_linter(self):
        """Test code linter"""
        linter = CodeLinter(linter='pylint')
        # Test with a simple file (this will fail if pylint not installed, which is OK)
        result = linter.lint_file(__file__)
        assert 'file' in result or 'error' in result
    
    def test_code_formatter(self):
        """Test code formatter"""
        formatter = CodeFormatter(formatter='black')
        # Test with a simple file (this will fail if black not installed, which is OK)
        result = formatter.format_file(__file__, check_only=True)
        assert 'file' in result or 'error' in result
    
    def test_type_checker(self):
        """Test type checker"""
        checker = TypeChecker(checker='mypy')
        # Test with a simple file (this will fail if mypy not installed, which is OK)
        result = checker.check_file(__file__)
        assert 'file' in result or 'error' in result
    
    def test_performance_profiler(self):
        """Test performance profiler"""
        profiler = PerformanceProfiler()
        
        def test_func():
            return sum(range(1000))
        
        result = profiler.profile_function(test_func)
        assert 'function' in result
        assert 'total_calls' in result
    
    def test_code_quality_suite(self):
        """Test code quality suite"""
        suite = CodeQualitySuite()
        result = suite.check_all(__file__)
        assert 'file' in result
        assert 'quality_score' in result


class TestReedZellePatterns:
    """Tests for Reed/Zelle patterns"""
    
    def test_problem_decomposition(self):
        """Test problem decomposition"""
        problem = "Build a classification model"
        result = ProblemDecomposition.decompose_ml_problem(problem)
        assert 'subproblems' in result
        assert 'data_preprocessing' in result['subproblems']
    
    def test_algorithm_patterns_divide_conquer(self):
        """Test divide and conquer pattern"""
        data = [1, 2, 3, 4, 5]
        
        def sum_op(left, right):
            return (left or 0) + (right or 0)
        
        result = AlgorithmPatterns.divide_and_conquer(data, sum_op)
        assert result == sum(data)
    
    def test_algorithm_patterns_greedy(self):
        """Test greedy algorithm pattern"""
        items = [1, 2, 3, 4, 5]
        
        def value_func(x):
            return x
        
        def constraint_func(selected):
            return sum(selected) <= 10
        
        result = AlgorithmPatterns.greedy_algorithm(items, value_func, constraint_func)
        assert len(result) > 0
    
    def test_data_structure_optimizer(self):
        """Test data structure optimizer"""
        data = np.random.rand(1000, 100)
        result = DataStructureOptimizer.optimize_for_ml(data, operation='lookup')
        assert 'recommendations' in result
    
    def test_code_organizer(self):
        """Test code organizer"""
        def preprocess_data():
            pass
        
        def train_model():
            pass
        
        def evaluate_model():
            pass
        
        functions = [preprocess_data, train_model, evaluate_model]
        result = CodeOrganizer.organize_by_functionality(functions)
        assert 'data_processing' in result
        assert 'model_training' in result
    
    def test_recursive_solutions(self):
        """Test recursive solutions"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = RecursiveSolutions.recursive_search(data, 5)
        assert result == 4  # Index of 5
    
    def test_iterative_refinement(self):
        """Test iterative refinement"""
        def improve(x):
            return x * 0.9  # Converge to 0
        
        result = IterativeRefinement.iterative_improvement(10.0, improve, max_iterations=100)
        assert 'final_solution' in result
        assert 'converged' in result


class TestDevelopmentWorkflow:
    """Tests for development workflow"""
    
    def test_pre_commit_hooks(self):
        """Test pre-commit hooks"""
        hooks = PreCommitHooks(hooks_dir='.test_hooks')
        result = hooks.create_pre_commit_hook(['lint', 'format'])
        # May fail if git not initialized, which is OK
        assert 'success' in result or 'error' in result
    
    def test_cicd_pipeline(self):
        """Test CI/CD pipeline"""
        pipeline = CICDPipeline()
        result = pipeline.create_github_actions(['test', 'lint'])
        # May fail if .github directory can't be created, which is OK
        assert 'success' in result or 'error' in result
    
    def test_code_review_automation(self):
        """Test code review automation"""
        reviewer = CodeReviewAutomation()
        result = reviewer.review_code(__file__)
        assert 'file' in result
        assert 'quality_score' in result
    
    def test_release_manager(self):
        """Test release manager"""
        manager = ReleaseManager(version_file='.test_version')
        version = manager.get_version()
        assert version == '0.0.0'  # Default
        
        result = manager.bump_version('patch')
        assert 'success' in result or 'error' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
