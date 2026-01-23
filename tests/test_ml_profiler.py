"""
Tests for ML Profiler
"""
import sys
from pathlib import Path
import pytest
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_profiler import MLProfiler, PipelineProfiler, ProfiledMLToolbox, profile_ml_pipeline
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    pytestmark = pytest.mark.skip("Features not available")


class TestMLProfiler:
    """Tests for ML Profiler"""
    
    def test_profiler_initialization(self):
        """Test profiler initialization"""
        profiler = MLProfiler()
        assert profiler is not None
        assert profiler.enable_memory_profiling == False
    
    def test_profile_function_decorator(self):
        """Test function profiling decorator"""
        profiler = MLProfiler()
        
        @profiler.profile_function
        def test_function(n):
            time.sleep(0.01)  # Simulate work
            return n * 2
        
        result = test_function(5)
        assert result == 10
        
        # Check profiling data
        assert len(profiler.function_times) > 0
        assert profiler.call_counts['__main__.test_function'] > 0
    
    def test_pipeline_profiling(self):
        """Test pipeline profiling"""
        profiler = MLProfiler()
        
        with profiler.profile_pipeline('test_pipeline'):
            time.sleep(0.01)
        
        assert 'test_pipeline' in profiler.pipeline_times
        assert profiler.pipeline_times['test_pipeline']['total_time'] > 0
    
    def test_get_function_statistics(self):
        """Test getting function statistics"""
        profiler = MLProfiler()
        
        @profiler.profile_function
        def test_function(n):
            time.sleep(0.001)
            return n
        
        # Run multiple times
        for i in range(5):
            test_function(i)
        
        stats = profiler.get_function_statistics()
        assert len(stats) > 0
        
        func_name = '__main__.test_function'
        if func_name in stats:
            func_stats = stats[func_name]
            assert 'call_count' in func_stats
            assert 'total_time' in func_stats
            assert 'mean_time' in func_stats
            assert func_stats['call_count'] == 5
    
    def test_identify_bottlenecks(self):
        """Test bottleneck identification"""
        profiler = MLProfiler()
        
        @profiler.profile_function
        def slow_function():
            time.sleep(0.1)
            return 1
        
        @profiler.profile_function
        def fast_function():
            time.sleep(0.001)
            return 1
        
        # Run functions
        for _ in range(5):
            slow_function()
            fast_function()
        
        bottlenecks = profiler.identify_bottlenecks()
        assert len(bottlenecks) > 0
    
    def test_generate_report(self):
        """Test report generation"""
        profiler = MLProfiler()
        
        @profiler.profile_function
        def test_function():
            time.sleep(0.01)
            return 1
        
        test_function()
        
        report = profiler.generate_report()
        assert 'PROFILING REPORT' in report
        assert 'STATISTICS' in report
    
    def test_export_data(self):
        """Test data export"""
        import tempfile
        import os
        
        profiler = MLProfiler()
        
        @profiler.profile_function
        def test_function():
            return 1
        
        test_function()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            profiler.export_data(temp_path)
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_reset(self):
        """Test profiler reset"""
        profiler = MLProfiler()
        
        @profiler.profile_function
        def test_function():
            return 1
        
        test_function()
        
        assert len(profiler.function_times) > 0
        
        profiler.reset()
        
        assert len(profiler.function_times) == 0
        assert len(profiler.call_counts) == 0


class TestPipelineProfiler:
    """Tests for Pipeline Profiler"""
    
    def test_pipeline_context_manager(self):
        """Test pipeline context manager"""
        profiler = MLProfiler()
        
        with profiler.profile_pipeline('test_pipeline') as pipeline:
            time.sleep(0.01)
            pipeline.add_step('step1', 0.005)
            pipeline.add_step('step2', 0.005)
        
        assert 'test_pipeline' in profiler.pipeline_times
        pipeline_data = profiler.pipeline_times['test_pipeline']
        assert len(pipeline_data['steps']) == 2


class TestProfiledMLToolbox:
    """Tests for Profiled ML Toolbox"""
    
    def test_profiled_toolbox_initialization(self):
        """Test profiled toolbox initialization"""
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            profiled = ProfiledMLToolbox(toolbox)
            assert profiled.toolbox is not None
            assert profiled.profiler is not None
        except ImportError:
            pytest.skip("ML Toolbox not available")
    
    def test_profile_operation(self):
        """Test profiling an operation"""
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            profiled = ProfiledMLToolbox(toolbox)
            
            def test_op():
                return 42
            
            result = profiled.profile_operation('test', test_op)
            assert result == 42
        except ImportError:
            pytest.skip("ML Toolbox not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
