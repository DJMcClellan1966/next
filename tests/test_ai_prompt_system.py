"""
Tests for AI Prompt System
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ai_prompt_system import (
        AIPromptSystem, GuidedWorkflow, ReportGenerator, create_ai_assistant
    )
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    pytestmark = pytest.mark.skip("Features not available")


class TestAIPromptSystem:
    """Tests for AI Prompt System"""
    
    def test_start_conversation(self):
        """Test starting conversation"""
        system = AIPromptSystem()
        welcome = system.start_conversation()
        
        assert 'Welcome' in welcome
        assert 'What would you like to do' in welcome
    
    def test_understand_intent(self):
        """Test intent understanding"""
        system = AIPromptSystem()
        
        # Test task selection
        intent = system._understand_intent("1")
        assert intent['type'] == 'task_selection'
        
        intent = system._understand_intent("I want to predict sales")
        assert intent['type'] == 'task_selection'
        
        # Test data upload
        intent = system._understand_intent("I have a CSV file")
        assert intent['type'] == 'data_upload'
    
    def test_load_data(self):
        """Test data loading"""
        system = AIPromptSystem()
        
        # Create test data
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Save to temp file
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result = system.load_data(temp_path)
            assert result['success'] == True
            assert 'dataframe' in system.user_data
        finally:
            os.unlink(temp_path)
    
    def test_data_analysis(self):
        """Test data analysis"""
        system = AIPromptSystem()
        
        # Create test data
        df = pd.DataFrame({
            'sales': np.random.rand(100) * 1000,
            'region': np.random.choice(['A', 'B', 'C'], 100),
            'date': pd.date_range('2024-01-01', periods=100)
        })
        
        system.user_data['dataframe'] = df
        system.current_task = 'analysis'
        
        results = system._run_data_analysis(df)
        
        assert results['task'] == 'data_analysis'
        assert 'summary' in results
        assert 'insights' in results
    
    def test_generate_report(self):
        """Test report generation"""
        system = AIPromptSystem()
        
        results = {
            'task': 'prediction',
            'model_type': 'classification',
            'target_column': 'target',
            'accuracy': 0.95
        }
        
        report = system._generate_human_readable_report(results)
        
        assert 'PREDICTION MODEL RESULTS' in report
        assert '95' in report or '0.95' in report


class TestGuidedWorkflow:
    """Tests for guided workflow"""
    
    def test_create_workflow(self):
        """Test workflow creation"""
        system = AIPromptSystem()
        workflow = GuidedWorkflow(system)
        
        steps = workflow.create_workflow('prediction')
        
        assert len(steps) > 0
        assert steps[0]['step'] == 1
        assert 'question' in steps[0]


class TestReportGenerator:
    """Tests for report generator"""
    
    def test_executive_summary(self):
        """Test executive summary generation"""
        results = {
            'task': 'prediction',
            'accuracy': 0.92
        }
        
        summary = ReportGenerator.generate_executive_summary(results)
        
        assert 'EXECUTIVE SUMMARY' in summary
        assert '92' in summary or '0.92' in summary
    
    def test_detailed_report(self):
        """Test detailed report generation"""
        results = {
            'task': 'clustering',
            'n_clusters': 3,
            'cluster_labels': [0, 1, 2, 0, 1, 2]
        }
        
        report = ReportGenerator.generate_detailed_report(results)
        
        assert 'CLUSTER ANALYSIS' in report
        assert '3' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
