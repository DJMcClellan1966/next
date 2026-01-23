"""
ML Code Agent
Main agent class that generates and executes ML code
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .knowledge_base import ToolboxKnowledgeBase, get_knowledge_base
from .code_generator import CodeGenerator
from .code_sandbox import CodeSandbox


class MLCodeAgent:
    """
    AI Agent for generating ML code using ML Toolbox
    
    Features:
    - Generate code from natural language
    - Execute and test code
    - Fix errors automatically
    - Learn from patterns
    """
    
    def __init__(self, use_llm: bool = True, max_iterations: int = 3):
        """
        Initialize ML Code Agent
        
        Args:
            use_llm: Whether to use LLM for code generation
            max_iterations: Maximum iterations for error fixing
        """
        self.kb = get_knowledge_base()
        self.generator = CodeGenerator(self.kb, use_llm=use_llm)
        self.sandbox = CodeSandbox()
        self.max_iterations = max_iterations
        self.history = []
    
    def build(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Build ML solution for a task
        
        Args:
            task: Task description in natural language
            context: Additional context (data info, requirements, etc.)
        
        Returns:
            Dictionary with 'code', 'result', 'success', 'iterations'
        
        Example:
            >>> agent = MLCodeAgent()
            >>> result = agent.build("Classify iris flowers")
            >>> print(result['code'])
        """
        iterations = 0
        code = None
        execution_result = None
        
        while iterations < self.max_iterations:
            iterations += 1
            
            # Generate code
            if code is None:
                generation_result = self.generator.generate(task, context)
                code = generation_result['code']
                
                if not generation_result['success']:
                    # Syntax error in generation
                    return {
                        'code': code,
                        'result': None,
                        'success': False,
                        'error': generation_result.get('error'),
                        'iterations': iterations,
                        'message': 'Code generation failed - syntax error'
                    }
            
            # Execute code
            execution_result = self.sandbox.execute(code)
            
            if execution_result['success']:
                # Success!
                self.history.append({
                    'task': task,
                    'code': code,
                    'success': True,
                    'iterations': iterations
                })
                
                return {
                    'code': code,
                    'result': execution_result,
                    'success': True,
                    'iterations': iterations,
                    'output': execution_result.get('output', ''),
                    'message': 'Code generated and executed successfully'
                }
            
            # Error occurred - try to fix
            if iterations < self.max_iterations:
                error = execution_result.get('error', 'Unknown error')
                code = self.generator.improve_code(code, error)
            else:
                # Max iterations reached
                self.history.append({
                    'task': task,
                    'code': code,
                    'success': False,
                    'error': execution_result.get('error'),
                    'iterations': iterations
                })
                
                return {
                    'code': code,
                    'result': execution_result,
                    'success': False,
                    'error': execution_result.get('error'),
                    'iterations': iterations,
                    'traceback': execution_result.get('traceback'),
                    'message': f'Failed after {iterations} iterations'
                }
        
        return {
            'code': code,
            'result': execution_result,
            'success': False,
            'error': 'Max iterations reached',
            'iterations': iterations
        }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get agent execution history"""
        return self.history
    
    def clear_history(self):
        """Clear execution history"""
        self.history = []
