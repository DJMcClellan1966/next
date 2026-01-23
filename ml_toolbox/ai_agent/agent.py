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
from .pattern_graph import PatternGraph, get_pattern_graph
from .pattern_composer import PatternComposer


class MLCodeAgent:
    """
    AI Agent for generating ML code using ML Toolbox
    
    Features:
    - Generate code from natural language
    - Execute and test code
    - Fix errors automatically
    - Learn from patterns
    """
    
    def __init__(self, use_llm: bool = True, max_iterations: int = 3, 
                 use_pattern_composition: bool = True):
        """
        Initialize ML Code Agent
        
        Args:
            use_llm: Whether to use LLM for code generation
            max_iterations: Maximum iterations for error fixing
            use_pattern_composition: Use innovative pattern composition (default: True)
        """
        self.kb = get_knowledge_base()
        self.graph = get_pattern_graph()
        self.composer = PatternComposer(self.graph, self.kb)
        self.generator = CodeGenerator(self.kb, use_llm=use_llm)
        self.sandbox = CodeSandbox()
        self.max_iterations = max_iterations
        self.use_pattern_composition = use_pattern_composition
        self.history = []
        
        # Initialize pattern graph with knowledge base patterns
        self._initialize_pattern_graph()
    
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
            
            # Generate code using innovative pattern composition
            if code is None:
                generation_result = None
                if self.use_pattern_composition:
                    # Use pattern composition (innovative approach)
                    pattern_sequence = self.graph.find_pattern_sequence(task)
                    if pattern_sequence:
                        code = self.composer.compose(pattern_sequence, context)
                        # Validate syntax
                        from .code_generator import CodeGenerator
                        temp_gen = CodeGenerator(self.kb, use_llm=False)
                        validation = temp_gen._validate_syntax(code)
                        if not validation['valid']:
                            generation_result = {'success': False, 'error': validation.get('error')}
                        else:
                            generation_result = {'success': True}
                    else:
                        # Fallback to generator
                        generation_result = self.generator.generate(task, context)
                        code = generation_result['code']
                else:
                    # Use traditional generation
                    generation_result = self.generator.generate(task, context)
                    code = generation_result['code']
                
                if generation_result and not generation_result.get('success', True):
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
                # Success! Record in pattern graph
                if self.use_pattern_composition:
                    pattern_sequence = self.graph.find_pattern_sequence(task)
                    self.graph.record_successful_composition(
                        task, pattern_sequence, code, execution_result
                    )
                
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
                
                # Record failure in pattern graph
                if self.use_pattern_composition:
                    pattern_sequence = self.graph.find_pattern_sequence(task)
                    self.graph.record_failed_composition(task, pattern_sequence, code, error)
                
                # Try to fix using composer or generator
                if self.use_pattern_composition:
                    code = self.composer.refine_composition(code, error)
                else:
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
    
    def _initialize_pattern_graph(self):
        """Initialize pattern graph with knowledge base patterns"""
        patterns = self.kb.get_all_patterns()
        for pattern in patterns:
            self.graph.add_pattern(pattern['name'], pattern)
            
            # Link related patterns
            if 'classification' in pattern['name']:
                self.graph.link_patterns(pattern['name'], 'preprocessing', 'requires')
            if 'regression' in pattern['name']:
                self.graph.link_patterns(pattern['name'], 'preprocessing', 'requires')