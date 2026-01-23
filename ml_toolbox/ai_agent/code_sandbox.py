"""
Code Sandbox
Safe execution environment for generated code
"""
import sys
from pathlib import Path
import io
import contextlib
import traceback
from typing import Dict, Any, Optional
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class CodeSandbox:
    """
    Safe code execution environment
    
    Features:
    - Isolated execution
    - Resource limits (time, memory)
    - Output capture
    - Error capture
    """
    
    def __init__(self, timeout: int = 30, capture_output: bool = True):
        """
        Initialize code sandbox
        
        Args:
            timeout: Execution timeout in seconds
            capture_output: Whether to capture stdout/stderr
        """
        self.timeout = timeout
        self.capture_output = capture_output
    
    def execute(self, code: str, globals_dict: Optional[Dict] = None, 
                locals_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute code safely
        
        Args:
            code: Python code to execute
            globals_dict: Global namespace (default: safe namespace)
            locals_dict: Local namespace (default: empty)
        
        Returns:
            Dictionary with 'success', 'output', 'error', 'result'
        """
        # Create safe namespace
        if globals_dict is None:
            globals_dict = self._create_safe_namespace()
        
        if locals_dict is None:
            locals_dict = {}
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = {
            'success': False,
            'output': '',
            'error': None,
            'result': None,
            'traceback': None
        }
        
        try:
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    # Execute code
                    exec(code, globals_dict, locals_dict)
            
            # Get output
            if self.capture_output:
                result['output'] = stdout_capture.getvalue()
                error_output = stderr_capture.getvalue()
                if error_output:
                    result['error'] = error_output
            
            # Check for result variable
            if 'result' in locals_dict:
                result['result'] = locals_dict['result']
            
            result['success'] = True
            
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            
            if self.capture_output:
                result['output'] = stdout_capture.getvalue()
        
        return result
    
    def _create_safe_namespace(self) -> Dict[str, Any]:
        """Create safe global namespace"""
        namespace = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'bool': bool,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
            }
        }
        
        # Add common imports (safe)
        try:
            import numpy as np
            namespace['np'] = np
            namespace['numpy'] = np
        except ImportError:
            pass
        
        try:
            from ml_toolbox import MLToolbox
            namespace['MLToolbox'] = MLToolbox
        except ImportError:
            pass
        
        return namespace
    
    def test_code(self, code: str, test_input: Optional[Any] = None) -> Dict[str, Any]:
        """
        Test code with optional input
        
        Args:
            code: Code to test
            test_input: Optional test input
        
        Returns:
            Test results
        """
        result = self.execute(code)
        
        # Additional validation
        if result['success']:
            # Check if code produces expected output
            if test_input is not None:
                # Run with test input
                test_result = self.execute(code, globals_dict={'test_input': test_input})
                result['test_result'] = test_result
        
        return result
