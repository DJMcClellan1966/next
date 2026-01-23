"""
Code Generator
Generates Python code from natural language using LLM and knowledge base
"""
import sys
from pathlib import Path
import re
import ast
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .knowledge_base import ToolboxKnowledgeBase, get_knowledge_base


class CodeGenerator:
    """
    Generate Python code from natural language
    
    Uses:
    - Toolbox Knowledge Base for context
    - LLM for code generation
    - Syntax validation
    """
    
    def __init__(self, knowledge_base: Optional[ToolboxKnowledgeBase] = None, use_llm: bool = True):
        """
        Initialize code generator
        
        Args:
            knowledge_base: Toolbox knowledge base (auto-created if None)
            use_llm: Whether to use LLM (fallback to templates if False)
        """
        self.kb = knowledge_base or get_knowledge_base()
        self.use_llm = use_llm
        self.llm = None
        
        if use_llm:
            try:
                from llm.quantum_llm_standalone import StandaloneQuantumLLM
                self.llm = StandaloneQuantumLLM()
            except ImportError:
                print("Warning: LLM not available, using template-based generation")
                self.use_llm = False
    
    def generate(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate Python code for a task
        
        Args:
            task: Task description in natural language
            context: Additional context (data shape, requirements, etc.)
        
        Returns:
            Dictionary with 'code', 'success', 'error', 'pattern_used'
        """
        # Find relevant solutions from knowledge base
        solutions = self.kb.find_solution(task)
        
        # Build prompt with context
        prompt = self._build_prompt(task, solutions, context)
        
        # Generate code
        if self.use_llm and self.llm:
            code = self._generate_with_llm(prompt)
        else:
            code = self._generate_from_template(task, solutions)
        
        # Validate syntax
        validation = self._validate_syntax(code)
        
        return {
            'code': code,
            'success': validation['valid'],
            'error': validation.get('error'),
            'pattern_used': solutions[0].get('name') or solutions[0].get('solution') if solutions else None,
            'solutions_found': len(solutions)
        }
    
    def _build_prompt(self, task: str, solutions: List[Dict], context: Optional[Dict]) -> str:
        """Build prompt for LLM"""
        prompt = f"""Generate Python code to solve this task using the ML Toolbox:

Task: {task}

Available ML Toolbox APIs:
- toolbox.fit(X, y) - Train a model (auto-detects task type)
- toolbox.predict(model, X) - Make predictions
- toolbox.register_model(model, name) - Register model
- toolbox.data.get_preprocessor(type) - Get data preprocessor
- toolbox.get_ml_math_optimizer() - Get optimized operations

"""
        
        if solutions:
            prompt += "Relevant code patterns:\n"
            for sol in solutions[:2]:  # Top 2 solutions
                if 'code' in sol:
                    prompt += f"\nPattern: {sol.get('name', 'unknown')}\n"
                    prompt += f"{sol['code']}\n"
        
        if context:
            prompt += f"\nContext: {context}\n"
        
        prompt += """
Generate complete, runnable Python code. Include:
1. Import statements
2. Data preparation (or use sample data)
3. Model training using toolbox.fit()
4. Model evaluation
5. Predictions

Code:"""
        
        return prompt
    
    def _generate_with_llm(self, prompt: str) -> str:
        """Generate code using LLM"""
        try:
            generated = self.llm.generate_grounded(prompt, max_length=2000)
            
            # Extract code block if present
            code_match = re.search(r'```python\n(.*?)\n```', generated, re.DOTALL)
            if code_match:
                return code_match.group(1)
            
            # Try without python marker
            code_match = re.search(r'```\n(.*?)\n```', generated, re.DOTALL)
            if code_match:
                return code_match.group(1)
            
            # Return as-is if no code block
            return generated
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return ""
    
    def _generate_from_template(self, task: str, solutions: List[Dict]) -> str:
        """Generate code from templates (fallback)"""
        if not solutions:
            # Default template
            return '''from ml_toolbox import MLToolbox
import numpy as np

# Initialize toolbox
toolbox = MLToolbox()

# Prepare data
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

# Train model
result = toolbox.fit(X, y)

# Get results
model = result['model']
print(f"Model trained successfully")
'''
        
        # Use first matching pattern
        pattern = self.kb.get_pattern(solutions[0].get('solution', solutions[0].get('name', '')))
        if pattern:
            return pattern['code']
        
        # Fallback to default
        return self._generate_from_template(task, [])
    
    def _validate_syntax(self, code: str) -> Dict[str, Any]:
        """
        Validate Python syntax
        
        Args:
            code: Python code string
        
        Returns:
            Dictionary with 'valid' and optional 'error'
        """
        try:
            ast.parse(code)
            return {'valid': True}
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"Syntax error: {e.msg} at line {e.lineno}",
                'line': e.lineno,
                'offset': e.offset
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation error: {str(e)}"
            }
    
    def improve_code(self, code: str, error: str) -> str:
        """
        Improve code based on error message
        
        Args:
            code: Original code
            error: Error message
        
        Returns:
            Improved code
        """
        if self.use_llm and self.llm:
            prompt = f"""Fix this Python code that has an error:

Code:
```python
{code}
```

Error:
{error}

Provide the fixed code:"""
            
            fixed = self._generate_with_llm(prompt)
            if fixed:
                return fixed
        
        # Simple fixes
        if "ImportError" in error or "ModuleNotFoundError" in error:
            # Try to add missing imports
            if "numpy" in error.lower():
                if "import numpy" not in code:
                    code = "import numpy as np\n" + code
            if "ml_toolbox" in error.lower():
                if "from ml_toolbox" not in code:
                    code = "from ml_toolbox import MLToolbox\n" + code
        
        return code
