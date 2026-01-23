"""
Pattern Composer - Compose code from patterns
Innovative: Builds code like LEGO blocks, not from training data
"""
import sys
from pathlib import Path
import re
import ast
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .pattern_graph import PatternGraph, get_pattern_graph
from .knowledge_base import ToolboxKnowledgeBase, get_knowledge_base


class PatternComposer:
    """
    Compose code from patterns
    
    Innovation: Instead of generating from scratch or training on billions
    of examples, composes code from reusable pattern blocks.
    """
    
    def __init__(self, pattern_graph: Optional[PatternGraph] = None,
                 knowledge_base: Optional[ToolboxKnowledgeBase] = None):
        """
        Initialize pattern composer
        
        Args:
            pattern_graph: Pattern graph (auto-created if None)
            knowledge_base: Knowledge base (auto-created if None)
        """
        self.graph = pattern_graph or get_pattern_graph()
        self.kb = knowledge_base or get_knowledge_base()
        self.variable_map = {}  # Track variables across patterns
    
    def compose(self, pattern_sequence: List[str], context: Optional[Dict] = None) -> str:
        """
        Compose code from pattern sequence
        
        Args:
            pattern_sequence: List of pattern IDs
            context: Additional context (data info, etc.)
        
        Returns:
            Complete Python code
        """
        code_parts = []
        imports = set()
        self.variable_map = {}
        
        # Track variables across patterns
        last_output = None
        
        for i, pattern_id in enumerate(pattern_sequence):
            # Get pattern from knowledge base
            pattern = self.kb.get_pattern(pattern_id)
            if not pattern:
                # Try to get from graph
                if pattern_id in self.graph.nodes:
                    pattern = {'code': self.graph.nodes[pattern_id].data.get('code', '')}
                else:
                    continue  # Skip unknown pattern
            
            # Get pattern code
            pattern_code = pattern.get('code', '')
            
            # Extract and collect imports
            pattern_imports = self._extract_imports(pattern_code)
            imports.update(pattern_imports)
            
            # Resolve variables (connect patterns)
            pattern_code = self._resolve_variables(
                pattern_code,
                last_output,
                context,
                i == 0  # First pattern
            )
            
            # Extract outputs from this pattern
            last_output = self._extract_outputs(pattern_code)
            
            code_parts.append(pattern_code)
        
        # Combine into complete code
        complete_code = self._combine_code(imports, code_parts, context)
        
        return complete_code
    
    def _extract_imports(self, code: str) -> set:
        """Extract import statements"""
        imports = set()
        
        # Find import statements
        import_pattern = r'^(import\s+\w+|from\s+\w+\s+import\s+[\w\s,]+)'
        for line in code.split('\n'):
            match = re.match(import_pattern, line.strip())
            if match:
                imports.add(line.strip())
        
        return imports
    
    def _resolve_variables(self, code: str, previous_output: Optional[Dict],
                         context: Optional[Dict], is_first: bool) -> str:
        """
        Resolve variables between patterns
        
        Connects patterns by mapping output variables to input variables
        """
        # If first pattern, use context or defaults
        if is_first:
            if context and 'data_name' in context:
                code = code.replace('X', context['data_name'])
            if context and 'target_name' in context:
                code = code.replace('y', context['target_name'])
        
        # Connect to previous pattern's output
        if previous_output:
            # Map common variable names
            if 'X' in previous_output and 'X' not in code:
                # Previous pattern output X, this pattern needs it
                code = code.replace('X_train', 'X')
                code = code.replace('X_test', 'X')
            
            if 'y' in previous_output and 'y' not in code:
                code = code.replace('y_train', 'y')
                code = code.replace('y_test', 'y')
            
            if 'model' in previous_output:
                code = code.replace('result[\'model\']', 'model')
        
        return code
    
    def _extract_outputs(self, code: str) -> Dict[str, str]:
        """Extract output variables from code"""
        outputs = {}
        
        # Look for common output patterns
        if 'result = toolbox.fit' in code or 'result =' in code:
            outputs['result'] = 'result'
            outputs['model'] = 'result[\'model\']'
        
        if 'X =' in code or 'X_train' in code:
            outputs['X'] = 'X'
        
        if 'y =' in code or 'y_train' in code:
            outputs['y'] = 'y'
        
        return outputs
    
    def _combine_code(self, imports: set, code_parts: List[str], 
                     context: Optional[Dict]) -> str:
        """Combine code parts into complete code"""
        # Sort imports
        sorted_imports = sorted(imports)
        
        # Build complete code
        complete_code = '\n'.join(sorted_imports)
        complete_code += '\n\n'
        complete_code += '# Generated code from pattern composition\n'
        
        if context:
            complete_code += f'# Context: {context}\n'
        
        complete_code += '\n'
        complete_code += '\n\n'.join(code_parts)
        
        return complete_code
    
    def refine_composition(self, code: str, error: str) -> str:
        """
        Refine composition based on error
        
        Innovation: Learns from errors to improve composition
        """
        # Simple refinements based on error type
        if 'ImportError' in error or 'ModuleNotFoundError' in error:
            # Add missing imports
            if 'numpy' in error.lower() and 'import numpy' not in code:
                code = 'import numpy as np\n' + code
            if 'ml_toolbox' in error.lower() and 'from ml_toolbox' not in code:
                code = 'from ml_toolbox import MLToolbox\n' + code
        
        elif 'NameError' in error:
            # Try to fix undefined variables
            var_match = re.search(r"name '(\w+)' is not defined", error)
            if var_match:
                var_name = var_match.group(1)
                # Add initialization if common variable
                if var_name == 'toolbox':
                    code = code.replace('toolbox.fit', 'MLToolbox().fit')
                    if 'toolbox =' not in code:
                        code = 'toolbox = MLToolbox()\n' + code
        
        elif 'AttributeError' in error:
            # Fix attribute errors
            attr_match = re.search(r"'(\w+)' object has no attribute '(\w+)'", error)
            if attr_match:
                obj_type, attr = attr_match.groups()
                # Try to fix common issues
                if obj_type == 'dict' and attr == 'model':
                    code = code.replace('result.model', 'result[\'model\']')
        
        return code
