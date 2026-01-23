"""
Pattern Graph - Knowledge Graph for Code Patterns
Innovative approach: Learn from pattern relationships, not billions of examples
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class PatternNode:
    """Node in pattern graph"""
    
    def __init__(self, pattern_id: str, pattern_data: Dict):
        self.id = pattern_id
        self.data = pattern_data
        self.relationships = []  # List of (pattern_id, relationship_type)
        self.success_count = 0
        self.failure_count = 0
        self.last_used = None
    
    def add_relationship(self, other_pattern_id: str, relationship_type: str):
        """Add relationship to another pattern"""
        self.relationships.append({
            'pattern': other_pattern_id,
            'type': relationship_type
        })
    
    def record_success(self):
        """Record successful use"""
        self.success_count += 1
        self.last_used = time.time()
    
    def record_failure(self):
        """Record failed use"""
        self.failure_count += 1
    
    def get_success_rate(self) -> float:
        """Get success rate"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class PatternGraph:
    """
    Graph-based knowledge representation for code patterns
    
    Innovation: Uses graph structure to understand pattern relationships,
    enabling composition without needing billions of training examples.
    """
    
    def __init__(self):
        """Initialize pattern graph"""
        self.nodes: Dict[str, PatternNode] = {}
        self.compositions: Dict[str, List[Dict]] = {}  # task -> successful compositions
        self.failures: Dict[str, List[Dict]] = {}  # task -> failed compositions
        self.incompatibilities: Set[tuple] = set()  # (pattern1, pattern2) that don't work
    
    def add_pattern(self, pattern_id: str, pattern_data: Dict):
        """Add a pattern to the graph"""
        if pattern_id not in self.nodes:
            self.nodes[pattern_id] = PatternNode(pattern_id, pattern_data)
        else:
            # Update existing
            self.nodes[pattern_id].data.update(pattern_data)
    
    def link_patterns(self, pattern1_id: str, pattern2_id: str, 
                     relationship_type: str = 'works_with'):
        """
        Link two patterns
        
        Relationship types:
        - 'works_with': Patterns can be used together
        - 'requires': Pattern1 requires Pattern2
        - 'follows': Pattern1 typically follows Pattern2
        - 'alternative': Patterns are alternatives
        """
        if pattern1_id not in self.nodes:
            self.add_pattern(pattern1_id, {})
        if pattern2_id not in self.nodes:
            self.add_pattern(pattern2_id, {})
        
        self.nodes[pattern1_id].add_relationship(pattern2_id, relationship_type)
        
        # Bidirectional for 'works_with'
        if relationship_type == 'works_with':
            self.nodes[pattern2_id].add_relationship(pattern1_id, relationship_type)
    
    def find_pattern_sequence(self, task: str) -> List[str]:
        """
        Find sequence of patterns to solve task
        
        Uses graph traversal to find best pattern composition
        """
        # 1. Check if we've solved this before
        if task in self.compositions:
            best = self.compositions[task][0]
            return best['pattern_sequence']
        
        # 2. Decompose task into sub-tasks
        sub_tasks = self._decompose_task(task)
        
        # 3. Match patterns to sub-tasks
        pattern_sequence = []
        for sub_task in sub_tasks:
            pattern = self._match_pattern_to_task(sub_task)
            if pattern:
                pattern_sequence.append(pattern)
        
        # 4. Validate sequence (check incompatibilities)
        pattern_sequence = self._validate_sequence(pattern_sequence)
        
        return pattern_sequence
    
    def _decompose_task(self, task: str) -> List[str]:
        """Decompose task into sub-tasks"""
        task_lower = task.lower()
        sub_tasks = []
        
        # Common decompositions
        if 'classify' in task_lower or 'classification' in task_lower:
            sub_tasks = ['data_loading', 'preprocessing', 'classification', 'evaluation']
        elif 'regress' in task_lower or 'predict' in task_lower:
            sub_tasks = ['data_loading', 'preprocessing', 'regression', 'evaluation']
        elif 'preprocess' in task_lower:
            sub_tasks = ['data_loading', 'preprocessing']
        else:
            # Generic
            sub_tasks = ['data_loading', 'preprocessing', 'training', 'evaluation']
        
        return sub_tasks
    
    def _match_pattern_to_task(self, sub_task: str) -> Optional[str]:
        """Match sub-task to pattern"""
        # Simple keyword matching (can be enhanced)
        task_to_pattern = {
            'data_loading': 'data_loading',
            'preprocessing': 'preprocessing',
            'classification': 'classification',
            'regression': 'regression',
            'evaluation': 'evaluation',
            'training': 'classification'  # Default
        }
        
        return task_to_pattern.get(sub_task)
    
    def _validate_sequence(self, sequence: List[str]) -> List[str]:
        """Validate pattern sequence (remove incompatibilities)"""
        validated = []
        for i, pattern in enumerate(sequence):
            # Check if incompatible with previous
            if validated:
                prev_pattern = validated[-1]
                if (prev_pattern, pattern) in self.incompatibilities:
                    continue  # Skip incompatible pattern
            
            validated.append(pattern)
        
        return validated
    
    def record_successful_composition(self, task: str, pattern_sequence: List[str], 
                                     code: str, execution_result: Dict):
        """Record successful pattern composition"""
        if task not in self.compositions:
            self.compositions[task] = []
        
        composition = {
            'pattern_sequence': pattern_sequence,
            'code': code,
            'execution_result': execution_result,
            'timestamp': time.time(),
            'success_rate': 1.0
        }
        
        self.compositions[task].append(composition)
        
        # Sort by success rate (best first)
        self.compositions[task].sort(key=lambda x: x['success_rate'], reverse=True)
        
        # Update pattern success counts
        for pattern_id in pattern_sequence:
            if pattern_id in self.nodes:
                self.nodes[pattern_id].record_success()
    
    def record_failed_composition(self, task: str, pattern_sequence: List[str], 
                                 code: str, error: str):
        """Record failed pattern composition"""
        if task not in self.failures:
            self.failures[task] = []
        
        self.failures[task].append({
            'pattern_sequence': pattern_sequence,
            'code': code,
            'error': error,
            'timestamp': time.time()
        })
        
        # Update pattern failure counts
        for pattern_id in pattern_sequence:
            if pattern_id in self.nodes:
                self.nodes[pattern_id].record_failure()
        
        # Mark incompatibility if consecutive patterns fail
        if len(pattern_sequence) >= 2:
            for i in range(len(pattern_sequence) - 1):
                self.incompatibilities.add(
                    (pattern_sequence[i], pattern_sequence[i+1])
                )
    
    def get_best_patterns(self, task_type: str, limit: int = 5) -> List[str]:
        """Get best patterns for task type (by success rate)"""
        relevant_patterns = [
            (pattern_id, node) 
            for pattern_id, node in self.nodes.items()
            if task_type in pattern_id.lower()
        ]
        
        # Sort by success rate
        relevant_patterns.sort(
            key=lambda x: x[1].get_success_rate(),
            reverse=True
        )
        
        return [pattern_id for pattern_id, _ in relevant_patterns[:limit]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        total_patterns = len(self.nodes)
        total_relationships = sum(len(node.relationships) for node in self.nodes.values())
        total_compositions = sum(len(comps) for comps in self.compositions.values())
        total_failures = sum(len(fails) for fails in self.failures.values())
        
        return {
            'total_patterns': total_patterns,
            'total_relationships': total_relationships,
            'total_compositions': total_compositions,
            'total_failures': total_failures,
            'success_rate': total_compositions / (total_compositions + total_failures) 
                           if (total_compositions + total_failures) > 0 else 0.0
        }


# Global pattern graph instance
_global_graph: Optional[PatternGraph] = None

def get_pattern_graph() -> PatternGraph:
    """Get global pattern graph instance"""
    global _global_graph
    if _global_graph is None:
        _global_graph = PatternGraph()
    return _global_graph
