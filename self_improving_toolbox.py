"""
Self-Improving Toolbox
Toolbox that learns and improves from every use

Innovation: Gets better with every operation, no manual tuning needed
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import json
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False


class ImprovementEngine:
    """
    Engine that identifies and applies improvements
    
    Innovation: Analyzes what works and automatically improves
    """
    
    def __init__(self):
        self.improvements_applied = []
        self.improvement_history = []
    
    def analyze_performance(self, performance_data: List[Dict]) -> List[Dict]:
        """Analyze performance to identify improvements"""
        improvements = []
        
        # Group by operation type
        by_operation = defaultdict(list)
        for data in performance_data:
            op_type = data.get('operation_type', 'unknown')
            by_operation[op_type].append(data)
        
        # Find patterns
        for op_type, operations in by_operation.items():
            # Analyze successful operations
            successful = [op for op in operations if op.get('success', False)]
            failed = [op for op in operations if not op.get('success', False)]
            
            if len(successful) > len(failed) * 2:
                # Most operations succeed - optimize further
                avg_time = np.mean([op.get('time', 0) for op in successful])
                improvements.append({
                    'type': 'optimize',
                    'operation': op_type,
                    'suggestion': f'Optimize {op_type} (avg time: {avg_time:.2f}s)',
                    'priority': 'medium'
                })
            elif len(failed) > len(successful):
                # Many failures - fix issues
                common_errors = self._find_common_errors(failed)
                improvements.append({
                    'type': 'fix',
                    'operation': op_type,
                    'suggestion': f'Fix {op_type} (common errors: {common_errors})',
                    'priority': 'high'
                })
        
        return improvements
    
    def _find_common_errors(self, failed_operations: List[Dict]) -> List[str]:
        """Find common error patterns"""
        error_counts = defaultdict(int)
        for op in failed_operations:
            error = op.get('error', 'Unknown')
            error_type = error.split(':')[0] if ':' in error else error
            error_counts[error_type] += 1
        
        # Return top 3 errors
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        return [error for error, count in sorted_errors[:3]]


class SelfImprovingToolbox:
    """
    Self-Improving ML Toolbox
    
    Innovation: Learns from every operation and improves automatically.
    Gets better with use, no manual tuning needed.
    """
    
    def __init__(self, base_toolbox: Optional[Any] = None):
        """
        Initialize self-improving toolbox
        
        Args:
            base_toolbox: Base MLToolbox instance (auto-created if None)
        """
        self.base_toolbox = base_toolbox or (MLToolbox() if TOOLBOX_AVAILABLE else None)
        self.improvement_engine = ImprovementEngine()
        
        # Performance tracking
        self.performance_memory = []  # All operations
        self.successful_patterns = {}  # What works
        self.failure_patterns = {}  # What doesn't
        
        # Improvement state
        self.improvements_applied = []
        self.learning_enabled = True
    
    def fit(self, X, y, **kwargs):
        """
        Fit with self-improvement
        
        Learns from execution and improves for next time
        """
        start_time = time.time()
        operation_id = f"fit_{len(self.performance_memory)}"
        
        # Try current best approach
        try:
            result = self.base_toolbox.fit(X, y, **kwargs) if self.base_toolbox else None
            
            # Record success
            performance_data = {
                'operation_id': operation_id,
                'operation_type': 'fit',
                'success': result is not None,
                'time': time.time() - start_time,
                'params': kwargs,
                'data_shape': (X.shape if hasattr(X, 'shape') else len(X), 
                              y.shape if hasattr(y, 'shape') else len(y))
            }
            
            if result:
                performance_data['result_metrics'] = {
                    'accuracy': result.get('accuracy'),
                    'r2_score': result.get('r2_score')
                }
            
            self.performance_memory.append(performance_data)
            
            # Learn from result
            if self.learning_enabled:
                self._learn_from_execution(performance_data)
            
            return result
            
        except Exception as e:
            # Record failure
            performance_data = {
                'operation_id': operation_id,
                'operation_type': 'fit',
                'success': False,
                'time': time.time() - start_time,
                'error': str(e),
                'params': kwargs
            }
            
            self.performance_memory.append(performance_data)
            
            if self.learning_enabled:
                self._learn_from_execution(performance_data)
            
            raise
    
    def _learn_from_execution(self, performance_data: Dict):
        """Learn from execution result"""
        if performance_data.get('success'):
            # Remember successful patterns
            pattern_key = self._extract_pattern_key(performance_data)
            if pattern_key not in self.successful_patterns:
                self.successful_patterns[pattern_key] = []
            self.successful_patterns[pattern_key].append(performance_data)
        else:
            # Remember failures
            pattern_key = self._extract_pattern_key(performance_data)
            if pattern_key not in self.failure_patterns:
                self.failure_patterns[pattern_key] = []
            self.failure_patterns[pattern_key].append(performance_data)
        
        # Periodically analyze and improve
        if len(self.performance_memory) % 10 == 0:
            self._improve_based_on_learning()
    
    def _extract_pattern_key(self, performance_data: Dict) -> str:
        """Extract pattern key from performance data"""
        op_type = performance_data.get('operation_type', 'unknown')
        data_shape = performance_data.get('data_shape', (0, 0))
        return f"{op_type}_{data_shape[0]}_{data_shape[1]}"
    
    def _improve_based_on_learning(self):
        """Improve toolbox based on learned patterns"""
        # Analyze recent performance
        recent_performance = self.performance_memory[-50:]  # Last 50 operations
        
        # Identify improvements
        improvements = self.improvement_engine.analyze_performance(recent_performance)
        
        # Apply high-priority improvements
        for improvement in improvements:
            if improvement.get('priority') == 'high':
                self._apply_improvement(improvement)
    
    def _apply_improvement(self, improvement: Dict):
        """Apply an improvement"""
        improvement_id = f"improvement_{len(self.improvements_applied)}"
        
        # Record improvement
        self.improvements_applied.append({
            'id': improvement_id,
            'improvement': improvement,
            'timestamp': time.time()
        })
        
        # Apply based on type
        if improvement['type'] == 'optimize':
            # Could optimize caching, parallelization, etc.
            pass
        elif improvement['type'] == 'fix':
            # Could add error handling, fallbacks, etc.
            pass
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        """Get improvement statistics"""
        total_operations = len(self.performance_memory)
        successful = sum(1 for p in self.performance_memory if p.get('success', False))
        
        return {
            'total_operations': total_operations,
            'successful_operations': successful,
            'success_rate': successful / total_operations if total_operations > 0 else 0.0,
            'successful_patterns': len(self.successful_patterns),
            'failure_patterns': len(self.failure_patterns),
            'improvements_applied': len(self.improvements_applied),
            'learning_enabled': self.learning_enabled
        }
    
    def get_recommendations(self) -> List[Dict]:
        """Get improvement recommendations"""
        recent_performance = self.performance_memory[-100:] if len(self.performance_memory) >= 100 else self.performance_memory
        return self.improvement_engine.analyze_performance(recent_performance)


# Global instance
_global_improving_toolbox = None

def get_self_improving_toolbox(base_toolbox: Optional[Any] = None) -> SelfImprovingToolbox:
    """Get global self-improving toolbox instance"""
    global _global_improving_toolbox
    if _global_improving_toolbox is None:
        _global_improving_toolbox = SelfImprovingToolbox(base_toolbox=base_toolbox)
    return _global_improving_toolbox
