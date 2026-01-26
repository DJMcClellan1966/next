"""
Pipeline Debugger

Debugging and visualization tools for pipelines.
"""
from typing import Any, Dict, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PipelineDebugger:
    """
    Debugging tools for pipelines
    
    Provides:
    - Stage-by-stage debugging
    - Data inspection
    - Execution trace
    - Performance profiling
    - Visualization
    """
    
    def __init__(self, enable_debugging: bool = True):
        """
        Initialize pipeline debugger
        
        Parameters
        ----------
        enable_debugging : bool, default=True
            Whether to enable debugging
        """
        self.enable_debugging = enable_debugging
        self.execution_trace: List[Dict[str, Any]] = []
        self.data_snapshots: Dict[str, Any] = {}
        self.breakpoints: List[str] = []
    
    def add_breakpoint(self, stage_name: str):
        """Add breakpoint at a stage"""
        if stage_name not in self.breakpoints:
            self.breakpoints.append(stage_name)
            logger.info(f"[PipelineDebugger] Added breakpoint at: {stage_name}")
    
    def remove_breakpoint(self, stage_name: str):
        """Remove breakpoint"""
        if stage_name in self.breakpoints:
            self.breakpoints.remove(stage_name)
            logger.info(f"[PipelineDebugger] Removed breakpoint at: {stage_name}")
    
    def clear_breakpoints(self):
        """Clear all breakpoints"""
        self.breakpoints = []
        logger.info("[PipelineDebugger] Cleared all breakpoints")
    
    def trace_stage(self, stage_name: str, input_data: Any, output_data: Any,
                   duration: float, success: bool, error: Optional[Exception] = None):
        """Trace stage execution"""
        if not self.enable_debugging:
            return
        
        trace_entry = {
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'success': success,
            'error': str(error) if error else None,
            'input_shape': self._get_shape(input_data),
            'output_shape': self._get_shape(output_data),
            'input_type': type(input_data).__name__,
            'output_type': type(output_data).__name__
        }
        
        self.execution_trace.append(trace_entry)
        
        # Check for breakpoint
        if stage_name in self.breakpoints:
            logger.info(f"[PipelineDebugger] Breakpoint hit at: {stage_name}")
            self._inspect_data(stage_name, input_data, output_data)
    
    def _get_shape(self, data: Any) -> Optional[tuple]:
        """Get shape of data"""
        if hasattr(data, 'shape'):
            return data.shape
        elif hasattr(data, '__len__'):
            return (len(data),)
        return None
    
    def _inspect_data(self, stage_name: str, input_data: Any, output_data: Any):
        """Inspect data at breakpoint"""
        logger.info(f"[PipelineDebugger] Inspecting data at {stage_name}:")
        logger.info(f"  Input: {self._describe_data(input_data)}")
        logger.info(f"  Output: {self._describe_data(output_data)}")
    
    def _describe_data(self, data: Any) -> str:
        """Describe data"""
        if hasattr(data, 'shape'):
            return f"shape={data.shape}, dtype={data.dtype}"
        elif hasattr(data, '__len__'):
            return f"length={len(data)}, type={type(data).__name__}"
        else:
            return f"type={type(data).__name__}"
    
    def snapshot_data(self, stage_name: str, data: Any, label: str = "data"):
        """Take snapshot of data at a stage"""
        snapshot_key = f"{stage_name}_{label}"
        self.data_snapshots[snapshot_key] = {
            'data': data,
            'stage': stage_name,
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'description': self._describe_data(data)
        }
        logger.debug(f"[PipelineDebugger] Snapshot taken: {snapshot_key}")
    
    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """Get execution trace"""
        return self.execution_trace.copy()
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get trace summary"""
        if not self.execution_trace:
            return {}
        
        total_duration = sum(entry['duration'] for entry in self.execution_trace)
        failed_stages = [e for e in self.execution_trace if not e['success']]
        
        return {
            'total_stages': len(self.execution_trace),
            'total_duration': total_duration,
            'failed_stages': len(failed_stages),
            'failed_stage_names': [e['stage'] for e in failed_stages],
            'stage_durations': {e['stage']: e['duration'] for e in self.execution_trace}
        }
    
    def visualize_trace(self, output_file: Optional[str] = None) -> str:
        """Create text visualization of execution trace"""
        if not self.execution_trace:
            return "No execution trace available"
        
        lines = ["Pipeline Execution Trace", "=" * 50]
        
        for i, entry in enumerate(self.execution_trace, 1):
            status = "[OK]" if entry['success'] else "[FAIL]"
            lines.append(f"\n[{i}] {status} {entry['stage']}")
            lines.append(f"    Duration: {entry['duration']:.4f}s")
            lines.append(f"    Input: {entry['input_shape'] or 'N/A'}")
            lines.append(f"    Output: {entry['output_shape'] or 'N/A'}")
            if entry['error']:
                lines.append(f"    Error: {entry['error']}")
        
        summary = self.get_trace_summary()
        lines.append("\n" + "=" * 50)
        lines.append("Summary:")
        lines.append(f"  Total Stages: {summary.get('total_stages', 0)}")
        lines.append(f"  Total Duration: {summary.get('total_duration', 0):.4f}s")
        lines.append(f"  Failed Stages: {summary.get('failed_stages', 0)}")
        
        visualization = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(visualization)
            logger.info(f"[PipelineDebugger] Trace visualization saved to: {output_file}")
        
        return visualization
    
    def clear_trace(self):
        """Clear execution trace"""
        self.execution_trace = []
        self.data_snapshots = {}
        logger.info("[PipelineDebugger] Execution trace cleared")
    
    def get_data_snapshot(self, stage_name: str, label: str = "data") -> Optional[Any]:
        """Get data snapshot"""
        snapshot_key = f"{stage_name}_{label}"
        return self.data_snapshots.get(snapshot_key, {}).get('data')
