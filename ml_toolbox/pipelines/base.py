"""
Base Pipeline Classes

Provides base classes for pipeline stages and pipelines.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)

# Optional imports for enhanced features
try:
    from .pipeline_monitoring import PipelineMonitor
    from .pipeline_retry import RetryHandler, RetryConfig
    from .pipeline_debugger import PipelineDebugger
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    PipelineMonitor = None
    RetryHandler = None
    RetryConfig = None
    PipelineDebugger = None


class StageStatus(Enum):
    """Pipeline stage status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStage(ABC):
    """
    Base class for pipeline stages
    
    Each stage represents a single step in a pipeline (e.g., preprocessing, training).
    Stages can be chained together to form a complete pipeline.
    """
    
    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize pipeline stage
        
        Parameters
        ----------
        name : str
            Stage name
        enabled : bool, default=True
            Whether stage is enabled
        """
        self.name = name
        self.enabled = enabled
        self.status = StageStatus.PENDING
        self.metadata: Dict[str, Any] = {}
        self.input_data: Optional[Any] = None
        self.output_data: Optional[Any] = None
        self.error: Optional[Exception] = None
    
    @abstractmethod
    def execute(self, input_data: Any, **kwargs) -> Any:
        """
        Execute the pipeline stage
        
        Parameters
        ----------
        input_data : Any
            Input data for this stage
        **kwargs
            Additional parameters
            
        Returns
        -------
        output_data : Any
            Output data from this stage
        """
        pass
    
    def run(self, input_data: Any, monitor=None, retry_handler=None, debugger=None, **kwargs) -> Any:
        """
        Run the stage with error handling and status tracking
        
        Parameters
        ----------
        input_data : Any
            Input data
        monitor : PipelineMonitor, optional
            Monitor for tracking metrics
        retry_handler : RetryHandler, optional
            Retry handler for error recovery
        debugger : PipelineDebugger, optional
            Debugger for debugging
        **kwargs
            Additional parameters
            
        Returns
        -------
        output_data : Any
            Output data
        """
        if not self.enabled:
            self.status = StageStatus.SKIPPED
            logger.info(f"[{self.name}] Stage skipped (disabled)")
            return input_data
        
        self.status = StageStatus.RUNNING
        self.input_data = input_data
        
        start_time = time.time()
        
        # Debugger snapshot
        if debugger:
            debugger.snapshot_data(self.name, input_data, "input")
        
        try:
            logger.info(f"[{self.name}] Executing stage...")
            
            # Execute with retry if handler provided
            if retry_handler:
                def execute_func():
                    return self.execute(input_data, **kwargs)
                output_data = retry_handler.execute_with_retry(self.name, execute_func)
            else:
                output_data = self.execute(input_data, **kwargs)
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.status = StageStatus.COMPLETED
            self.output_data = output_data
            
            # Monitor tracking
            if monitor and monitor.current_metrics:
                monitor.track_stage(self.name, start_time, end_time, input_data, output_data, success=True)
            
            # Debugger trace
            if debugger:
                debugger.trace_stage(self.name, input_data, output_data, duration, success=True)
                debugger.snapshot_data(self.name, output_data, "output")
            
            logger.info(f"[{self.name}] Stage completed successfully in {duration:.4f}s")
            return output_data
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            self.status = StageStatus.FAILED
            self.error = e
            
            # Monitor tracking
            if monitor and monitor.current_metrics:
                monitor.track_stage(self.name, start_time, end_time, input_data, None, success=False, error=e)
                monitor.current_metrics.add_error(self.name, e)
            
            # Debugger trace
            if debugger:
                debugger.trace_stage(self.name, input_data, None, duration, success=False, error=e)
            
            logger.error(f"[{self.name}] Stage failed in {duration:.4f}s: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get stage status and metadata"""
        return {
            'name': self.name,
            'status': self.status.value,
            'enabled': self.enabled,
            'metadata': self.metadata,
            'error': str(self.error) if self.error else None
        }
    
    def reset(self):
        """Reset stage state"""
        self.status = StageStatus.PENDING
        self.metadata = {}
        self.input_data = None
        self.output_data = None
        self.error = None


class BasePipeline(ABC):
    """
    Base class for pipelines
    
    Pipelines orchestrate multiple stages in sequence.
    """
    
    def __init__(self, name: str, toolbox=None, enable_monitoring: bool = True,
                 enable_retry: bool = False, enable_debugging: bool = False):
        """
        Initialize pipeline
        
        Parameters
        ----------
        name : str
            Pipeline name
        toolbox : MLToolbox, optional
            ML Toolbox instance
        enable_monitoring : bool, default=True
            Enable pipeline monitoring
        enable_retry : bool, default=False
            Enable retry logic
        enable_debugging : bool, default=False
            Enable debugging
        """
        self.name = name
        self.toolbox = toolbox
        self.stages: List[PipelineStage] = []
        self.state: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        
        # Enhanced features
        self.enable_monitoring = enable_monitoring
        self.enable_retry = enable_retry
        self.enable_debugging = enable_debugging
        
        if ENHANCED_FEATURES_AVAILABLE:
            self.monitor = PipelineMonitor(enable_tracking=enable_monitoring) if enable_monitoring else None
            self.retry_handler = RetryHandler() if enable_retry else None
            self.debugger = PipelineDebugger(enable_debugging=enable_debugging) if enable_debugging else None
        else:
            self.monitor = None
            self.retry_handler = None
            self.debugger = None
    
    def add_stage(self, stage: PipelineStage):
        """Add a stage to the pipeline"""
        self.stages.append(stage)
        logger.info(f"[{self.name}] Added stage: {stage.name}")
    
    def remove_stage(self, stage_name: str):
        """Remove a stage from the pipeline"""
        self.stages = [s for s in self.stages if s.name != stage_name]
        logger.info(f"[{self.name}] Removed stage: {stage_name}")
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the pipeline
        
        Parameters
        ----------
        *args, **kwargs
            Pipeline-specific arguments
            
        Returns
        -------
        result : Any
            Pipeline execution result
        """
        pass
    
    def get_stage(self, stage_name: str) -> Optional[PipelineStage]:
        """Get a stage by name"""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            'name': self.name,
            'stages': [stage.get_status() for stage in self.stages],
            'state': self.state,
            'history_count': len(self.history)
        }
    
    def reset(self):
        """Reset pipeline state"""
        self.state = {}
        self.history = []
        for stage in self.stages:
            stage.reset()
