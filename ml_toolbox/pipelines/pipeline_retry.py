"""
Pipeline Retry Logic

Retry failed stages with exponential backoff and error recovery.
"""
from typing import Any, Dict, Optional, Callable
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategies"""
    NONE = "none"
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"


class RetryConfig:
    """Configuration for retry logic"""
    
    def __init__(self, max_retries: int = 3, strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
                 base_delay: float = 1.0, max_delay: float = 60.0,
                 retryable_errors: Optional[list] = None):
        """
        Initialize retry configuration
        
        Parameters
        ----------
        max_retries : int, default=3
            Maximum number of retry attempts
        strategy : RetryStrategy, default=EXPONENTIAL_BACKOFF
            Retry strategy
        base_delay : float, default=1.0
            Base delay in seconds
        max_delay : float, default=60.0
            Maximum delay in seconds
        retryable_errors : list, optional
            List of error types that should be retried
        """
        self.max_retries = max_retries
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_errors = retryable_errors or [Exception]
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Check if error should be retried"""
        if attempt >= self.max_retries:
            return False
        
        # Check if error type is retryable
        for retryable_error in self.retryable_errors:
            if isinstance(error, retryable_error):
                return True
        
        return False
    
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry"""
        if self.strategy == RetryStrategy.NONE:
            return 0.0
        elif self.strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2 ** attempt)
            return min(delay, self.max_delay)
        elif self.strategy == RetryStrategy.FIXED_INTERVAL:
            return self.base_delay
        else:
            return 0.0


class RetryHandler:
    """
    Handle retries for pipeline stages
    
    Provides:
    - Automatic retry on failure
    - Configurable retry strategies
    - Error recovery
    - Retry logging
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler
        
        Parameters
        ----------
        config : RetryConfig, optional
            Retry configuration (default: RetryConfig())
        """
        self.config = config or RetryConfig()
        self.retry_history: Dict[str, list] = {}
    
    def execute_with_retry(self, stage_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic
        
        Parameters
        ----------
        stage_name : str
            Stage name for logging
        func : callable
            Function to execute
        *args, **kwargs
            Arguments for function
            
        Returns
        -------
        result : Any
            Function result
            
        Raises
        ------
        Exception
            Last exception if all retries fail
        """
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # Record successful attempt
                if attempt > 0:
                    logger.info(f"[RetryHandler] {stage_name} succeeded on attempt {attempt + 1}")
                    self._record_retry(stage_name, attempt + 1, success=True)
                
                return result
            
            except Exception as e:
                last_error = e
                
                # Check if should retry
                if not self.config.should_retry(e, attempt):
                    logger.error(f"[RetryHandler] {stage_name} failed (non-retryable): {e}")
                    self._record_retry(stage_name, attempt + 1, success=False, error=e)
                    raise
                
                # Calculate delay
                if attempt < self.config.max_retries:
                    delay = self.config.get_delay(attempt)
                    logger.warning(f"[RetryHandler] {stage_name} failed on attempt {attempt + 1}/{self.config.max_retries + 1}: {e}. Retrying in {delay:.2f}s...")
                    self._record_retry(stage_name, attempt + 1, success=False, error=e)
                    
                    if delay > 0:
                        time.sleep(delay)
        
        # All retries exhausted
        logger.error(f"[RetryHandler] {stage_name} failed after {self.config.max_retries + 1} attempts")
        self._record_retry(stage_name, self.config.max_retries + 1, success=False, error=last_error)
        raise last_error
    
    def _record_retry(self, stage_name: str, attempt: int, success: bool, error: Optional[Exception] = None):
        """Record retry attempt"""
        if stage_name not in self.retry_history:
            self.retry_history[stage_name] = []
        
        self.retry_history[stage_name].append({
            'attempt': attempt,
            'success': success,
            'error': str(error) if error else None,
            'timestamp': time.time()
        })
    
    def get_retry_statistics(self, stage_name: Optional[str] = None) -> Dict[str, Any]:
        """Get retry statistics"""
        if stage_name:
            if stage_name not in self.retry_history:
                return {}
            
            retries = self.retry_history[stage_name]
            return {
                'total_attempts': len(retries),
                'successful_attempts': sum(1 for r in retries if r['success']),
                'failed_attempts': sum(1 for r in retries if not r['success']),
                'retry_history': retries
            }
        else:
            return {
                stage: self.get_retry_statistics(stage)
                for stage in self.retry_history.keys()
            }
