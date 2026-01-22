"""
Enhanced Logging Infrastructure
Structured logging for ML Toolbox

Features:
- Structured logging (JSON)
- Log levels
- File/console handlers
- Log rotation
- Integration with monitoring
"""
import logging
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import warnings

try:
    from logging.handlers import RotatingFileHandler
    ROTATING_HANDLER_AVAILABLE = True
except ImportError:
    ROTATING_HANDLER_AVAILABLE = False


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class MLToolboxLogger:
    """
    Enhanced logger for ML Toolbox
    
    Provides structured logging with rotation and multiple handlers
    """
    
    def __init__(
        self,
        name: str = 'ml_toolbox',
        level: str = 'INFO',
        format_type: str = 'json',
        log_file: Optional[str] = None,
        rotation: bool = True,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5
    ):
        """
        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_type: 'json' or 'text'
            log_file: Path to log file (optional)
            rotation: Whether to enable log rotation
            max_bytes: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if format_type == 'json':
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            if rotation and ROTATING_HANDLER_AVAILABLE:
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
            else:
                file_handler = logging.FileHandler(log_file)
            
            if format_type == 'json':
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method"""
        extra = {'extra_fields': kwargs} if kwargs else {}
        self.logger.log(level, message, extra=extra)
    
    def get_logger(self) -> logging.Logger:
        """Get underlying logger"""
        return self.logger


def setup_logging(
    level: str = 'INFO',
    format_type: str = 'json',
    log_file: Optional[str] = None,
    **kwargs
) -> MLToolboxLogger:
    """
    Setup logging for ML Toolbox
    
    Args:
        level: Log level
        format_type: 'json' or 'text'
        log_file: Path to log file
        **kwargs: Additional logger arguments
        
    Returns:
        MLToolboxLogger instance
    """
    return MLToolboxLogger(
        level=level,
        format_type=format_type,
        log_file=log_file,
        **kwargs
    )
