"""
Configuration Management
Centralized configuration for ML Toolbox

Features:
- Environment variable support
- Config files (YAML/JSON)
- Default configurations
- Configuration validation
- Secrets management
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import warnings


@dataclass
class MLToolboxConfig:
    """
    ML Toolbox Configuration
    
    Centralized configuration for all compartments
    """
    # Data Compartment
    data_preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        'dedup_threshold': 0.85,
        'enable_compression': True,
        'compression_ratio': 0.5,
        'enable_scrubbing': True,
        'use_advanced_scrubbing': False
    })
    
    # Infrastructure Compartment
    quantum_kernel: Dict[str, Any] = field(default_factory=lambda: {
        'use_sentence_transformers': True,
        'use_quantum_methods': True,
        'similarity_metric': 'quantum',
        'cache_size': 50000,
        'enable_caching': True
    })
    
    # Algorithms Compartment
    algorithms: Dict[str, Any] = field(default_factory=lambda: {
        'cv_folds': 5,
        'random_state': 42,
        'n_jobs': -1
    })
    
    # MLOps Compartment
    mlops: Dict[str, Any] = field(default_factory=lambda: {
        'monitoring': {
            'data_drift_alpha': 0.05,
            'concept_drift_threshold': 0.1,
            'performance_window_size': 100
        },
        'deployment': {
            'api_host': '0.0.0.0',
            'api_port': 8000,
            'api_timeout': 30
        },
        'experiment_tracking': {
            'storage_dir': 'experiments',
            'auto_save': True
        }
    })
    
    # Logging
    logging: Dict[str, Any] = field(default_factory=lambda: {
        'level': 'INFO',
        'format': 'json',  # 'json' or 'text'
        'file': None,  # Path to log file
        'rotation': True,
        'max_bytes': 10485760,  # 10MB
        'backup_count': 5
    })
    
    # Security
    security: Dict[str, Any] = field(default_factory=lambda: {
        'api_key_required': False,
        'api_keys': [],  # List of valid API keys
        'rate_limit': {
            'enabled': False,
            'requests_per_minute': 60
        },
        'cors': {
            'enabled': True,
            'allowed_origins': ['*']
        }
    })
    
    # Model Persistence
    model_persistence: Dict[str, Any] = field(default_factory=lambda: {
        'storage_dir': 'models',
        'format': 'pickle',  # 'pickle', 'joblib', 'onnx'
        'compress': False,
        'include_metadata': True
    })
    
    @classmethod
    def from_env(cls) -> 'MLToolboxConfig':
        """
        Load configuration from environment variables
        
        Environment variables:
        - MLTOOLBOX_DATA_DEDUP_THRESHOLD
        - MLTOOLBOX_QUANTUM_KERNEL_CACHE_SIZE
        - MLTOOLBOX_MLOPS_API_PORT
        - MLTOOLBOX_LOGGING_LEVEL
        - etc.
        """
        config = cls()
        
        # Data preprocessing
        if 'MLTOOLBOX_DATA_DEDUP_THRESHOLD' in os.environ:
            config.data_preprocessing['dedup_threshold'] = float(
                os.environ['MLTOOLBOX_DATA_DEDUP_THRESHOLD']
            )
        
        # Quantum kernel
        if 'MLTOOLBOX_QUANTUM_KERNEL_CACHE_SIZE' in os.environ:
            config.quantum_kernel['cache_size'] = int(
                os.environ['MLTOOLBOX_QUANTUM_KERNEL_CACHE_SIZE']
            )
        
        # MLOps
        if 'MLTOOLBOX_MLOPS_API_PORT' in os.environ:
            config.mlops['deployment']['api_port'] = int(
                os.environ['MLTOOLBOX_MLOPS_API_PORT']
            )
        
        # Logging
        if 'MLTOOLBOX_LOGGING_LEVEL' in os.environ:
            config.logging['level'] = os.environ['MLTOOLBOX_LOGGING_LEVEL']
        
        if 'MLTOOLBOX_LOGGING_FILE' in os.environ:
            config.logging['file'] = os.environ['MLTOOLBOX_LOGGING_FILE']
        
        # Security
        if 'MLTOOLBOX_API_KEY' in os.environ:
            config.security['api_key_required'] = True
            config.security['api_keys'] = [
                key.strip() for key in os.environ['MLTOOLBOX_API_KEY'].split(',')
            ]
        
        return config
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'MLToolboxConfig':
        """
        Load configuration from file (YAML or JSON)
        
        Args:
            config_path: Path to config file
            
        Returns:
            MLToolboxConfig instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                try:
                    data = yaml.safe_load(f)
                except ImportError:
                    warnings.warn("PyYAML not available. Install with: pip install pyyaml")
                    raise ImportError("PyYAML required for YAML config files")
            elif config_path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        config = cls()
        
        # Update config from file
        for key, value in data.items():
            if hasattr(config, key):
                if isinstance(value, dict) and isinstance(getattr(config, key), dict):
                    getattr(config, key).update(value)
                else:
                    setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_file(self, config_path: Union[str, Path], format: str = 'yaml'):
        """
        Save configuration to file
        
        Args:
            config_path: Path to save config file
            format: 'yaml' or 'json'
        """
        config_path = Path(config_path)
        data = self.to_dict()
        
        with open(config_path, 'w') as f:
            if format == 'yaml':
                try:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                except ImportError:
                    warnings.warn("PyYAML not available. Install with: pip install pyyaml")
                    raise ImportError("PyYAML required for YAML config files")
            elif format == 'json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate configuration
        
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings_list = []
        
        # Validate data preprocessing
        if not 0 <= self.data_preprocessing['dedup_threshold'] <= 1:
            errors.append("data_preprocessing.dedup_threshold must be between 0 and 1")
        
        # Validate compression ratio
        if not 0 < self.data_preprocessing['compression_ratio'] <= 1:
            errors.append("data_preprocessing.compression_ratio must be between 0 and 1")
        
        # Validate MLOps API port
        if not 1 <= self.mlops['deployment']['api_port'] <= 65535:
            errors.append("mlops.deployment.api_port must be between 1 and 65535")
        
        # Validate logging level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging['level'] not in valid_levels:
            errors.append(f"logging.level must be one of {valid_levels}")
        
        # Warnings
        if self.security['api_key_required'] and not self.security['api_keys']:
            warnings_list.append("API key required but no API keys configured")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings_list
        }


def get_config(
    config_path: Optional[Union[str, Path]] = None,
    use_env: bool = True
) -> MLToolboxConfig:
    """
    Get ML Toolbox configuration
    
    Args:
        config_path: Optional path to config file
        use_env: Whether to load from environment variables
        
    Returns:
        MLToolboxConfig instance
    """
    if config_path:
        config = MLToolboxConfig.from_file(config_path)
    elif use_env:
        config = MLToolboxConfig.from_env()
    else:
        config = MLToolboxConfig()
    
    # Validate
    validation = config.validate()
    if not validation['valid']:
        warnings.warn(f"Configuration validation failed: {validation['errors']}")
    
    if validation['warnings']:
        for warning in validation['warnings']:
            warnings.warn(warning)
    
    return config
