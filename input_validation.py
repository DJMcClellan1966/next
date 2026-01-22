"""
Input Validation
Validate inputs for ML Toolbox components

Features:
- Type checking
- Range validation
- Format validation
- Custom validators
- Error messages
"""
from typing import Any, Optional, Callable, List, Tuple
import numpy as np
from pathlib import Path
import warnings


class ValidationError(Exception):
    """Custom validation error"""
    pass


class InputValidator:
    """Input validator for ML Toolbox"""
    
    @staticmethod
    def validate_array(
        X: Any,
        min_dim: int = 1,
        max_dim: int = 2,
        min_samples: int = 1,
        allow_empty: bool = False,
        dtype: Optional[type] = None
    ) -> np.ndarray:
        """
        Validate array input
        
        Args:
            X: Input to validate
            min_dim: Minimum dimensions
            max_dim: Maximum dimensions
            min_samples: Minimum number of samples
            allow_empty: Whether to allow empty arrays
            dtype: Expected data type
            
        Returns:
            Validated numpy array
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            X = np.asarray(X)
        except Exception as e:
            raise ValidationError(f"Could not convert to array: {e}")
        
        if not allow_empty and len(X) == 0:
            raise ValidationError("Array cannot be empty")
        
        if X.ndim < min_dim or X.ndim > max_dim:
            raise ValidationError(
                f"Array must have {min_dim}-{max_dim} dimensions, got {X.ndim}"
            )
        
        if X.ndim == 2 and X.shape[0] < min_samples:
            raise ValidationError(
                f"Array must have at least {min_samples} samples, got {X.shape[0]}"
            )
        
        if dtype and not np.issubdtype(X.dtype, dtype):
            warnings.warn(f"Expected dtype {dtype}, got {X.dtype}")
        
        return X
    
    @staticmethod
    def validate_range(
        value: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "value"
    ) -> float:
        """
        Validate value is in range
        
        Args:
            value: Value to validate
            min_val: Minimum value
            max_val: Maximum value
            name: Name of parameter (for error messages)
            
        Returns:
            Validated value
            
        Raises:
            ValidationError: If validation fails
        """
        if min_val is not None and value < min_val:
            raise ValidationError(
                f"{name} must be >= {min_val}, got {value}"
            )
        
        if max_val is not None and value > max_val:
            raise ValidationError(
                f"{name} must be <= {max_val}, got {value}"
            )
        
        return value
    
    @staticmethod
    def validate_file_path(
        path: Any,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False
    ) -> Path:
        """
        Validate file path
        
        Args:
            path: Path to validate
            must_exist: Whether path must exist
            must_be_file: Whether path must be a file
            must_be_dir: Whether path must be a directory
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            path = Path(path)
        except Exception as e:
            raise ValidationError(f"Invalid path: {e}")
        
        if must_exist and not path.exists():
            raise ValidationError(f"Path does not exist: {path}")
        
        if must_be_file and not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        
        if must_be_dir and not path.is_dir():
            raise ValidationError(f"Path is not a directory: {path}")
        
        return path
    
    @staticmethod
    def validate_model(
        model: Any,
        required_methods: List[str] = ['predict']
    ) -> Any:
        """
        Validate model has required methods
        
        Args:
            model: Model to validate
            required_methods: List of required method names
            
        Returns:
            Validated model
            
        Raises:
            ValidationError: If validation fails
        """
        for method_name in required_methods:
            if not hasattr(model, method_name):
                raise ValidationError(
                    f"Model must have '{method_name}' method"
                )
        
        return model
    
    @staticmethod
    def validate_config(
        config: dict,
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None
    ) -> dict:
        """
        Validate configuration dictionary
        
        Args:
            config: Configuration dictionary
            required_keys: List of required keys
            optional_keys: List of optional keys
            
        Returns:
            Validated configuration
            
        Raises:
            ValidationError: If validation fails
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValidationError(
                f"Missing required configuration keys: {missing_keys}"
            )
        
        if optional_keys:
            unknown_keys = [
                key for key in config.keys()
                if key not in required_keys and key not in optional_keys
            ]
            if unknown_keys:
                warnings.warn(f"Unknown configuration keys: {unknown_keys}")
        
        return config
