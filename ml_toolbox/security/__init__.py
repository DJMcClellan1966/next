"""
ML Toolbox Security Module
ML security framework and threat detection
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ml_toolbox.security.ml_security_framework import MLSecurityFramework
    from ml_toolbox.security.permission_manager import (
        Permission, Role, User, Group, PermissionManager, get_permission_manager
    )
    __all__ = [
        'MLSecurityFramework',
        'Permission',
        'Role',
        'User',
        'Group',
        'PermissionManager',
        'get_permission_manager'
    ]
except ImportError as e:
    __all__ = []
    import warnings
    warnings.warn(f"Security module imports failed: {e}")
