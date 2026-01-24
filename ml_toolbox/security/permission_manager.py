"""
Permission Management System
Extracted from PocketFence-Family repository concepts

Features:
- Permission-based access control
- Family/group permissions
- Role-based access
- Permission inheritance
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import json
import datetime
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class Permission:
    """Represents a single permission"""
    
    def __init__(self, name: str, description: str = "", level: str = "read"):
        """
        Initialize permission
        
        Args:
            name: Permission name
            description: Permission description
            level: Permission level ('read', 'write', 'execute', 'admin')
        """
        self.name = name
        self.description = description
        self.level = level
        self.created_at = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert permission to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'level': self.level,
            'created_at': self.created_at.isoformat()
        }


class Role:
    """Represents a role with associated permissions"""
    
    def __init__(self, role_name: str, permissions: Optional[List[Permission]] = None):
        """
        Initialize role
        
        Args:
            role_name: Name of the role
            permissions: List of permissions for this role
        """
        self.role_name = role_name
        self.permissions: Set[str] = set()
        if permissions:
            for perm in permissions:
                self.permissions.add(perm.name)
        self.created_at = datetime.datetime.now()
    
    def add_permission(self, permission: Permission):
        """Add permission to role"""
        self.permissions.add(permission.name)
    
    def has_permission(self, permission_name: str) -> bool:
        """Check if role has permission"""
        return permission_name in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert role to dictionary"""
        return {
            'role_name': self.role_name,
            'permissions': list(self.permissions),
            'created_at': self.created_at.isoformat()
        }


class User:
    """Represents a user with roles and permissions"""
    
    def __init__(self, user_id: str, username: str, roles: Optional[List[Role]] = None):
        """
        Initialize user
        
        Args:
            user_id: Unique user identifier
            username: Username
            roles: List of roles assigned to user
        """
        self.user_id = user_id
        self.username = username
        self.roles: List[Role] = roles or []
        self.direct_permissions: Set[str] = set()
        self.created_at = datetime.datetime.now()
    
    def add_role(self, role: Role):
        """Add role to user"""
        if role not in self.roles:
            self.roles.append(role)
    
    def add_permission(self, permission: Permission):
        """Add direct permission to user"""
        self.direct_permissions.add(permission.name)
    
    def has_permission(self, permission_name: str) -> bool:
        """Check if user has permission (through roles or direct)"""
        # Check direct permissions
        if permission_name in self.direct_permissions:
            return True
        
        # Check role permissions
        for role in self.roles:
            if role.has_permission(permission_name):
                return True
        
        return False
    
    def get_all_permissions(self) -> Set[str]:
        """Get all permissions for user (from roles and direct)"""
        all_perms = set(self.direct_permissions)
        for role in self.roles:
            all_perms.update(role.permissions)
        return all_perms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'roles': [role.role_name for role in self.roles],
            'permissions': list(self.get_all_permissions()),
            'created_at': self.created_at.isoformat()
        }


class Group:
    """Represents a group (family/team) with users and permissions"""
    
    def __init__(self, group_id: str, group_name: str, parent_group: Optional['Group'] = None):
        """
        Initialize group
        
        Args:
            group_id: Unique group identifier
            group_name: Group name
            parent_group: Parent group (for inheritance)
        """
        self.group_id = group_id
        self.group_name = group_name
        self.parent_group = parent_group
        self.users: List[User] = []
        self.roles: List[Role] = []
        self.permissions: Set[str] = set()
        self.created_at = datetime.datetime.now()
    
    def add_user(self, user: User):
        """Add user to group"""
        if user not in self.users:
            self.users.append(user)
    
    def add_role(self, role: Role):
        """Add role to group"""
        if role not in self.roles:
            self.roles.append(role)
            # Add role permissions to group
            self.permissions.update(role.permissions)
    
    def has_permission(self, permission_name: str) -> bool:
        """Check if group has permission (including inherited)"""
        # Check direct permissions
        if permission_name in self.permissions:
            return True
        
        # Check inherited permissions from parent
        if self.parent_group:
            return self.parent_group.has_permission(permission_name)
        
        return False
    
    def get_all_users(self) -> List[User]:
        """Get all users in group (including inherited)"""
        users = list(self.users)
        if self.parent_group:
            users.extend(self.parent_group.get_all_users())
        return users
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert group to dictionary"""
        return {
            'group_id': self.group_id,
            'group_name': self.group_name,
            'parent_group': self.parent_group.group_id if self.parent_group else None,
            'users': [user.user_id for user in self.users],
            'roles': [role.role_name for role in self.roles],
            'permissions': list(self.permissions),
            'created_at': self.created_at.isoformat()
        }


class PermissionManager:
    """
    Permission Management System
    
    Manages users, roles, groups, and permissions
    """
    
    def __init__(self):
        """Initialize permission manager"""
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.groups: Dict[str, Group] = {}
    
    def create_permission(self, name: str, description: str = "", level: str = "read") -> Permission:
        """Create a new permission"""
        perm = Permission(name, description, level)
        self.permissions[name] = perm
        return perm
    
    def create_role(self, role_name: str, permission_names: Optional[List[str]] = None) -> Role:
        """Create a new role"""
        permissions = []
        if permission_names:
            for perm_name in permission_names:
                if perm_name in self.permissions:
                    permissions.append(self.permissions[perm_name])
        
        role = Role(role_name, permissions)
        self.roles[role_name] = role
        return role
    
    def create_user(self, user_id: str, username: str, role_names: Optional[List[str]] = None) -> User:
        """Create a new user"""
        roles = []
        if role_names:
            for role_name in role_names:
                if role_name in self.roles:
                    roles.append(self.roles[role_name])
        
        user = User(user_id, username, roles)
        self.users[user_id] = user
        return user
    
    def create_group(self, group_id: str, group_name: str, 
                    parent_group_id: Optional[str] = None) -> Group:
        """Create a new group"""
        parent_group = None
        if parent_group_id and parent_group_id in self.groups:
            parent_group = self.groups[parent_group_id]
        
        group = Group(group_id, group_name, parent_group)
        self.groups[group_id] = group
        return group
    
    def check_permission(self, user_id: str, permission_name: str) -> bool:
        """
        Check if user has permission
        
        Args:
            user_id: User identifier
            permission_name: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        # Check user's direct permissions
        if user.has_permission(permission_name):
            return True
        
        # Check group permissions
        for group in self.groups.values():
            if user in group.users:
                if group.has_permission(permission_name):
                    return True
        
        return False
    
    def grant_permission(self, user_id: str, permission_name: str):
        """Grant permission to user"""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        if permission_name not in self.permissions:
            raise ValueError(f"Permission {permission_name} not found")
        
        self.users[user_id].add_permission(self.permissions[permission_name])
    
    def revoke_permission(self, user_id: str, permission_name: str):
        """Revoke permission from user"""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        self.users[user_id].direct_permissions.discard(permission_name)
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for user"""
        if user_id not in self.users:
            return set()
        
        return self.users[user_id].get_all_permissions()
    
    def export_config(self) -> Dict[str, Any]:
        """Export permission configuration"""
        return {
            'permissions': {name: perm.to_dict() for name, perm in self.permissions.items()},
            'roles': {name: role.to_dict() for name, role in self.roles.items()},
            'users': {uid: user.to_dict() for uid, user in self.users.items()},
            'groups': {gid: group.to_dict() for gid, group in self.groups.items()}
        }


def get_permission_manager() -> PermissionManager:
    """Get or create permission manager instance"""
    return PermissionManager()
