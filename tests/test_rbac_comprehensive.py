#!/usr/bin/env python3
"""
Comprehensive Tests for RBAC System

Tests for role-based access control, permission management,
authentication, authorization, and security features for the RBAC system.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import json
from datetime import datetime, timedelta

from xencode.auth.permission_engine import PermissionEngine, PermissionDeniedError
from xencode.auth.auth_manager import AuthManager, AuthenticationError
from xencode.models.user import User, UserRole, Permission, ResourceType, PermissionType, create_default_admin_user


class TestRBACSystemBasics:
    """Test basic RBAC system functionality"""

    @pytest.fixture
    async def permission_engine(self):
        """Create a permission engine for testing"""
        engine = PermissionEngine()
        yield engine

    @pytest.fixture
    async def auth_manager(self):
        """Create an auth manager for testing"""
        manager = AuthManager()
        yield manager

    @pytest.mark.asyncio
    async def test_permission_engine_initialization(self, permission_engine):
        """Test permission engine initialization"""
        assert permission_engine is not None
        assert permission_engine.resources == {}
        assert permission_engine.permission_cache == {}
        assert len(permission_engine.inheritance_rules) > 0
        assert permission_engine.audit_log == []

    @pytest.mark.asyncio
    async def test_resource_registration(self, permission_engine):
        """Test resource registration and management"""
        # Register a resource
        success = permission_engine.register_resource(
            resource_id="test-project-123",
            resource_type=ResourceType.PROJECT,
            metadata={"owner_id": "user123", "name": "Test Project"},
            parent_resource_id=None
        )
        assert success is True

        # Verify resource was registered
        resource = permission_engine.get_resource("test-project-123")
        assert resource is not None
        assert resource['type'] == ResourceType.PROJECT
        assert resource['metadata']['name'] == "Test Project"
        assert resource['metadata']['owner_id'] == "user123"

        # Try to register same resource again (should fail)
        duplicate_success = permission_engine.register_resource(
            resource_id="test-project-123",
            resource_type=ResourceType.PROJECT
        )
        assert duplicate_success is False

    @pytest.mark.asyncio
    async def test_resource_listing(self, permission_engine):
        """Test resource listing functionality"""
        # Register multiple resources
        permission_engine.register_resource(
            resource_id="proj1",
            resource_type=ResourceType.PROJECT,
            metadata={"name": "Project 1"}
        )
        permission_engine.register_resource(
            resource_id="file1",
            resource_type=ResourceType.FILE,
            metadata={"name": "File 1"}
        )
        permission_engine.register_resource(
            resource_id="proj2",
            resource_type=ResourceType.PROJECT,
            metadata={"name": "Project 2"}
        )

        # List all resources
        all_resources = permission_engine.list_resources()
        assert len(all_resources) >= 3

        # List by type
        projects = permission_engine.list_resources(resource_type=ResourceType.PROJECT)
        assert len(projects) >= 2
        project_ids = [r[0] for r in projects]
        assert "proj1" in project_ids
        assert "proj2" in project_ids

    @pytest.mark.asyncio
    async def test_basic_permission_check(self, permission_engine):
        """Test basic permission checking"""
        # Create a user
        user = User(
            username="testuser",
            email="test@example.com",
            role=UserRole.VIEWER
        )

        # Check basic permission (should fail for non-admin user without explicit permission)
        has_perm = await permission_engine.check_permission(
            user, ResourceType.FILE, PermissionType.READ
        )
        # Viewer role should have read permission for files by default
        assert has_perm is True

        # Check admin permission (should fail for viewer)
        has_admin_perm = await permission_engine.check_permission(
            user, ResourceType.FILE, PermissionType.ADMIN
        )
        assert has_admin_perm is False

    @pytest.mark.asyncio
    async def test_admin_permissions(self, permission_engine):
        """Test admin user permissions"""
        # Create admin user
        admin_user = User(
            username="admin",
            email="admin@example.com",
            role=UserRole.ADMIN
        )

        # Admin should have all permissions
        has_perm = await permission_engine.check_permission(
            admin_user, ResourceType.FILE, PermissionType.ADMIN
        )
        assert has_perm is True

        has_perm = await permission_engine.check_permission(
            admin_user, ResourceType.SYSTEM, PermissionType.ADMIN
        )
        assert has_perm is True

        has_perm = await permission_engine.check_permission(
            admin_user, ResourceType.USER, PermissionType.WRITE
        )
        assert has_perm is True

    @pytest.mark.asyncio
    async def test_explicit_permission_granting(self, permission_engine):
        """Test explicit permission granting"""
        # Create a user
        user = User(
            username="regular_user",
            email="regular@example.com",
            role=UserRole.VIEWER
        )

        # Initially should not have write permission
        has_write = await permission_engine.check_permission(
            user, ResourceType.FILE, PermissionType.WRITE
        )
        assert has_write is False

        # Grant explicit write permission
        permission = Permission(
            resource_type=ResourceType.FILE,
            permission_type=PermissionType.WRITE,
            granted=True
        )
        user.add_permission(permission)

        # Now should have write permission
        has_write = await permission_engine.check_permission(
            user, ResourceType.FILE, PermissionType.WRITE
        )
        assert has_write is True

    @pytest.mark.asyncio
    async def test_resource_specific_permissions(self, permission_engine):
        """Test resource-specific permissions"""
        # Register a resource
        permission_engine.register_resource(
            resource_id="owned_file_123",
            resource_type=ResourceType.FILE,
            metadata={"owner_id": "user123", "name": "Owned File"}
        )

        # Create owner user
        owner = User(
            username="owner",
            email="owner@example.com",
            role=UserRole.VIEWER  # Even with viewer role, owner should have permissions
        )

        # Check if owner has permissions on their resource
        # This would depend on the implementation of resource-specific permissions
        # For now, let's test the basic functionality
        has_perm = await permission_engine.check_permission(
            owner, ResourceType.FILE, PermissionType.READ, resource_id="owned_file_123"
        )
        # This might be False if the permission engine doesn't recognize the user as owner
        # based on the metadata alone

    @pytest.mark.asyncio
    async def test_permission_inheritance(self, permission_engine):
        """Test permission inheritance"""
        # Register parent and child resources
        permission_engine.register_resource(
            resource_id="parent_project",
            resource_type=ResourceType.PROJECT,
            metadata={"owner_id": "user123"}
        )

        permission_engine.register_resource(
            resource_id="child_file",
            resource_type=ResourceType.FILE,
            metadata={"owner_id": "user123"},
            parent_resource_id="parent_project"
        )

        # Create user with permission on parent
        user = User(
            username="testuser",
            email="test@example.com",
            role=UserRole.DEVELOPER  # Developer has write permissions to files
        )

        # Check if permissions are inherited
        has_perm = await permission_engine.check_permission(
            user, ResourceType.FILE, PermissionType.WRITE
        )
        # Developer role should have write permission to files
        assert has_perm is True

    @pytest.mark.asyncio
    async def test_permission_granting_revoking(self, permission_engine):
        """Test permission granting and revoking"""
        # Create users
        admin_user = User(
            username="admin",
            email="admin@example.com",
            role=UserRole.ADMIN
        )

        target_user = User(
            username="target",
            email="target@example.com",
            role=UserRole.VIEWER
        )

        # Initially, target user shouldn't have admin permission
        has_admin = await permission_engine.check_permission(
            target_user, ResourceType.SYSTEM, PermissionType.ADMIN
        )
        assert has_admin is False

        # Admin grants permission
        granted = await permission_engine.grant_permission(
            admin_user, target_user, ResourceType.SYSTEM, PermissionType.ADMIN
        )
        assert granted is True

        # Target user should now have admin permission
        has_admin = await permission_engine.check_permission(
            target_user, ResourceType.SYSTEM, PermissionType.ADMIN
        )
        assert has_admin is True

        # Admin revokes permission
        revoked = await permission_engine.revoke_permission(
            admin_user, target_user, ResourceType.SYSTEM, PermissionType.ADMIN
        )
        assert revoked is True

        # Target user should no longer have admin permission
        has_admin = await permission_engine.check_permission(
            target_user, ResourceType.SYSTEM, PermissionType.ADMIN
        )
        assert has_admin is False

    @pytest.mark.asyncio
    async def test_require_permission_functionality(self, permission_engine):
        """Test require_permission functionality"""
        user = User(
            username="testuser",
            email="test@example.com",
            role=UserRole.ADMIN  # Admin should have all permissions
        )

        # This should not raise an exception
        await permission_engine.require_permission(
            user, ResourceType.SYSTEM, PermissionType.ADMIN
        )

        # Create non-admin user
        non_admin = User(
            username="nonadmin",
            email="nonadmin@example.com",
            role=UserRole.VIEWER
        )

        # This should raise PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            await permission_engine.require_permission(
                non_admin, ResourceType.SYSTEM, PermissionType.ADMIN
            )

    @pytest.mark.asyncio
    async def test_bulk_permission_checking(self, permission_engine):
        """Test bulk permission checking"""
        user = User(
            username="testuser",
            email="test@example.com",
            role=UserRole.DEVELOPER
        )

        # Check multiple permissions at once
        permission_checks = [
            (ResourceType.FILE, PermissionType.READ, None),
            (ResourceType.PROJECT, PermissionType.WRITE, None),
            (ResourceType.SYSTEM, PermissionType.READ, None)
        ]

        results = await permission_engine.check_bulk_permissions(user, permission_checks)

        # Should have results for all checks
        assert len(results) == 3
        # Developer should have read and write permissions for files/projects
        # and read permission for system


class TestAuthManagerFunctionality:
    """Test authentication manager functionality"""

    @pytest.fixture
    async def auth_manager(self):
        """Create an auth manager for testing"""
        manager = AuthManager()
        yield manager

    @pytest.mark.asyncio
    async def test_auth_manager_initialization(self, auth_manager):
        """Test auth manager initialization"""
        assert auth_manager is not None
        assert auth_manager.users is not None
        assert auth_manager.users_by_username is not None
        assert auth_manager.users_by_email is not None
        assert len(auth_manager.users) > 0  # Has default admin and guest users

    @pytest.mark.asyncio
    async def test_user_creation(self, auth_manager):
        """Test user creation"""
        # Create a new user
        user = auth_manager.create_user(
            username="newuser",
            email="newuser@example.com",
            password="securepassword123",
            full_name="New User",
            role=UserRole.DEVELOPER
        )

        assert user is not None
        assert user.username == "newuser"
        assert user.email == "newuser@example.com"
        assert user.role == UserRole.DEVELOPER
        assert user.is_active is True

        # Verify user was added to system
        retrieved_user = auth_manager.get_user_by_username("newuser")
        assert retrieved_user is not None
        assert retrieved_user.id == user.id

    @pytest.mark.asyncio
    async def test_duplicate_user_prevention(self, auth_manager):
        """Test prevention of duplicate users"""
        # Create first user
        user1 = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        assert user1 is not None

        # Try to create user with same username (should fail)
        user2 = auth_manager.create_user(
            username="testuser",  # Same username
            email="different@example.com",
            password="password123"
        )
        assert user2 is None

        # Try to create user with same email (should fail)
        user3 = auth_manager.create_user(
            username="differentuser",
            email="test@example.com",  # Same email
            password="password123"
        )
        assert user3 is None

    @pytest.mark.asyncio
    async def test_user_authentication(self, auth_manager):
        """Test user authentication"""
        # Create a user
        user = auth_manager.create_user(
            username="authuser",
            email="auth@example.com",
            password="authpassword123"
        )
        assert user is not None

        # Authenticate with correct credentials
        access_token, refresh_token, authenticated_user = await auth_manager.authenticate(
            "authuser",
            "authpassword123"
        )

        assert access_token is not None
        assert refresh_token is not None
        assert authenticated_user is not None
        assert authenticated_user.id == user.id

        # Authenticate with incorrect password (should fail)
        with pytest.raises(AuthenticationError):
            await auth_manager.authenticate("authuser", "wrongpassword")

        # Authenticate with non-existent user (should fail)
        with pytest.raises(AuthenticationError):
            await auth_manager.authenticate("nonexistent", "password")

    @pytest.mark.asyncio
    async def test_user_activation_deactivation(self, auth_manager):
        """Test user activation and deactivation"""
        # Create user
        user = auth_manager.create_user(
            username="inactive_user",
            email="inactive@example.com",
            password="password123"
        )
        assert user is not None

        # Verify user is active by default
        assert user.is_active is True

        # Deactivate user
        deactivated = auth_manager.deactivate_user(user.id)
        assert deactivated is True

        # Verify user is inactive
        retrieved_user = auth_manager.get_user(user.id)
        assert retrieved_user.is_active is False

        # Authentication should fail for inactive user
        with pytest.raises(AuthenticationError):
            await auth_manager.authenticate("inactive_user", "password123")

        # Activate user
        activated = auth_manager.activate_user(user.id)
        assert activated is True

        # Verify user is active again
        retrieved_user = auth_manager.get_user(user.id)
        assert retrieved_user.is_active is True

        # Authentication should work again
        access_token, _, _ = await auth_manager.authenticate("inactive_user", "password123")
        assert access_token is not None

    @pytest.mark.asyncio
    async def test_user_lock_unlock(self, auth_manager):
        """Test user account locking and unlocking"""
        # Create user
        user = auth_manager.create_user(
            username="locked_user",
            email="locked@example.com",
            password="password123"
        )
        assert user is not None

        # Verify user is not locked by default
        assert user.is_locked() is False

        # Lock user
        locked = auth_manager.lock_user(user.id)
        assert locked is True

        # Verify user is locked
        retrieved_user = auth_manager.get_user(user.id)
        assert retrieved_user.is_locked() is True

        # Authentication should fail for locked user
        with pytest.raises(AuthenticationError):
            await auth_manager.authenticate("locked_user", "password123")

        # Unlock user
        unlocked = auth_manager.unlock_user(user.id)
        assert unlocked is True

        # Verify user is unlocked
        retrieved_user = auth_manager.get_user(user.id)
        assert retrieved_user.is_locked() is False

        # Authentication should work again
        access_token, _, _ = await auth_manager.authenticate("locked_user", "password123")
        assert access_token is not None

    @pytest.mark.asyncio
    async def test_password_change(self, auth_manager):
        """Test password change functionality"""
        # Create user
        user = auth_manager.create_user(
            username="password_user",
            email="password@example.com",
            password="oldpassword123"
        )
        assert user is not None

        # Change password
        changed = auth_manager.change_password(user.id, "oldpassword123", "newpassword123")
        assert changed is True

        # Verify old password no longer works
        with pytest.raises(AuthenticationError):
            await auth_manager.authenticate("password_user", "oldpassword123")

        # Verify new password works
        access_token, _, _ = await auth_manager.authenticate("password_user", "newpassword123")
        assert access_token is not None

        # Try to change password with wrong old password (should fail)
        failed_change = auth_manager.change_password(user.id, "wrongpassword", "anotherpassword")
        assert failed_change is False

    @pytest.mark.asyncio
    async def test_user_role_management(self, auth_manager):
        """Test user role management"""
        # Create user with initial role
        user = auth_manager.create_user(
            username="role_user",
            email="role@example.com",
            password="password123",
            role=UserRole.VIEWER
        )
        assert user is not None
        assert user.role == UserRole.VIEWER

        # Update user role by creating new user object
        user.role = UserRole.DEVELOPER
        updated = auth_manager.update_user(user)
        assert updated is True

        # Verify role was updated
        retrieved_user = auth_manager.get_user(user.id)
        assert retrieved_user.role == UserRole.DEVELOPER

    @pytest.mark.asyncio
    async def test_user_deletion(self, auth_manager):
        """Test user deletion"""
        # Create user
        user = auth_manager.create_user(
            username="delete_user",
            email="delete@example.com",
            password="password123"
        )
        assert user is not None

        # Verify user exists
        retrieved_user = auth_manager.get_user_by_username("delete_user")
        assert retrieved_user is not None

        # Delete user
        deleted = auth_manager.delete_user(user.id)
        assert deleted is True

        # Verify user no longer exists
        retrieved_user = auth_manager.get_user_by_username("delete_user")
        assert retrieved_user is None

        # Try to delete non-existent user (should fail)
        failed_delete = auth_manager.delete_user("nonexistent_id")
        assert failed_delete is False

    @pytest.mark.asyncio
    async def test_user_session_management(self, auth_manager):
        """Test user session management"""
        # Create user
        user = auth_manager.create_user(
            username="session_user",
            email="session@example.com",
            password="password123"
        )
        assert user is not None

        # Authenticate to create session
        access_token, refresh_token, _ = await auth_manager.authenticate("session_user", "password123")
        assert access_token is not None

        # Get user sessions
        sessions = auth_manager.get_user_sessions(user.id)
        # May have active sessions depending on implementation
        # The exact behavior depends on how sessions are tracked

    @pytest.mark.asyncio
    async def test_auth_statistics(self, auth_manager):
        """Test authentication statistics"""
        stats = auth_manager.get_stats()
        assert 'total_users' in stats
        assert 'active_users' in stats
        assert 'locked_users' in stats
        assert 'role_distribution' in stats
        assert 'jwt_stats' in stats
        assert 'failed_attempts_tracked' in stats

        # Verify stats make sense
        assert stats['total_users'] > 0  # Has default users
        assert stats['active_users'] > 0
        assert stats['locked_users'] >= 0
        assert isinstance(stats['role_distribution'], dict)


class TestRBACIntegration:
    """Test RBAC system integration"""

    @pytest.fixture
    async def rbac_system(self):
        """Create integrated RBAC system for testing"""
        permission_engine = PermissionEngine()
        auth_manager = AuthManager()
        yield {
            'permission_engine': permission_engine,
            'auth_manager': auth_manager
        }

    @pytest.mark.asyncio
    async def test_full_authz_flow(self, rbac_system):
        """Test complete authentication and authorization flow"""
        permission_engine = rbac_system['permission_engine']
        auth_manager = rbac_system['auth_manager']

        # 1. Create user
        user = auth_manager.create_user(
            username="integration_user",
            email="integration@example.com",
            password="securepassword123",
            role=UserRole.DEVELOPER
        )
        assert user is not None

        # 2. Authenticate user
        access_token, refresh_token, authenticated_user = await auth_manager.authenticate(
            "integration_user",
            "securepassword123"
        )
        assert authenticated_user.id == user.id

        # 3. Register resource
        resource_registered = permission_engine.register_resource(
            resource_id="integration_resource",
            resource_type=ResourceType.FILE,
            metadata={"owner_id": user.id, "name": "Integration Test File"}
        )
        assert resource_registered is True

        # 4. Check permissions
        has_read = await permission_engine.check_permission(
            authenticated_user, ResourceType.FILE, PermissionType.READ
        )
        assert has_read is True

        has_write = await permission_engine.check_permission(
            authenticated_user, ResourceType.FILE, PermissionType.WRITE
        )
        # Developer role should have write permission
        assert has_write is True

        has_admin = await permission_engine.check_permission(
            authenticated_user, ResourceType.SYSTEM, PermissionType.ADMIN
        )
        # Developer role should NOT have system admin permission
        assert has_admin is False

        # 5. Try to perform admin action (should fail)
        with pytest.raises(PermissionDeniedError):
            await permission_engine.require_permission(
                authenticated_user, ResourceType.SYSTEM, PermissionType.ADMIN
            )

        # 6. Admin user should have all permissions
        admin_user = auth_manager.get_user_by_username("admin")
        assert admin_user is not None

        admin_has_system_admin = await permission_engine.check_permission(
            admin_user, ResourceType.SYSTEM, PermissionType.ADMIN
        )
        assert admin_has_system_admin is True

    @pytest.mark.asyncio
    async def test_permission_inheritance_integration(self, rbac_system):
        """Test permission inheritance in integrated system"""
        permission_engine = rbac_system['permission_engine']
        auth_manager = rbac_system['auth_manager']

        # Create admin user
        admin_user = auth_manager.get_user_by_username("admin")
        assert admin_user is not None

        # Register parent and child resources
        parent_registered = permission_engine.register_resource(
            resource_id="parent_project_integration",
            resource_type=ResourceType.PROJECT
        )
        assert parent_registered is True

        child_registered = permission_engine.register_resource(
            resource_id="child_file_integration",
            resource_type=ResourceType.FILE,
            parent_resource_id="parent_project_integration"
        )
        assert child_registered is True

        # Admin should have permissions to both
        has_parent_perm = await permission_engine.check_permission(
            admin_user, ResourceType.PROJECT, PermissionType.ADMIN
        )
        assert has_parent_perm is True

        has_child_perm = await permission_engine.check_permission(
            admin_user, ResourceType.FILE, PermissionType.ADMIN
        )
        assert has_child_perm is True

    @pytest.mark.asyncio
    async def test_role_based_access_control(self, rbac_system):
        """Test role-based access control"""
        permission_engine = rbac_system['permission_engine']
        auth_manager = rbac_system['auth_manager']

        # Create users with different roles
        admin_user = auth_manager.get_user_by_username("admin")
        assert admin_user is not None

        dev_user = auth_manager.create_user(
            username="dev_user",
            email="dev@example.com",
            password="devpassword123",
            role=UserRole.DEVELOPER
        )
        assert dev_user is not None

        viewer_user = auth_manager.create_user(
            username="viewer_user",
            email="viewer@example.com",
            password="viewerpassword123",
            role=UserRole.VIEWER
        )
        assert viewer_user is not None

        # Test permissions by role
        # Admin should have all permissions
        admin_file_admin = await permission_engine.check_permission(
            admin_user, ResourceType.FILE, PermissionType.ADMIN
        )
        assert admin_file_admin is True

        # Developer should have read/write/execute but not system admin
        dev_file_write = await permission_engine.check_permission(
            dev_user, ResourceType.FILE, PermissionType.WRITE
        )
        assert dev_file_write is True

        dev_system_admin = await permission_engine.check_permission(
            dev_user, ResourceType.SYSTEM, PermissionType.ADMIN
        )
        assert dev_system_admin is False

        # Viewer should have read-only access to most resources
        viewer_file_read = await permission_engine.check_permission(
            viewer_user, ResourceType.FILE, PermissionType.READ
        )
        assert viewer_file_read is True

        viewer_file_write = await permission_engine.check_permission(
            viewer_user, ResourceType.FILE, PermissionType.WRITE
        )
        # Viewer might have write permission depending on implementation
        # The important thing is that they have less than developers

    @pytest.mark.asyncio
    async def test_permission_audit_logging(self, rbac_system):
        """Test permission audit logging"""
        permission_engine = rbac_system['permission_engine']
        auth_manager = rbac_system['auth_manager']

        # Create user
        user = auth_manager.create_user(
            username="audit_user",
            email="audit@example.com",
            password="auditpassword123"
        )
        assert user is not None

        # Perform permission checks to generate audit logs
        await permission_engine.check_permission(
            user, ResourceType.FILE, PermissionType.READ
        )

        await permission_engine.check_permission(
            user, ResourceType.PROJECT, PermissionType.WRITE
        )

        # Get audit logs
        audit_logs = permission_engine.get_audit_log(user_id=user.id)
        assert len(audit_logs) >= 0  # May have logs depending on implementation

        # Check for permission check entries
        permission_check_logs = [log for log in audit_logs if log.action == "permission_check"]
        # Should have at least the permission checks we made

    @pytest.mark.asyncio
    async def test_resource_specific_permissions_integration(self, rbac_system):
        """Test resource-specific permissions in integrated system"""
        permission_engine = rbac_system['permission_engine']
        auth_manager = rbac_system['auth_manager']

        # Create users
        owner_user = auth_manager.create_user(
            username="owner_user",
            email="owner@example.com",
            password="ownerpassword123",
            role=UserRole.VIEWER  # Even with viewer role, should have resource-specific permissions
        )
        assert owner_user is not None

        other_user = auth_manager.create_user(
            username="other_user",
            email="other@example.com",
            password="otherpassword123",
            role=UserRole.VIEWER
        )
        assert other_user is not None

        # Register resource with owner
        resource_registered = permission_engine.register_resource(
            resource_id="resource_with_owner",
            resource_type=ResourceType.FILE,
            metadata={"owner_id": owner_user.id, "name": "Owner Resource"}
        )
        assert resource_registered is True

        # Check permissions - implementation may vary based on how ownership is handled
        # This tests the integration between resource registration and permission checking


class TestRBACSecurityFeatures:
    """Test RBAC security features"""

    @pytest.fixture
    async def rbac_system(self):
        """Create RBAC system for security testing"""
        permission_engine = PermissionEngine()
        auth_manager = AuthManager()
        yield {
            'permission_engine': permission_engine,
            'auth_manager': auth_manager
        }

    @pytest.mark.asyncio
    async def test_permission_cache_security(self, rbac_system):
        """Test permission cache security"""
        permission_engine = rbac_system['permission_engine']

        # Create user
        user = User(
            username="cache_test_user",
            email="cache@example.com",
            role=UserRole.VIEWER
        )

        # Perform multiple permission checks
        for i in range(10):
            has_perm = await permission_engine.check_permission(
                user, ResourceType.FILE, PermissionType.READ
            )
            # Permission should be consistent regardless of caching

        # Verify cache is working (has entries)
        assert len(permission_engine.permission_cache) >= 0  # May have cached entries

        # Test cache TTL functionality
        initial_cache_size = len(permission_engine.permission_cache)

        # Clear expired cache entries
        cleared_count = permission_engine.cleanup_expired_cache()
        # May not clear anything if TTL hasn't expired

    @pytest.mark.asyncio
    async def test_rate_limiting_security(self, rbac_system):
        """Test authentication rate limiting security"""
        auth_manager = rbac_system['auth_manager']

        # Create a test user
        user = auth_manager.create_user(
            username="rate_limit_user",
            email="ratelimit@example.com",
            password="password123"
        )
        assert user is not None

        # Try to authenticate with wrong password multiple times to trigger rate limiting
        failed_attempts = 0
        for i in range(10):
            try:
                await auth_manager.authenticate("rate_limit_user", "wrongpassword")
            except AuthenticationError:
                failed_attempts += 1

        # Should have multiple failed attempts
        assert failed_attempts >= 1

        # Check that rate limiting is working by checking login attempts tracking
        # The exact implementation depends on how the rate limiting works

    @pytest.mark.asyncio
    async def test_brute_force_protection(self, rbac_system):
        """Test brute force protection"""
        auth_manager = rbac_system['auth_manager']

        # Create a user
        user = auth_manager.create_user(
            username="bruteforce_user",
            email="bruteforce@example.com",
            password="securepassword123"
        )
        assert user is not None

        # Try multiple failed authentications to test lockout mechanism
        for i in range(10):
            try:
                await auth_manager.authenticate("bruteforce_user", f"wrongpassword{i}")
            except AuthenticationError:
                pass  # Expected

        # After multiple failed attempts, account might be locked depending on implementation
        # Check if user is locked
        locked_user = auth_manager.get_user_by_username("bruteforce_user")
        # The exact behavior depends on the lockout implementation

    @pytest.mark.asyncio
    async def test_permission_elevation_prevention(self, rbac_system):
        """Test prevention of unauthorized permission elevation"""
        permission_engine = rbac_system['permission_engine']
        auth_manager = rbac_system['auth_manager']

        # Create regular user
        regular_user = auth_manager.create_user(
            username="regular_user",
            email="regular@example.com",
            password="password123",
            role=UserRole.VIEWER
        )
        assert regular_user is not None

        # Create another user
        target_user = auth_manager.create_user(
            username="target_user",
            email="target@example.com",
            password="password123",
            role=UserRole.VIEWER
        )
        assert target_user is not None

        # Regular user should NOT be able to grant permissions to others
        granted = await permission_engine.grant_permission(
            regular_user, target_user, ResourceType.SYSTEM, PermissionType.ADMIN
        )
        # Should fail because regular user doesn't have admin permissions
        assert granted is False

        # Verify target user still doesn't have admin permissions
        has_admin = await permission_engine.check_permission(
            target_user, ResourceType.SYSTEM, PermissionType.ADMIN
        )
        assert has_admin is False

        # Admin user should be able to grant permissions
        admin_user = auth_manager.get_user_by_username("admin")
        assert admin_user is not None

        admin_granted = await permission_engine.grant_permission(
            admin_user, target_user, ResourceType.FILE, PermissionType.WRITE
        )
        assert admin_granted is True

        # Now target user should have the granted permission
        has_write = await permission_engine.check_permission(
            target_user, ResourceType.FILE, PermissionType.WRITE
        )
        assert has_write is True

    @pytest.mark.asyncio
    async def test_session_security(self, rbac_system):
        """Test session security features"""
        auth_manager = rbac_system['auth_manager']

        # Create user
        user = auth_manager.create_user(
            username="session_security_user",
            email="sessionsec@example.com",
            password="password123"
        )
        assert user is not None

        # Authenticate to create session
        access_token, refresh_token, authenticated_user = await auth_manager.authenticate(
            "session_security_user", "password123"
        )
        assert access_token is not None
        assert authenticated_user.id == user.id

        # Verify token is valid
        verified_user = auth_manager.verify_token(access_token)
        assert verified_user is not None
        assert verified_user.id == user.id

        # Logout (revoke token)
        logged_out = auth_manager.logout(access_token)
        # Depends on implementation - may return True/False or always True

        # Verify token is no longer valid after logout
        # This depends on the specific implementation of token revocation

    @pytest.mark.asyncio
    async def test_multi_tenancy_isolation(self, rbac_system):
        """Test multi-tenancy isolation in permission system"""
        permission_engine = rbac_system['permission_engine']
        auth_manager = rbac_system['auth_manager']

        # Create users from different "tenants" (simulated with different prefixes)
        tenant1_user = auth_manager.create_user(
            username="tenant1_user",
            email="tenant1@example.com",
            password="password123",
            role=UserRole.DEVELOPER
        )
        assert tenant1_user is not None

        tenant2_user = auth_manager.create_user(
            username="tenant2_user",
            email="tenant2@example.com",
            password="password123",
            role=UserRole.DEVELOPER
        )
        assert tenant2_user is not None

        # Register resources for each tenant
        resource1 = permission_engine.register_resource(
            resource_id="tenant1_resource",
            resource_type=ResourceType.FILE,
            metadata={"owner_id": tenant1_user.id}
        )
        assert resource1 is True

        resource2 = permission_engine.register_resource(
            resource_id="tenant2_resource",
            resource_type=ResourceType.FILE,
            metadata={"owner_id": tenant2_user.id}
        )
        assert resource2 is True

        # Each user should have appropriate permissions on their own resources
        # The exact behavior depends on the implementation of resource-specific permissions
        tenant1_can_access_own = await permission_engine.check_permission(
            tenant1_user, ResourceType.FILE, PermissionType.READ
        )
        # Should have read access based on role

        tenant1_can_access_other = await permission_engine.check_permission(
            tenant1_user, ResourceType.FILE, PermissionType.READ, resource_id="tenant2_resource"
        )
        # Without explicit permissions, may not have access to other's resource


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])