#!/usr/bin/env python3
"""
Authentication Manager

Manages user authentication, login/logout, password validation,
and integrates with JWT handler for token-based authentication.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from xencode.models.user import User, UserSession, UserRole, create_default_admin_user, create_guest_user
from xencode.auth.jwt_handler import JWTHandler


class AuthenticationError(Exception):
    """Authentication-related errors"""
    pass


class AuthorizationError(Exception):
    """Authorization-related errors"""
    pass


class AuthManager:
    """Manages authentication and user sessions"""
    
    def __init__(self, jwt_handler: Optional[JWTHandler] = None):
        self.jwt_handler = jwt_handler or JWTHandler()
        
        # In-memory user store (in production, this would be a database)
        self.users: Dict[str, User] = {}
        self.users_by_username: Dict[str, str] = {}  # username -> user_id mapping
        self.users_by_email: Dict[str, str] = {}     # email -> user_id mapping
        
        # Initialize with default admin user
        self._initialize_default_users()
        
        # Login attempt tracking
        self.login_attempts: Dict[str, List[datetime]] = {}
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 30
    
    def _initialize_default_users(self) -> None:
        """Initialize system with default users"""
        
        # Create default admin user
        admin_user = create_default_admin_user()
        self.add_user(admin_user)
        
        # Create guest user
        guest_user = create_guest_user()
        self.add_user(guest_user)
    
    def add_user(self, user: User) -> bool:
        """Add user to the system"""
        
        # Check if username or email already exists
        if user.username in self.users_by_username:
            return False
        
        if user.email in self.users_by_email:
            return False
        
        # Add user
        self.users[user.id] = user
        self.users_by_username[user.username] = user.id
        self.users_by_email[user.email] = user.id
        
        return True
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        user_id = self.users_by_username.get(username)
        return self.users.get(user_id) if user_id else None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        user_id = self.users_by_email.get(email)
        return self.users.get(user_id) if user_id else None
    
    def update_user(self, user: User) -> bool:
        """Update user information"""
        if user.id not in self.users:
            return False
        
        old_user = self.users[user.id]
        
        # Update username mapping if changed
        if old_user.username != user.username:
            del self.users_by_username[old_user.username]
            self.users_by_username[user.username] = user.id
        
        # Update email mapping if changed
        if old_user.email != user.email:
            del self.users_by_email[old_user.email]
            self.users_by_email[user.email] = user.id
        
        # Update user
        self.users[user.id] = user
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user from system"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        # Remove from mappings
        del self.users_by_username[user.username]
        del self.users_by_email[user.email]
        del self.users[user_id]
        
        # Revoke all user sessions
        self.jwt_handler.revoke_user_sessions(user_id)
        
        return True
    
    async def authenticate(self, 
                          username_or_email: str, 
                          password: str,
                          ip_address: Optional[str] = None,
                          user_agent: Optional[str] = None) -> Tuple[str, str, User]:
        """Authenticate user and return tokens"""
        
        # Check rate limiting
        if not self._check_rate_limit(username_or_email, ip_address):
            raise AuthenticationError("Too many login attempts. Please try again later.")
        
        # Find user
        user = self.get_user_by_username(username_or_email)
        if not user:
            user = self.get_user_by_email(username_or_email)
        
        if not user:
            self._record_failed_attempt(username_or_email, ip_address)
            raise AuthenticationError("Invalid credentials")
        
        # Check if account is locked
        if user.is_locked():
            raise AuthenticationError("Account is temporarily locked")
        
        # Check if account is active
        if not user.is_active:
            raise AuthenticationError("Account is disabled")
        
        # Verify password
        if not user.verify_password(password):
            user.record_failed_login()
            self.update_user(user)
            self._record_failed_attempt(username_or_email, ip_address)
            raise AuthenticationError("Invalid credentials")
        
        # Successful authentication
        user.record_successful_login()
        self.update_user(user)
        self._clear_failed_attempts(username_or_email, ip_address)
        
        # Generate tokens
        access_token, refresh_token, session = self.jwt_handler.generate_tokens(
            user, ip_address, user_agent
        )
        
        return access_token, refresh_token, user
    
    def _check_rate_limit(self, identifier: str, ip_address: Optional[str] = None) -> bool:
        """Check if login attempts are within rate limit"""
        
        now = datetime.now()
        cutoff = now - timedelta(minutes=self.lockout_duration_minutes)
        
        # Check attempts for username/email
        user_attempts = self.login_attempts.get(identifier, [])
        user_attempts = [attempt for attempt in user_attempts if attempt > cutoff]
        
        if len(user_attempts) >= self.max_login_attempts:
            return False
        
        # Check attempts for IP address if provided
        if ip_address:
            ip_attempts = self.login_attempts.get(f"ip:{ip_address}", [])
            ip_attempts = [attempt for attempt in ip_attempts if attempt > cutoff]
            
            if len(ip_attempts) >= self.max_login_attempts * 3:  # More lenient for IP
                return False
        
        return True
    
    def _record_failed_attempt(self, identifier: str, ip_address: Optional[str] = None) -> None:
        """Record failed login attempt"""
        
        now = datetime.now()
        
        # Record for username/email
        if identifier not in self.login_attempts:
            self.login_attempts[identifier] = []
        self.login_attempts[identifier].append(now)
        
        # Record for IP address
        if ip_address:
            ip_key = f"ip:{ip_address}"
            if ip_key not in self.login_attempts:
                self.login_attempts[ip_key] = []
            self.login_attempts[ip_key].append(now)
    
    def _clear_failed_attempts(self, identifier: str, ip_address: Optional[str] = None) -> None:
        """Clear failed login attempts after successful login"""
        
        if identifier in self.login_attempts:
            del self.login_attempts[identifier]
        
        if ip_address:
            ip_key = f"ip:{ip_address}"
            if ip_key in self.login_attempts:
                del self.login_attempts[ip_key]
    
    def logout(self, token: str) -> bool:
        """Logout user by revoking token"""
        return self.jwt_handler.revoke_token(token)
    
    def logout_all_sessions(self, user_id: str) -> int:
        """Logout user from all sessions"""
        return self.jwt_handler.revoke_user_sessions(user_id)
    
    def refresh_token(self, refresh_token: str) -> Optional[Tuple[str, UserSession]]:
        """Refresh access token"""
        return self.jwt_handler.refresh_access_token(refresh_token)
    
    def verify_token(self, token: str) -> Optional[User]:
        """Verify token and return user"""
        
        payload = self.jwt_handler.verify_token(token, 'access')
        if not payload:
            return None
        
        user_id = payload.get('user_id')
        if not user_id:
            return None
        
        user = self.get_user(user_id)
        if not user or not user.is_active:
            return None
        
        return user
    
    def create_user(self, 
                   username: str,
                   email: str,
                   password: str,
                   full_name: str = "",
                   role: UserRole = UserRole.VIEWER) -> Optional[User]:
        """Create new user"""
        
        # Validate input
        if not username or not email or not password:
            return None
        
        # Check if user already exists
        if self.get_user_by_username(username) or self.get_user_by_email(email):
            return None
        
        # Create user
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            role=role,
            is_active=True,
            is_verified=False
        )
        user.set_password(password)
        
        # Add to system
        if self.add_user(user):
            return user
        
        return None
    
    def change_password(self, user_id: str, 
                       old_password: str, 
                       new_password: str) -> bool:
        """Change user password"""
        
        user = self.get_user(user_id)
        if not user:
            return False
        
        # Verify old password
        if not user.verify_password(old_password):
            return False
        
        # Set new password
        user.set_password(new_password)
        user.require_password_change = False
        
        # Update user
        self.update_user(user)
        
        # Revoke all existing sessions to force re-login
        self.jwt_handler.revoke_user_sessions(user_id)
        
        return True
    
    def reset_password(self, user_id: str, new_password: str) -> bool:
        """Reset user password (admin function)"""
        
        user = self.get_user(user_id)
        if not user:
            return False
        
        # Set new password
        user.set_password(new_password)
        user.require_password_change = True
        
        # Update user
        self.update_user(user)
        
        # Revoke all existing sessions
        self.jwt_handler.revoke_user_sessions(user_id)
        
        return True
    
    def lock_user(self, user_id: str, duration_minutes: int = 30) -> bool:
        """Lock user account"""
        
        user = self.get_user(user_id)
        if not user:
            return False
        
        user.lock_account(duration_minutes)
        self.update_user(user)
        
        # Revoke all sessions
        self.jwt_handler.revoke_user_sessions(user_id)
        
        return True
    
    def unlock_user(self, user_id: str) -> bool:
        """Unlock user account"""
        
        user = self.get_user(user_id)
        if not user:
            return False
        
        user.unlock_account()
        self.update_user(user)
        
        return True
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user account"""
        
        user = self.get_user(user_id)
        if not user:
            return False
        
        user.is_active = False
        self.update_user(user)
        
        # Revoke all sessions
        self.jwt_handler.revoke_user_sessions(user_id)
        
        return True
    
    def activate_user(self, user_id: str) -> bool:
        """Activate user account"""
        
        user = self.get_user(user_id)
        if not user:
            return False
        
        user.is_active = True
        self.update_user(user)
        
        return True
    
    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """Get active sessions for user"""
        return self.jwt_handler.get_active_sessions(user_id)
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        return self.jwt_handler.cleanup_expired_sessions()
    
    def get_all_users(self, include_inactive: bool = False) -> List[User]:
        """Get all users"""
        users = list(self.users.values())
        
        if not include_inactive:
            users = [user for user in users if user.is_active]
        
        return users
    
    def get_users_by_role(self, role: UserRole) -> List[User]:
        """Get users by role"""
        return [user for user in self.users.values() if user.role == role and user.is_active]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        
        total_users = len(self.users)
        active_users = len([u for u in self.users.values() if u.is_active])
        locked_users = len([u for u in self.users.values() if u.is_locked()])
        
        # Count users by role
        role_counts = {}
        for role in UserRole:
            role_counts[role.value] = len(self.get_users_by_role(role))
        
        # Get JWT stats
        jwt_stats = self.jwt_handler.get_stats()
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'locked_users': locked_users,
            'role_distribution': role_counts,
            'jwt_stats': jwt_stats,
            'failed_attempts_tracked': len(self.login_attempts)
        }


# Global auth manager instance
auth_manager = AuthManager()