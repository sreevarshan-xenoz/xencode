#!/usr/bin/env python3
"""
JWT Handler

Handles JWT token generation, validation, and management for authentication.
Provides secure token-based authentication with configurable expiration and refresh.
"""

import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None

from xencode.models.user import User, UserSession, UserRole


class JWTHandler:
    """Handles JWT token operations"""
    
    def __init__(self, 
                 secret_key: Optional[str] = None,
                 algorithm: str = "HS256",
                 access_token_expire_minutes: int = 60,
                 refresh_token_expire_days: int = 30):
        
        if not JWT_AVAILABLE:
            raise ImportError(
                "PyJWT is required for JWT handling. "
                "Install with: pip install PyJWT"
            )
        
        self.secret_key = secret_key or self._generate_secret_key()
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
        # Store active sessions
        self.active_sessions: Dict[str, UserSession] = {}
        
        # Blacklisted tokens (for logout)
        self.blacklisted_tokens: set = set()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        return secrets.token_urlsafe(64)
    
    def generate_tokens(self, user: User, 
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None) -> Tuple[str, str, UserSession]:
        """Generate access and refresh tokens for user"""
        
        now = datetime.utcnow()
        
        # Create session
        session = UserSession(
            user_id=user.id,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=now,
            expires_at=now + timedelta(days=self.refresh_token_expire_days),
            last_activity=now
        )
        
        # Access token payload
        access_payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role.value,
            'session_id': session.id,
            'type': 'access',
            'iat': now,
            'exp': now + timedelta(minutes=self.access_token_expire_minutes)
        }
        
        # Refresh token payload
        refresh_payload = {
            'user_id': user.id,
            'session_id': session.id,
            'type': 'refresh',
            'iat': now,
            'exp': now + timedelta(days=self.refresh_token_expire_days)
        }
        
        # Generate tokens
        access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
        
        # Store session with refresh token
        session.token = refresh_token
        self.active_sessions[session.id] = session
        
        return access_token, refresh_token, session
    
    def verify_token(self, token: str, token_type: str = 'access') -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                return None
            
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Verify token type
            if payload.get('type') != token_type:
                return None
            
            # Check if session is still active (for refresh tokens)
            if token_type == 'refresh':
                session_id = payload.get('session_id')
                session = self.active_sessions.get(session_id)
                if not session or not session.is_valid():
                    return None
                
                # Update last activity
                session.last_activity = datetime.now()
            
            return payload
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Tuple[str, UserSession]]:
        """Generate new access token using refresh token"""
        
        # Verify refresh token
        payload = self.verify_token(refresh_token, 'refresh')
        if not payload:
            return None
        
        session_id = payload.get('session_id')
        session = self.active_sessions.get(session_id)
        if not session or not session.is_valid():
            return None
        
        # Generate new access token
        now = datetime.utcnow()
        access_payload = {
            'user_id': payload['user_id'],
            'username': payload.get('username', ''),
            'role': payload.get('role', UserRole.VIEWER.value),
            'session_id': session_id,
            'type': 'access',
            'iat': now,
            'exp': now + timedelta(minutes=self.access_token_expire_minutes)
        }
        
        access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
        
        # Update session activity
        session.last_activity = datetime.now()
        
        return access_token, session
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token (add to blacklist)"""
        try:
            # Decode to get session info
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
            
            # Add to blacklist
            self.blacklisted_tokens.add(token)
            
            # If it's a refresh token, remove the session
            if payload.get('type') == 'refresh':
                session_id = payload.get('session_id')
                if session_id in self.active_sessions:
                    self.active_sessions[session_id].is_active = False
            
            return True
            
        except Exception:
            return False
    
    def revoke_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user"""
        revoked_count = 0
        
        # Find and revoke all user sessions
        for session in list(self.active_sessions.values()):
            if session.user_id == user_id and session.is_active:
                session.is_active = False
                self.blacklisted_tokens.add(session.token)
                revoked_count += 1
        
        return revoked_count
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and tokens"""
        cleaned_count = 0
        
        # Remove expired sessions
        expired_sessions = []
        for session_id, session in self.active_sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)
                self.blacklisted_tokens.add(session.token)
                cleaned_count += 1
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        return cleaned_count
    
    def get_active_sessions(self, user_id: Optional[str] = None) -> List[UserSession]:
        """Get active sessions, optionally filtered by user"""
        sessions = []
        
        for session in self.active_sessions.values():
            if session.is_valid():
                if user_id is None or session.user_id == user_id:
                    sessions.append(session)
        
        return sessions
    
    def get_session_info(self, session_id: str) -> Optional[UserSession]:
        """Get session information"""
        return self.active_sessions.get(session_id)
    
    def validate_session(self, session_id: str) -> bool:
        """Validate if session is active and valid"""
        session = self.active_sessions.get(session_id)
        return session is not None and session.is_valid()
    
    def extend_session(self, session_id: str, extend_hours: int = 24) -> bool:
        """Extend session expiration"""
        session = self.active_sessions.get(session_id)
        if session and session.is_valid():
            session.refresh(extend_hours)
            return True
        return False
    
    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """Get token information without verification (for debugging)"""
        try:
            # Decode without verification to get payload info
            payload = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
            return payload
        except Exception:
            return None
    
    def create_service_token(self, service_name: str, 
                           permissions: List[str],
                           expire_days: int = 365) -> str:
        """Create a service token for system-to-system authentication"""
        
        now = datetime.utcnow()
        payload = {
            'service_name': service_name,
            'permissions': permissions,
            'type': 'service',
            'iat': now,
            'exp': now + timedelta(days=expire_days)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_service_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify service token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get('type') != 'service':
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get JWT handler statistics"""
        active_sessions = len([s for s in self.active_sessions.values() if s.is_valid()])
        expired_sessions = len([s for s in self.active_sessions.values() if s.is_expired()])
        
        return {
            'total_sessions': len(self.active_sessions),
            'active_sessions': active_sessions,
            'expired_sessions': expired_sessions,
            'blacklisted_tokens': len(self.blacklisted_tokens),
            'algorithm': self.algorithm,
            'access_token_expire_minutes': self.access_token_expire_minutes,
            'refresh_token_expire_days': self.refresh_token_expire_days
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export configuration (without secret key)"""
        return {
            'algorithm': self.algorithm,
            'access_token_expire_minutes': self.access_token_expire_minutes,
            'refresh_token_expire_days': self.refresh_token_expire_days,
            'has_secret_key': bool(self.secret_key)
        }


# Global JWT handler instance
jwt_handler = JWTHandler()