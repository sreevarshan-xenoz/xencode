#!/usr/bin/env python3
"""
API Authentication

Centralized authentication utilities for FastAPI endpoints.
Provides JWT verification, user extraction, and authorization dependencies.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


class AuthenticationError(Exception):
    """Authentication failed"""
    pass


class AuthorizationError(Exception):
    """Authorization failed"""
    pass


async def verify_jwt_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    required: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Verify JWT token for authenticated endpoints
    
    Args:
        credentials: HTTP Bearer credentials from request
        required: Whether authentication is required (default True)
        
    Returns:
        Decoded JWT payload if valid, None if not authenticated and not required
        
    Raises:
        HTTPException: If authentication fails and is required
    """
    if credentials is None:
        if required:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None
    
    token = credentials.credentials
    
    if not token:
        if required:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None
    
    try:
        # Import JWT handler
        from xencode.auth.jwt_handler import JWTHandler
        from xencode.auth.vault import get_vault
        
        # Get secret key from vault or environment
        vault = get_vault()
        secret_key = vault.get_secret("jwt_secret_key")
        
        if not secret_key:
            # Fallback to environment variable
            import os
            secret_key = os.getenv("XENCODE_JWT_SECRET_KEY")
        
        if not secret_key:
            logger.warning("No JWT secret key configured - using default (INSECURE)")
            # Use a default for development only
            secret_key = "dev-secret-key-change-in-production"
        
        # Create JWT handler with the secret
        jwt_handler = JWTHandler(secret_key=secret_key)
        
        # Verify the token
        payload = jwt_handler.verify_token(token, token_type='access')
        
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload
        
    except HTTPException:
        raise
    except ImportError as e:
        logger.error(f"JWT handler import failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    payload: Dict[str, Any] = Depends(verify_jwt_token)
) -> Dict[str, Any]:
    """
    Get current user from JWT payload
    
    Returns user information extracted from the JWT token
    """
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    
    return {
        'user_id': payload.get('user_id'),
        'username': payload.get('username'),
        'role': payload.get('role'),
        'session_id': payload.get('session_id'),
    }


async def require_role(
    required_role: str,
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Require specific user role for endpoint access
    
    Args:
        required_role: The role required to access the endpoint
        user: Current user from get_current_user
        
    Raises:
        HTTPException: If user doesn't have required role
    """
    user_role = user.get('role', '')
    
    # Role hierarchy (higher roles include lower roles)
    role_hierarchy = {
        'admin': ['admin', 'developer', 'viewer'],
        'developer': ['developer', 'viewer'],
        'viewer': ['viewer'],
    }
    
    allowed_roles = role_hierarchy.get(required_role, [required_role])
    
    if user_role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Required role: {required_role}",
        )
    
    return user


# Optional authentication - doesn't fail if no token provided
async def verify_token_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Verify JWT token if provided, but don't require it
    
    Returns None if no token provided, payload if valid
    """
    return await verify_jwt_token(credentials, required=False)


# Helper for collaborative features that need auth
async def verify_collaboration_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Verify authentication for collaborative features
    
    Collaborative features require authentication, but we accept
    any valid token for now (can be enhanced with session validation)
    """
    return await verify_jwt_token(credentials, required=True)
