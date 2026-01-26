"""
Authentication utilities for Xencode
Provides authentication functionality for API endpoints
"""
import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from dataclasses import dataclass
import jwt
from jwt import PyJWT


@dataclass
class AuthToken:
    """Data class representing an authentication token"""
    token: str
    expires_at: datetime
    user_id: str
    scopes: list


class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    pass


class APIKeyAuthenticator:
    """Handles API key-based authentication"""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.hashed_keys: Dict[str, str] = {}  # hashed_key -> user_id mapping
    
    def create_api_key(self, user_id: str, description: str = "", scopes: list = None) -> str:
        """
        Create a new API key for a user.
        
        Args:
            user_id: User identifier
            description: Description for the API key
            scopes: List of permissions/scopes for the API key
            
        Returns:
            Generated API key
        """
        if scopes is None:
            scopes = ["read", "write"]
        
        # Generate a secure random API key
        api_key = secrets.token_urlsafe(32)
        
        # Store the API key info
        self.api_keys[api_key] = {
            'user_id': user_id,
            'description': description,
            'scopes': scopes,
            'created_at': datetime.now(),
            'last_used': None
        }
        
        # Store a hash of the key for security
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self.hashed_keys[key_hash] = user_id
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Verify an API key and return user information.
        
        Args:
            api_key: API key to verify
            
        Returns:
            User information if valid, None otherwise
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.hashed_keys:
            user_id = self.hashed_keys[key_hash]
            if api_key in self.api_keys:
                # Update last used timestamp
                self.api_keys[api_key]['last_used'] = datetime.now()
                return self.api_keys[api_key]
        
        return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked, False if not found
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.hashed_keys:
            del self.hashed_keys[key_hash]
            if api_key in self.api_keys:
                del self.api_keys[api_key]
            return True
        
        return False
    
    def list_user_api_keys(self, user_id: str) -> list:
        """
        List all API keys for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of API key information
        """
        user_keys = []
        for key, info in self.api_keys.items():
            if info['user_id'] == user_id:
                user_keys.append({
                    'key_prefix': key[:8],  # Only show first 8 chars for security
                    'description': info['description'],
                    'scopes': info['scopes'],
                    'created_at': info['created_at'],
                    'last_used': info['last_used']
                })
        return user_keys


class JWTAuthenticator:
    """Handles JWT-based authentication"""
    
    def __init__(self, secret_key: str = None):
        """
        Initialize the JWT authenticator.
        
        Args:
            secret_key: Secret key for signing JWT tokens (generates random if not provided)
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = 'HS256'
    
    def generate_token(self, user_id: str, expires_in: int = 3600, scopes: list = None) -> str:
        """
        Generate a JWT token for a user.
        
        Args:
            user_id: User identifier
            expires_in: Token expiration time in seconds (default 1 hour)
            scopes: List of permissions/scopes for the token
            
        Returns:
            Generated JWT token
        """
        if scopes is None:
            scopes = ["read", "write"]
        
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow(),
            'scopes': scopes
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify a JWT token and return user information.
        
        Args:
            token: JWT token to verify
            
        Returns:
            User information if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired (this is handled by jwt.decode automatically)
            return {
                'user_id': payload['user_id'],
                'scopes': payload['scopes'],
                'exp': payload['exp'],
                'iat': payload['iat']
            }
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_token(self, token: str, new_expires_in: int = 3600) -> Optional[str]:
        """
        Refresh a JWT token with a new expiration time.
        
        Args:
            token: Existing JWT token
            new_expires_in: New expiration time in seconds
            
        Returns:
            New JWT token if valid, None otherwise
        """
        user_info = self.verify_token(token)
        if user_info:
            return self.generate_token(
                user_info['user_id'],
                expires_in=new_expires_in,
                scopes=user_info['scopes']
            )
        return None


class HMACAuthenticator:
    """Handles HMAC-based authentication for API requests"""
    
    def __init__(self, secret_key: str = None):
        """
        Initialize the HMAC authenticator.
        
        Args:
            secret_key: Secret key for generating HMAC signatures (generates random if not provided)
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
    
    def generate_signature(self, data: str, timestamp: int = None) -> str:
        """
        Generate an HMAC signature for data.
        
        Args:
            data: Data to sign
            timestamp: Timestamp to include in signature (uses current time if not provided)
            
        Returns:
            HMAC signature
        """
        if timestamp is None:
            timestamp = int(time.time())
        
        # Create a string to sign: timestamp + data
        string_to_sign = f"{timestamp}:{data}"
        
        # Generate HMAC signature
        signature = hmac.new(
            self.secret_key.encode(),
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{timestamp}.{signature}"
    
    def verify_signature(self, data: str, signature: str, max_age: int = 300) -> bool:
        """
        Verify an HMAC signature.
        
        Args:
            data: Original data that was signed
            signature: Signature to verify
            max_age: Maximum age of the signature in seconds (default 5 minutes)
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            timestamp_str, sig = signature.split('.', 1)
            timestamp = int(timestamp_str)
        except (ValueError, TypeError):
            return False
        
        # Check if signature is too old
        if time.time() - timestamp > max_age:
            return False
        
        # Generate expected signature
        string_to_sign = f"{timestamp}:{data}"
        expected_sig = hmac.new(
            self.secret_key.encode(),
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures securely
        return hmac.compare_digest(sig, expected_sig)


class Authenticator:
    """Main authenticator class that combines different authentication methods"""
    
    def __init__(self, secret_key: str = None):
        """
        Initialize the main authenticator.
        
        Args:
            secret_key: Secret key for JWT and HMAC (generates random if not provided)
        """
        self.api_key_auth = APIKeyAuthenticator()
        self.jwt_auth = JWTAuthenticator(secret_key)
        self.hmac_auth = HMACAuthenticator(secret_key)
    
    def authenticate_request(self, headers: Dict[str, str], body: str = "") -> Optional[Dict[str, Any]]:
        """
        Authenticate a request using various methods.
        
        Args:
            headers: Request headers
            body: Request body (for HMAC verification)
            
        Returns:
            User information if authenticated, None otherwise
        """
        # Check for API key in Authorization header
        auth_header = headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]  # Remove 'Bearer ' prefix
            return self.jwt_auth.verify_token(token)
        elif auth_header.startswith('API-Key '):
            api_key = auth_header[8:]  # Remove 'API-Key ' prefix
            return self.api_key_auth.verify_api_key(api_key)
        
        # Check for API key in X-API-Key header
        api_key = headers.get('X-API-Key')
        if api_key:
            return self.api_key_auth.verify_api_key(api_key)
        
        # Check for HMAC signature
        signature = headers.get('X-Signature')
        timestamp = headers.get('X-Timestamp')
        if signature and timestamp and body:
            # Reconstruct the data for verification
            string_to_sign = f"{timestamp}:{body}"
            expected_sig = hmac.new(
                self.hmac_auth.secret_key.encode(),
                string_to_sign.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if hmac.compare_digest(signature, expected_sig):
                # For HMAC auth, we might want to associate with a user
                # In a real implementation, you might map API keys to users differently
                return {'user_id': 'hmac_authenticated', 'scopes': ['read', 'write']}
        
        return None
    
    def create_user_session(self, user_id: str, scopes: list = None) -> str:
        """
        Create a session token for a user.
        
        Args:
            user_id: User identifier
            scopes: List of permissions for the session
            
        Returns:
            Session token
        """
        return self.jwt_auth.generate_token(user_id, expires_in=3600, scopes=scopes or ["read", "write"])
    
    def create_api_key(self, user_id: str, description: str = "", scopes: list = None) -> str:
        """
        Create an API key for a user.
        
        Args:
            user_id: User identifier
            description: Description for the API key
            scopes: List of permissions for the API key
            
        Returns:
            Generated API key
        """
        return self.api_key_auth.create_api_key(user_id, description, scopes)


# Global authenticator instance
authenticator = Authenticator()


def get_authenticator() -> Authenticator:
    """Get the global authenticator instance"""
    return authenticator


def authenticate_request(headers: Dict[str, str], body: str = "") -> Optional[Dict[str, Any]]:
    """
    Convenience function to authenticate a request.
    
    Args:
        headers: Request headers
        body: Request body
        
    Returns:
        User information if authenticated, None otherwise
    """
    return authenticator.authenticate_request(headers, body)


def create_user_session(user_id: str, scopes: list = None) -> str:
    """
    Convenience function to create a user session.
    
    Args:
        user_id: User identifier
        scopes: List of permissions for the session
        
    Returns:
        Session token
    """
    return authenticator.create_user_session(user_id, scopes)


def create_api_key(user_id: str, description: str = "", scopes: list = None) -> str:
    """
    Convenience function to create an API key.
    
    Args:
        user_id: User identifier
        description: Description for the API key
        scopes: List of permissions for the API key
        
    Returns:
        Generated API key
    """
    return authenticator.create_api_key(user_id, description, scopes)