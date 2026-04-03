#!/usr/bin/env python3
"""
Qwen OAuth2 Device Flow Authentication

Handles OAuth2 device flow authentication for Qwen AI services,
including token caching, refreshing, and API call integration.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import aiohttp
import base64


@dataclass
class QwenCredentials:
    """Qwen API credentials data structure"""
    access_token: str
    refresh_token: Optional[str]
    expires_in: int  # seconds
    token_type: str
    created_at: float  # timestamp for expiry tracking


class QwenAuthError(Exception):
    """Exception raised for Qwen authentication errors"""
    pass


class QwenAuthManager:
    """Manages Qwen OAuth2 device flow authentication"""

    # Qwen's exact constants from their code
    CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
    SCOPE = "openid profile email model.completion"
    DEVICE_AUTH_URL = "https://chat.qwen.ai/api/v1/oauth2/device/code"
    TOKEN_URL = "https://chat.qwen.ai/api/v1/oauth2/token"
    
    # Local storage for credentials
    CREDS_FILE = Path.home() / ".xencode_qwen_creds.json"
    REQUEST_TIMEOUT_SECONDS = 20
    RETRY_ATTEMPTS = 3

    def __init__(self):
        self.credentials: Optional[QwenCredentials] = None
        self._auth_lock = asyncio.Lock()

    async def get_or_authenticate(self, force_reauth: bool = False) -> QwenCredentials:
        """
        Get existing credentials or authenticate via device flow
        
        Returns:
            QwenCredentials: Valid credentials
        """
        async with self._auth_lock:
            if not force_reauth:
                # Check for cached credentials
                cached_creds = self._load_cached_credentials()
                if cached_creds:
                    if self._is_token_valid(cached_creds):
                        self.credentials = cached_creds
                        return cached_creds

                    # If cached credentials are expired, prefer refresh before full re-auth.
                    if cached_creds.refresh_token:
                        try:
                            refreshed = await self.refresh_access_token(cached_creds.refresh_token)
                            return refreshed
                        except Exception:
                            # Fall through to device flow when refresh fails.
                            pass

            # Perform fresh authentication
            creds = await self._authenticate_via_device_flow()
            self._save_credentials(creds)
            self.credentials = creds
            return creds

    async def _post_json_with_retry(
        self,
        session: aiohttp.ClientSession,
        url: str,
        *,
        data: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
        retry_attempts: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Any], str]:
        """POST request helper with timeout and retry for transient failures."""
        attempts = retry_attempts or self.RETRY_ATTEMPTS
        timeout = timeout_seconds or self.REQUEST_TIMEOUT_SECONDS
        last_error = ""

        for attempt in range(1, attempts + 1):
            try:
                request_timeout = aiohttp.ClientTimeout(total=timeout)
                async with session.post(
                    url,
                    data=data,
                    json=json_payload,
                    headers=headers,
                    timeout=request_timeout,
                ) as response:
                    text = await response.text()
                    parsed_json: Dict[str, Any] = {}
                    if text:
                        try:
                            maybe_json = json.loads(text)
                            if isinstance(maybe_json, dict):
                                parsed_json = maybe_json
                        except json.JSONDecodeError:
                            parsed_json = {}
                    return response.status, parsed_json, text
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = str(e)
                if attempt < attempts:
                    await asyncio.sleep(min(2 * attempt, 5))

        raise QwenAuthError(f"Network request failed after {attempts} attempts: {last_error}")

    def has_valid_cached_credentials(self) -> bool:
        """Return whether a valid cached Qwen credential is available."""
        cached_creds = self._load_cached_credentials()
        if not cached_creds:
            return False
        return self._is_token_valid(cached_creds)

    def _load_cached_credentials(self) -> Optional[QwenCredentials]:
        """Load cached credentials from file"""
        try:
            if not self.CREDS_FILE.exists():
                return None
            
            with open(self.CREDS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return QwenCredentials(
                access_token=data['access_token'],
                refresh_token=data.get('refresh_token'),
                expires_in=data['expires_in'],
                token_type=data['token_type'],
                created_at=data.get('created_at', time.time())
            )
        except (json.JSONDecodeError, KeyError, IOError):
            return None

    def _save_credentials(self, creds: QwenCredentials) -> None:
        """Save credentials to file"""
        try:
            data = {
                'access_token': creds.access_token,
                'refresh_token': creds.refresh_token,
                'expires_in': creds.expires_in,
                'token_type': creds.token_type,
                'created_at': creds.created_at
            }
            
            # Ensure parent directory exists
            self.CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.CREDS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            # Best-effort hardening: owner read/write only on POSIX systems.
            if os.name == "posix":
                os.chmod(self.CREDS_FILE, 0o600)
                
        except IOError as e:
            raise QwenAuthError(f"Failed to save credentials: {e}")

    def _is_token_valid(self, creds: QwenCredentials) -> bool:
        """
        Check if credentials are still valid (not expired)
        
        Args:
            creds: Credentials to check
            
        Returns:
            bool: True if valid, False if expired
        """
        # Check if token has expired (with 5-minute buffer)
        elapsed = time.time() - creds.created_at
        return elapsed < (creds.expires_in - 300)  # 5 minutes buffer

    async def _authenticate_via_device_flow(self) -> QwenCredentials:
        """
        Authenticate using OAuth2 device flow
        
        Returns:
            QwenCredentials: Fresh credentials
        """
        timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Step 1: Request device authorization (with PKCE)
            pkce_verifier, pkce_challenge = self._generate_pkce_pair()
            
            device_auth_data = {
                'client_id': self.CLIENT_ID,
                'scope': self.SCOPE,
                'code_challenge': pkce_challenge,
                'code_challenge_method': 'S256'
            }
            
            status_code, device_auth_resp, raw_text = await self._post_json_with_retry(
                session,
                self.DEVICE_AUTH_URL,
                data=device_auth_data,
            )
            if status_code != 200:
                raise QwenAuthError(
                    f"Device authorization request failed: {status_code} {raw_text[:200]}"
                )

            if not device_auth_resp:
                raise QwenAuthError("Device authorization returned an empty response")
            
            # Step 2: Show user the verification details
            verification_uri = device_auth_resp.get('verification_uri_complete') or device_auth_resp['verification_uri']
            user_code = device_auth_resp['user_code']
            
            print("\n🚀 Qwen AI Authentication Required")
            print("=" * 40)
            print(f"Visit: {verification_uri}")
            print(f"Enter code: {user_code}")
            print("=" * 40)
            print("Please complete the authentication in your browser...")
            
            # Step 3: Poll for token (with exponential backoff)
            interval = device_auth_resp.get('interval', 5)  # seconds
            expires_in = device_auth_resp.get('expires_in', 300)  # seconds
            
            start_time = time.time()
            while time.time() - start_time < expires_in:
                await asyncio.sleep(interval)
                
                token_data = {
                    'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
                    'device_code': device_auth_resp['device_code'],
                    'client_id': self.CLIENT_ID,
                    'code_verifier': pkce_verifier
                }

                try:
                    token_status, token_json, token_raw = await self._post_json_with_retry(
                        session,
                        self.TOKEN_URL,
                        data=token_data,
                        retry_attempts=1,
                    )
                except QwenAuthError:
                    # transient network issue during polling - continue trying until device code expiry
                    continue

                if token_status == 200:
                    if not token_json:
                        raise QwenAuthError("Token endpoint returned an invalid response")
                        # Success - got tokens
                    return QwenCredentials(
                        access_token=token_json['access_token'],
                        refresh_token=token_json.get('refresh_token'),
                        expires_in=token_json['expires_in'],
                        token_type=token_json['token_type'],
                        created_at=time.time()
                    )
                elif token_json.get('error') == 'authorization_pending':
                    # Still waiting for user - continue polling
                    continue
                elif token_json.get('error') == 'slow_down':
                    # Server requested slower polling - increase interval by 5 seconds
                    interval += 5
                    continue
                elif token_json.get('error') == 'expired_token':
                    raise QwenAuthError("Authentication code expired. Please try again.")
                else:
                    error_desc = token_json.get('error_description') or token_raw[:200] or 'Unknown error'
                    raise QwenAuthError(f"Token request failed: {error_desc}")
            
            raise QwenAuthError("Authentication timed out. Please try again.")

    def _generate_pkce_pair(self) -> tuple[str, str]:
        """
        Generate PKCE code verifier and challenge pair
        
        Returns:
            tuple: (verifier, challenge)
        """
        import hashlib
        import secrets
        
        # Generate 32-byte random string
        verifier_bytes = secrets.token_bytes(32)
        code_verifier = base64.urlsafe_b64encode(verifier_bytes).decode('utf-8').rstrip('=')
        
        # Generate SHA256 hash of verifier
        hashed = hashlib.sha256(code_verifier.encode('utf-8')).digest()
        code_challenge = base64.urlsafe_b64encode(hashed).decode('utf-8').rstrip('=')
        
        return code_verifier, code_challenge

    async def refresh_access_token(self, refresh_token: str) -> QwenCredentials:
        """
        Refresh access token using refresh token
        
        Args:
            refresh_token: The refresh token to use
            
        Returns:
            QwenCredentials: New credentials with refreshed access token
        """
        timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            refresh_data = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': self.CLIENT_ID
            }
            status_code, token_json, raw_text = await self._post_json_with_retry(
                session,
                self.TOKEN_URL,
                data=refresh_data,
            )
            if status_code != 200:
                raise QwenAuthError(f"Token refresh failed: {status_code} {raw_text[:200]}")

            if not token_json:
                raise QwenAuthError("Token refresh returned an invalid response")
                
            new_creds = QwenCredentials(
                access_token=token_json['access_token'],
                refresh_token=token_json.get('refresh_token', refresh_token),  # May return same refresh token
                expires_in=token_json['expires_in'],
                token_type=token_json['token_type'],
                created_at=time.time()
            )

            # Save the refreshed credentials
            self._save_credentials(new_creds)
            self.credentials = new_creds

            return new_creds

    async def call_qwen_completion(self, prompt: str, model: str = "qwen-max-coder-7b-instruct") -> str:
        """
        Call Qwen completion API with authenticated credentials
        
        Args:
            prompt: The prompt to send
            model: The model to use (default: qwen-max-coder-7b-instruct)
            
        Returns:
            str: The completion response
        """
        if not self.credentials:
            self.credentials = await self.get_or_authenticate()
        
        # Check if token is still valid, refresh if needed
        if not self._is_token_valid(self.credentials):
            if self.credentials.refresh_token:
                await self.refresh_access_token(self.credentials.refresh_token)
            else:
                # Token expired and no refresh token - re-authenticate
                await self.get_or_authenticate()
        
        headers = {
            'Authorization': f'Bearer {self.credentials.access_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}]
        }
        
        timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post('https://chat.qwen.ai/v1/chat/completions', 
                                  json=payload, headers=headers, timeout=timeout) as response:
                
                if response.status == 401:  # Unauthorized - token expired
                    # Try to refresh and retry once
                    if self.credentials.refresh_token:
                        await self.refresh_access_token(self.credentials.refresh_token)
                        
                        # Retry the request with new token
                        headers['Authorization'] = f'Bearer {self.credentials.access_token}'
                        async with session.post('https://chat.qwen.ai/v1/chat/completions', 
                                              json=payload, headers=headers, timeout=timeout) as retry_response:
                            if retry_response.status != 200:
                                raise QwenAuthError(f"Completion API failed after refresh: {retry_response.status}")
                            
                            result = await retry_response.json()
                            return result['choices'][0]['message']['content']
                    else:
                        raise QwenAuthError("Access token expired and no refresh token available")
                
                elif response.status != 200:
                    raise QwenAuthError(f"Completion API failed: {response.status}")
                
                result = await response.json()
                return result['choices'][0]['message']['content']

    def clear_credentials(self) -> bool:
        """
        Clear cached credentials
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.CREDS_FILE.exists():
                self.CREDS_FILE.unlink()
            self.credentials = None
            return True
        except IOError:
            return False


# Global instance for convenience
qwen_auth_manager = QwenAuthManager()
