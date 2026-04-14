#!/usr/bin/env python3
"""
Email Verification Flow

Generates time-limited email verification tokens, sends verification
emails, and validates tokens on confirmation.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional

from xencode.models.user import User


class EmailVerificationError(Exception):
    pass


class EmailVerificationManager:
    """Manages email verification tokens and email sending."""

    TOKEN_BYTES = 32
    TOKEN_EXPIRE_HOURS = 24

    def __init__(self):
        self._token_store: dict[str, dict] = {}

    def generate_token(self, user: User) -> str:
        """Generate a verification token for a user and store it."""
        token = secrets.token_urlsafe(self.TOKEN_BYTES)
        self._token_store[token] = {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=self.TOKEN_EXPIRE_HOURS),
            "used": False,
        }
        return token

    def verify_token(self, token: str) -> Optional[str]:
        """
        Verify a token and return the user_id if valid.
        Returns None if token is invalid, expired, or already used.
        """
        record = self._token_store.get(token)
        if not record:
            return None
        if record["used"]:
            return None
        if datetime.now() > record["expires_at"]:
            return None
        record["used"] = True
        return record["user_id"]

    def send_verification_email(self, user: User, base_url: str = "https://xencode.local/verify") -> str:
        """
        Generate a verification token and return the verification URL.
        In production this would call an email service — currently returns
        the URL so it can be logged or sent via a configured channel.
        """
        token = self.generate_token(user)
        verify_url = f"{base_url.rstrip('/')}/?token={token}"
        return verify_url

    def resend_verification(self, user: User, base_url: str = "https://xencode.local/verify") -> Optional[str]:
        """Regenerate a fresh token for an unverified user and return the URL."""
        if user.is_verified:
            return None
        return self.send_verification_email(user, base_url)


# Global instance
email_verification_manager = EmailVerificationManager()
