"""
Data encryption utilities for Xencode
Provides encryption for sensitive information
"""
import os
import base64
from typing import Union, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets


class DataEncryption:
    """Class for encrypting and decrypting sensitive data"""
    
    def __init__(self, password: str = None, key: bytes = None):
        """
        Initialize the encryption utility.
        
        Args:
            password: Password to derive encryption key from (optional if key is provided)
            key: Encryption key (optional if password is provided)
        """
        if key is not None:
            self.key = key
        elif password is not None:
            self.key = self._derive_key_from_password(password)
        else:
            # Generate a random key if neither password nor key is provided
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """
        Derive an encryption key from a password using PBKDF2.
        
        Args:
            password: Password to derive key from
            
        Returns:
            Derived encryption key
        """
        # Generate a random salt
        salt = os.urandom(16)
        
        # Use PBKDF2 to derive a key from the password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        # Store salt for decryption (in a real app, you'd store this securely)
        self._salt = salt
        return key
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            Encrypted data as bytes
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted_data = self.cipher.encrypt(data)
        return encrypted_data
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data as bytes
            
        Returns:
            Decrypted data as string
        """
        decrypted_data = self.cipher.decrypt(encrypted_data)
        return decrypted_data.decode('utf-8')
    
    def encrypt_to_string(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data and return as base64-encoded string.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            Encrypted data as base64-encoded string
        """
        encrypted_data = self.encrypt(data)
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_from_string(self, encrypted_string: str) -> str:
        """
        Decrypt data from base64-encoded string.
        
        Args:
            encrypted_string: Encrypted data as base64-encoded string
            
        Returns:
            Decrypted data as string
        """
        encrypted_data = base64.b64decode(encrypted_string.encode('utf-8'))
        return self.decrypt(encrypted_data)


class AESEncryption:
    """Alternative encryption using AES-GCM for additional security"""
    
    def __init__(self, key: bytes = None):
        """
        Initialize AES-GCM encryption.
        
        Args:
            key: Encryption key (generates random if not provided)
        """
        self.key = key or AESGCM.generate_key(bit_length=256)
        self.aesgcm = AESGCM(self.key)
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data using AES-GCM.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            Encrypted data with nonce prepended
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Generate a random nonce for this encryption
        nonce = os.urandom(12)  # 96-bit nonce for AES-GCM
        
        # Encrypt the data
        encrypted_data = self.aesgcm.encrypt(nonce, data, associated_data=None)
        
        # Return nonce + encrypted data
        return nonce + encrypted_data
    
    def decrypt(self, encrypted_data_with_nonce: bytes) -> str:
        """
        Decrypt data using AES-GCM.
        
        Args:
            encrypted_data_with_nonce: Encrypted data with nonce prepended
            
        Returns:
            Decrypted data as string
        """
        # Extract nonce (first 12 bytes) and encrypted data
        nonce = encrypted_data_with_nonce[:12]
        encrypted_data = encrypted_data_with_nonce[12:]
        
        # Decrypt the data
        decrypted_data = self.aesgcm.decrypt(nonce, encrypted_data, associated_data=None)
        return decrypted_data.decode('utf-8')
    
    def encrypt_to_string(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data and return as base64-encoded string.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            Encrypted data as base64-encoded string
        """
        encrypted_data = self.encrypt(data)
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_from_string(self, encrypted_string: str) -> str:
        """
        Decrypt data from base64-encoded string.
        
        Args:
            encrypted_string: Encrypted data as base64-encoded string
            
        Returns:
            Decrypted data as string
        """
        encrypted_data = base64.b64decode(encrypted_string.encode('utf-8'))
        return self.decrypt(encrypted_data)


class SecureConfig:
    """Secure configuration storage with encryption"""
    
    def __init__(self, password: str = None):
        """
        Initialize secure configuration.
        
        Args:
            password: Password for encryption (generates random if not provided)
        """
        self.password = password or secrets.token_urlsafe(32)
        self.encryption = DataEncryption(password=self.password)
        self.config_data = {}
    
    def set_encrypted(self, key: str, value: str) -> None:
        """
        Set an encrypted configuration value.
        
        Args:
            key: Configuration key
            value: Value to encrypt and store
        """
        encrypted_value = self.encryption.encrypt_to_string(value)
        self.config_data[key] = encrypted_value
    
    def get_decrypted(self, key: str) -> Optional[str]:
        """
        Get a decrypted configuration value.
        
        Args:
            key: Configuration key
            
        Returns:
            Decrypted value or None if key doesn't exist
        """
        if key not in self.config_data:
            return None
        
        encrypted_value = self.config_data[key]
        try:
            return self.encryption.decrypt_from_string(encrypted_value)
        except Exception:
            # If decryption fails, return None
            return None
    
    def update_password(self, new_password: str) -> None:
        """
        Update the password and re-encrypt all values.
        
        Args:
            new_password: New password for encryption
        """
        # Decrypt all current values
        decrypted_values = {}
        for key, encrypted_value in self.config_data.items():
            try:
                decrypted_values[key] = self.encryption.decrypt_from_string(encrypted_value)
            except Exception:
                # If we can't decrypt a value, skip it
                continue
        
        # Create new encryption with new password
        self.password = new_password
        self.encryption = DataEncryption(password=new_password)
        
        # Re-encrypt all values
        self.config_data = {}
        for key, value in decrypted_values.items():
            self.set_encrypted(key, value)


class SensitiveDataManager:
    """Manager for handling sensitive data with automatic encryption/decryption"""
    
    def __init__(self, encryption_key: bytes = None):
        """
        Initialize sensitive data manager.
        
        Args:
            encryption_key: Key for encryption (generates random if not provided)
        """
        self.encryption = DataEncryption(key=encryption_key or Fernet.generate_key())
        self.data_store = {}
    
    def store_sensitive(self, key: str, data: str, encrypt: bool = True) -> None:
        """
        Store sensitive data.
        
        Args:
            key: Key for the data
            data: Data to store
            encrypt: Whether to encrypt the data (default True)
        """
        if encrypt:
            encrypted_data = self.encryption.encrypt_to_string(data)
            self.data_store[key] = {
                'data': encrypted_data,
                'encrypted': True
            }
        else:
            self.data_store[key] = {
                'data': data,
                'encrypted': False
            }
    
    def retrieve_sensitive(self, key: str, decrypt: bool = True) -> Optional[str]:
        """
        Retrieve sensitive data.
        
        Args:
            key: Key for the data
            decrypt: Whether to decrypt the data (default True)
            
        Returns:
            Retrieved data or None if key doesn't exist
        """
        if key not in self.data_store:
            return None
        
        stored = self.data_store[key]
        data = stored['data']
        
        if stored['encrypted'] and decrypt:
            try:
                return self.encryption.decrypt_from_string(data)
            except Exception:
                # If decryption fails, return None
                return None
        elif not stored['encrypted'] and decrypt:
            # Data wasn't encrypted but we want to decrypt
            return None
        else:
            # Return as-is
            return data
    
    def delete_sensitive(self, key: str) -> bool:
        """
        Delete sensitive data.
        
        Args:
            key: Key for the data to delete
            
        Returns:
            True if deleted, False if key didn't exist
        """
        if key in self.data_store:
            del self.data_store[key]
            return True
        return False
    
    def list_keys(self) -> list:
        """
        List all stored keys.
        
        Returns:
            List of stored keys
        """
        return list(self.data_store.keys())


# Global instances
sensitive_data_manager = SensitiveDataManager()
secure_config = SecureConfig()


def get_sensitive_data_manager() -> SensitiveDataManager:
    """Get the global sensitive data manager instance"""
    return sensitive_data_manager


def get_secure_config() -> SecureConfig:
    """Get the global secure config instance"""
    return secure_config


def encrypt_data(data: Union[str, bytes], password: str = None) -> str:
    """
    Convenience function to encrypt data.
    
    Args:
        data: Data to encrypt
        password: Password for encryption (uses random if not provided)
        
    Returns:
        Encrypted data as base64-encoded string
    """
    encryption = DataEncryption(password=password)
    return encryption.encrypt_to_string(data)


def decrypt_data(encrypted_data: str, password: str) -> str:
    """
    Convenience function to decrypt data.
    
    Args:
        encrypted_data: Encrypted data as base64-encoded string
        password: Password used for encryption
        
    Returns:
        Decrypted data as string
    """
    encryption = DataEncryption(password=password)
    return encryption.decrypt_from_string(encrypted_data)


def store_sensitive_data(key: str, data: str) -> None:
    """
    Convenience function to store sensitive data.
    
    Args:
        key: Key for the data
        data: Data to store
    """
    manager = get_sensitive_data_manager()
    manager.store_sensitive(key, data)


def retrieve_sensitive_data(key: str) -> Optional[str]:
    """
    Convenience function to retrieve sensitive data.
    
    Args:
        key: Key for the data
        
    Returns:
        Retrieved data or None if key doesn't exist
    """
    manager = get_sensitive_data_manager()
    return manager.retrieve_sensitive(key)


def set_secure_config(key: str, value: str) -> None:
    """
    Convenience function to set secure config.
    
    Args:
        key: Configuration key
        value: Value to encrypt and store
    """
    config = get_secure_config()
    config.set_encrypted(key, value)


def get_secure_config_value(key: str) -> Optional[str]:
    """
    Convenience function to get secure config value.
    
    Args:
        key: Configuration key
        
    Returns:
        Decrypted value or None if key doesn't exist
    """
    config = get_secure_config()
    return config.get_decrypted(key)