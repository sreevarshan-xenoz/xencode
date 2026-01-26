"""
Secure communication channels for inter-agent communication in Xencode
"""
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod
import hashlib
import hmac
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json
from datetime import datetime, timedelta


class SecureChannel(ABC):
    """Abstract base class for secure communication channels."""
    
    @abstractmethod
    def encrypt_message(self, message_data: str) -> str:
        """Encrypt a message for secure transmission."""
        pass
    
    @abstractmethod
    def decrypt_message(self, encrypted_data: str) -> str:
        """Decrypt a received message."""
        pass
    
    @abstractmethod
    def sign_message(self, message_data: str) -> str:
        """Sign a message to ensure authenticity."""
        pass
    
    @abstractmethod
    def verify_signature(self, message_data: str, signature: str) -> bool:
        """Verify the signature of a received message."""
        pass


class SymmetricEncryptionChannel(SecureChannel):
    """Secure channel using symmetric encryption."""
    
    def __init__(self, shared_secret: Optional[str] = None):
        if shared_secret is None:
            self.shared_secret = secrets.token_urlsafe(32)
        else:
            self.shared_secret = shared_secret
            
        # Derive encryption key from shared secret
        password = self.shared_secret.encode()
        salt = b'stable_salt_for_demo'  # In production, use random salt and store securely
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher_suite = Fernet(key)
        
        # Signing key derived similarly
        signing_key = hashlib.sha256(password).digest()
        self.signing_key = signing_key
        
    def encrypt_message(self, message_data: str) -> str:
        """Encrypt a message using the cipher suite."""
        encrypted_bytes = self.cipher_suite.encrypt(message_data.encode())
        return base64.b64encode(encrypted_bytes).decode()
    
    def decrypt_message(self, encrypted_data: str) -> str:
        """Decrypt a message using the cipher suite."""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
        return decrypted_bytes.decode()
    
    def sign_message(self, message_data: str) -> str:
        """Sign a message using HMAC."""
        signature = hmac.new(
            self.signing_key,
            message_data.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_signature(self, message_data: str, signature: str) -> bool:
        """Verify the signature of a message."""
        expected_signature = self.sign_message(message_data)
        return hmac.compare_digest(expected_signature, signature)


class ChannelManager:
    """Manages secure communication channels between agents."""
    
    def __init__(self):
        self.channels: Dict[Tuple[str, str], SecureChannel] = {}  # (agent1, agent2) -> channel
        self.agent_secrets: Dict[str, str] = {}  # agent_id -> secret
        self.default_channel: Optional[SecureChannel] = None
        
    def create_channel(self, agent1_id: str, agent2_id: str, 
                      shared_secret: Optional[str] = None) -> SecureChannel:
        """Create a secure channel between two agents."""
        # Sort agent IDs to ensure consistent key regardless of order
        sorted_agents = tuple(sorted([agent1_id, agent2_id]))
        
        if shared_secret is None:
            # Generate a unique secret for this channel
            shared_secret = f"{agent1_id}:{agent2_id}:{secrets.token_urlsafe(16)}"
            
        channel = SymmetricEncryptionChannel(shared_secret)
        self.channels[sorted_agents] = channel
        
        # Store secrets for each agent (for demo purposes)
        if agent1_id not in self.agent_secrets:
            self.agent_secrets[agent1_id] = shared_secret
        if agent2_id not in self.agent_secrets:
            self.agent_secrets[agent2_id] = shared_secret
            
        return channel
        
    def get_channel(self, agent1_id: str, agent2_id: str) -> Optional[SecureChannel]:
        """Get the secure channel between two agents."""
        sorted_agents = tuple(sorted([agent1_id, agent2_id]))
        return self.channels.get(sorted_agents)
        
    def encrypt_for_agents(self, agent1_id: str, agent2_id: str, 
                          message_data: str) -> Optional[str]:
        """Encrypt a message for transmission between two agents."""
        channel = self.get_channel(agent1_id, agent2_id)
        if channel:
            return channel.encrypt_message(message_data)
        return None
        
    def decrypt_for_agents(self, agent1_id: str, agent2_id: str, 
                          encrypted_data: str) -> Optional[str]:
        """Decrypt a message received from another agent."""
        channel = self.get_channel(agent1_id, agent2_id)
        if channel:
            return channel.decrypt_message(encrypted_data)
        return None
        
    def sign_for_agents(self, agent1_id: str, agent2_id: str, 
                       message_data: str) -> Optional[str]:
        """Sign a message for transmission between two agents."""
        channel = self.get_channel(agent1_id, agent2_id)
        if channel:
            return channel.sign_message(message_data)
        return None
        
    def verify_signature_for_agents(self, agent1_id: str, agent2_id: str, 
                                  message_data: str, signature: str) -> bool:
        """Verify the signature of a message from another agent."""
        channel = self.get_channel(agent1_id, agent2_id)
        if channel:
            return channel.verify_signature(message_data, signature)
        return False
        
    def create_default_channel(self) -> SecureChannel:
        """Create a default channel for broadcast messages."""
        self.default_channel = SymmetricEncryptionChannel()
        return self.default_channel
        
    def get_default_channel(self) -> Optional[SecureChannel]:
        """Get the default channel."""
        return self.default_channel


# Example usage and testing
if __name__ == "__main__":
    # Test the secure channel
    channel = SymmetricEncryptionChannel()
    
    original_message = "Hello, this is a secret message!"
    
    # Encrypt the message
    encrypted = channel.encrypt_message(original_message)
    print(f"Encrypted: {encrypted}")
    
    # Decrypt the message
    decrypted = channel.decrypt_message(encrypted)
    print(f"Decrypted: {decrypted}")
    print(f"Match: {original_message == decrypted}")
    
    # Sign and verify
    signature = channel.sign_message(original_message)
    print(f"Signature: {signature}")
    
    is_valid = channel.verify_signature(original_message, signature)
    print(f"Signature valid: {is_valid}")
    
    # Test channel manager
    manager = ChannelManager()
    channel_ab = manager.create_channel("agent_a", "agent_b")
    
    encrypted_ab = manager.encrypt_for_agents("agent_a", "agent_b", "Secret for A and B")
    print(f"Encrypted for A-B: {encrypted_ab}")
    
    decrypted_ab = manager.decrypt_for_agents("agent_a", "agent_b", encrypted_ab)
    print(f"Decrypted for A-B: {decrypted_ab}")