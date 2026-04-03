#!/usr/bin/env python3
"""
Unit tests for Credential Vault
"""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

from xencode.auth.credential_vault import (
    Credential,
    CredentialVault,
    CredentialBackend,
    WindowsCredentialManagerBackend,
    EnvironmentBackend,
    get_vault,
    get_credential,
    get_secret,
    set_credential,
    delete_credential,
)


class TestCredential:
    """Tests for Credential dataclass"""
    
    def test_credential_creation(self):
        """Test creating credential"""
        cred = Credential(
            service="openai",
            username="api_key",
            secret="sk-test123",
            description="Test API key",
        )
        
        assert cred.service == "openai"
        assert cred.username == "api_key"
        assert cred.secret == "sk-test123"
        assert cred.description == "Test API key"
    
    def test_credential_default_metadata(self):
        """Test credential default metadata"""
        cred = Credential(
            service="test",
            username="user",
            secret="secret",
        )
        
        assert cred.metadata == {}
    
    def test_credential_with_metadata(self):
        """Test credential with custom metadata"""
        cred = Credential(
            service="test",
            username="user",
            secret="secret",
            metadata={"source": "migration"},
        )
        
        assert cred.metadata == {"source": "migration"}


class TestEnvironmentBackend:
    """Tests for EnvironmentBackend"""
    
    @pytest.fixture
    def backend(self):
        """Create environment backend"""
        return EnvironmentBackend()
    
    def test_is_available(self, backend):
        """Test environment backend is always available"""
        assert backend.is_available() is True
    
    def test_set_and_get_credential(self, backend):
        """Test storing and retrieving credential"""
        cred = Credential(
            service="test_service",
            username="test_user",
            secret="test_secret",
        )
        
        # Set credential
        assert backend.set_credential(cred) is True
        
        # Get credential
        retrieved = backend.get_credential("test_service", "test_user")
        
        assert retrieved is not None
        assert retrieved.service == "test_service"
        assert retrieved.username == "test_user"
        assert retrieved.secret == "test_secret"
    
    def test_get_nonexistent_credential(self, backend):
        """Test getting nonexistent credential"""
        result = backend.get_credential("nonexistent", "user")
        assert result is None
    
    def test_delete_credential(self, backend):
        """Test deleting credential"""
        cred = Credential(
            service="test",
            username="user",
            secret="secret",
        )
        
        backend.set_credential(cred)
        assert backend.delete_credential("test", "user") is True
        
        # Verify deleted
        retrieved = backend.get_credential("test", "user")
        assert retrieved is None
    
    def test_list_services(self, backend):
        """Test listing services"""
        backend.set_credential(Credential("service1", "user", "secret"))
        backend.set_credential(Credential("service2", "user", "secret"))
        
        services = backend.list_services()
        
        assert "service1" in services
        assert "service2" in services
    
    def test_env_variable_format(self, backend):
        """Test environment variable name format"""
        env_name = backend._make_env_name("openai", "api_key")
        assert env_name == "XENCODE_OPENAI_API_KEY"


class TestWindowsCredentialManagerBackend:
    """Tests for WindowsCredentialManagerBackend"""
    
    @pytest.fixture
    def backend(self):
        """Create Windows backend (may not be available on all systems)"""
        return WindowsCredentialManagerBackend()
    
    def test_is_available_non_windows(self, backend):
        """Test availability check on non-Windows"""
        if sys.platform != "win32":
            assert backend.is_available() is False
    
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_is_available_windows(self, backend):
        """Test availability on Windows"""
        # May or may not be available depending on system
        result = backend.is_available()
        assert isinstance(result, bool)
    
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_set_and_get_credential(self, backend):
        """Test credential storage on Windows"""
        if not backend.is_available():
            pytest.skip("Windows Credential Manager not available")
        
        cred = Credential(
            service="test_win",
            username="test_user",
            secret="test_secret",
        )
        
        # Set credential
        assert backend.set_credential(cred) is True
        
        # Get credential
        retrieved = backend.get_credential("test_win", "test_user")
        
        assert retrieved is not None
        assert retrieved.secret == "test_secret"
        
        # Clean up
        backend.delete_credential("test_win", "test_user")
    
    def test_target_name_format(self, backend):
        """Test Windows target name format"""
        target = backend._make_target_name("openai", "api_key")
        assert target.startswith("Xencode_")
        assert "openai" in target
        assert "api_key" in target


class TestCredentialVault:
    """Tests for CredentialVault"""
    
    @pytest.fixture
    def vault(self):
        """Create vault with environment backend only"""
        with patch.object(WindowsCredentialManagerBackend, 'is_available', return_value=False):
            return CredentialVault(prefer_windows=True)
    
    def test_vault_initialization(self, vault):
        """Test vault initializes with backends"""
        assert len(vault.backends) >= 1  # At least environment backend
    
    def test_vault_get_status(self, vault):
        """Test vault status"""
        status = vault.get_status()
        
        assert 'backends' in status
        assert 'services' in status
        assert 'has_credentials' in status
        assert 'primary_backend' in status
    
    def test_vault_set_and_get(self, vault):
        """Test storing and retrieving from vault"""
        cred = Credential(
            service="vault_test",
            username="api_key",
            secret="vault_secret_123",
        )
        
        # Set credential
        assert vault.set(cred) is True
        
        # Get credential
        retrieved = vault.get("vault_test", "api_key")
        
        assert retrieved is not None
        assert retrieved.secret == "vault_secret_123"
        
        # Clean up
        vault.delete("vault_test", "api_key")
    
    def test_vault_get_secret(self, vault):
        """Test getting secret value directly"""
        cred = Credential(
            service="secret_test",
            username="key",
            secret="my_secret_value",
        )
        
        vault.set(cred)
        secret = vault.get_secret("secret_test", "key")
        
        assert secret == "my_secret_value"
        
        vault.delete("secret_test", "key")
    
    def test_vault_get_nonexistent(self, vault):
        """Test getting nonexistent credential"""
        result = vault.get("nonexistent_service", "nonexistent_user")
        assert result is None
    
    def test_vault_list_services(self, vault):
        """Test listing services"""
        # Start fresh
        vault.set(Credential("svc1", "user", "secret"))
        vault.set(Credential("svc2", "user", "secret"))
        
        services = vault.list_services()
        
        assert "svc1" in services
        assert "svc2" in services
        
        vault.delete("svc1", "user")
        vault.delete("svc2", "user")
    
    def test_vault_has_credentials(self, vault):
        """Test checking if vault has credentials"""
        # Note: Environment backend may have credentials from other tests
        # Just test that the method returns a boolean
        result = vault.has_credentials()
        assert isinstance(result, bool)
        
        # Add a credential and verify it returns True
        vault.set(Credential("test_has", "user", "secret"))
        assert vault.has_credentials() is True
        
        vault.delete("test_has", "user")
    
    def test_vault_backend_priority(self):
        """Test backend priority (Windows first)"""
        # Mock Windows backend as available
        with patch.object(WindowsCredentialManagerBackend, 'is_available', return_value=True):
            vault = CredentialVault(prefer_windows=True)
            
            # Windows backend should be first
            assert isinstance(vault.backends[0], WindowsCredentialManagerBackend)
    
    def test_vault_migration_from_config(self, vault, tmp_path):
        """Test migrating credentials from config file"""
        import json
        
        # Create test config file
        config_data = {
            'providers': {
                'openai': {
                    'api_key': 'sk-test123',
                    'base_url': 'https://api.openai.com/v1',
                },
                'anthropic': {
                    'api_key': 'sk-ant-test456',
                },
            }
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Migrate
        result = vault.migrate_from_config(config_file, delete_after=False)
        
        assert result['migrated'] == 2
        assert result['failed'] == 0
        assert len(result['errors']) == 0
        
        # Verify credentials were migrated
        openai_cred = vault.get("openai", "api_key")
        assert openai_cred is not None
        assert openai_cred.secret == 'sk-test123'
        
        anthropic_cred = vault.get("anthropic", "api_key")
        assert anthropic_cred is not None
        assert anthropic_cred.secret == 'sk-ant-test456'
        
        # Clean up
        vault.delete("openai", "api_key")
        vault.delete("anthropic", "api_key")
    
    def test_vault_migration_with_env_reference(self, vault, tmp_path):
        """Test migration skips environment variable references"""
        import json
        
        config_data = {
            'providers': {
                'openai': {
                    'api_key': '${OPENAI_API_KEY}',  # Env reference, should skip
                },
                'test_provider': {
                    'api_key': 'actual_secret',
                },
            }
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        result = vault.migrate_from_config(config_file)
        
        assert result['migrated'] == 1  # Only test_provider
        assert result['failed'] == 0


class TestGlobalVault:
    """Tests for global vault functions"""
    
    def test_get_vault_singleton(self):
        """Test get_vault returns singleton"""
        # Reset singleton
        import xencode.auth.credential_vault as vault_module
        vault_module._vault = None
        
        vault1 = get_vault()
        vault2 = get_vault()
        
        assert vault1 is vault2
    
    def test_get_credential(self):
        """Test global get_credential function"""
        # Reset singleton and mock Windows backend
        import xencode.auth.credential_vault as vault_module
        vault_module._vault = None
        
        with patch.object(WindowsCredentialManagerBackend, 'is_available', return_value=False):
            # Set a credential
            cred = Credential("global_test", "user", "secret")
            set_credential(cred)
            
            # Get it back
            retrieved = get_credential("global_test", "user")
            assert retrieved is not None
            assert retrieved.secret == "secret"
            
            # Clean up
            delete_credential("global_test", "user")
        
        # Reset singleton
        vault_module._vault = None
    
    def test_get_secret(self):
        """Test global get_secret function"""
        # Reset singleton and mock Windows backend
        import xencode.auth.credential_vault as vault_module
        vault_module._vault = None
        
        with patch.object(WindowsCredentialManagerBackend, 'is_available', return_value=False):
            # Set a credential
            set_credential(Credential("secret_test", "key", "my_secret"))
            
            # Get secret
            secret = get_secret("secret_test", "key")
            assert secret == "my_secret"
            
            # Clean up
            delete_credential("secret_test", "key")
        
        # Reset singleton
        vault_module._vault = None


class TestCredentialServiceConstants:
    """Tests for service name constants"""
    
    def test_service_constants(self):
        """Test service name constants"""
        assert CredentialVault.SERVICE_QWEN == "qwen"
        assert CredentialVault.SERVICE_OPENROUTER == "openrouter"
        assert CredentialVault.SERVICE_OPENAI == "openai"
        assert CredentialVault.SERVICE_ANTHROPIC == "anthropic"
        assert CredentialVault.SERVICE_OLLAMA == "ollama"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
