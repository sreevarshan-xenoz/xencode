#!/usr/bin/env python3
"""
Unit tests for FileBasedCredentialBackend
"""

import pytest
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from xencode.auth.credential_vault import (
    Credential,
    CredentialVault,
    FileBasedCredentialBackend,
    WindowsCredentialManagerBackend,
    EnvironmentBackend,
    get_vault,
    CRYPTO_AVAILABLE,
)


class TestFileBasedCredentialBackend:
    """Tests for FileBasedCredentialBackend"""

    @pytest.fixture
    def temp_vault_path(self):
        """Create a temporary vault file path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_vault.json"

    @pytest.fixture
    def backend(self, temp_vault_path):
        """Create file-based backend with temp storage"""
        return FileBasedCredentialBackend(vault_path=temp_vault_path)

    def test_is_available(self, backend):
        """Test backend is always available"""
        assert backend.is_available() is True

    def test_is_encrypted(self, backend):
        """Test encryption status matches crypto library availability"""
        assert backend.is_encrypted() == CRYPTO_AVAILABLE

    def test_get_vault_path(self, backend, temp_vault_path):
        """Test vault path is correct"""
        assert backend.get_vault_path() == temp_vault_path

    def test_set_and_get_credential(self, backend):
        """Test storing and retrieving a credential"""
        cred = Credential(
            service="test_service",
            username="api_key",
            secret="sk-test-secret-123",
            description="Test credential",
        )

        # Set credential
        assert backend.set_credential(cred) is True

        # Get credential
        retrieved = backend.get_credential("test_service", "api_key")

        assert retrieved is not None
        assert retrieved.service == "test_service"
        assert retrieved.username == "api_key"
        assert retrieved.secret == "sk-test-secret-123"
        assert retrieved.description == "Test credential"
        assert retrieved.metadata.get("source") == "file_vault"

    def test_get_nonexistent_credential(self, backend):
        """Test getting a credential that doesn't exist"""
        result = backend.get_credential("nonexistent", "user")
        assert result is None

    def test_get_empty_string_secret(self, backend):
        """Test storing and retrieving an empty secret"""
        cred = Credential("empty_test", "key", "")
        assert backend.set_credential(cred) is True

        retrieved = backend.get_credential("empty_test", "key")
        assert retrieved is not None
        assert retrieved.secret == ""

    def test_delete_credential(self, backend):
        """Test deleting a credential"""
        cred = Credential("delete_test", "user", "secret")
        assert backend.set_credential(cred) is True

        # Verify it exists
        assert backend.get_credential("delete_test", "user") is not None

        # Delete it
        assert backend.delete_credential("delete_test", "user") is True

        # Verify it's gone
        assert backend.get_credential("delete_test", "user") is None

    def test_delete_nonexistent_credential(self, backend):
        """Test deleting a credential that doesn't exist"""
        assert backend.delete_credential("nonexistent", "user") is False

    def test_list_services(self, backend):
        """Test listing services"""
        backend.set_credential(Credential("svc1", "user", "secret"))
        backend.set_credential(Credential("svc2", "user", "secret"))
        backend.set_credential(Credential("svc1", "other_user", "secret"))

        services = backend.list_services()

        assert "svc1" in services
        assert "svc2" in services
        assert len(services) == 2  # svc1 should appear only once

    def test_list_services_empty(self, backend):
        """Test listing services when vault is empty"""
        assert backend.list_services() == []

    def test_clear_vault(self, backend):
        """Test clearing the entire vault"""
        backend.set_credential(Credential("svc1", "user", "secret"))
        backend.set_credential(Credential("svc2", "user", "secret"))

        assert len(backend.list_services()) == 2

        assert backend.clear_vault() is True
        assert backend.list_services() == []

    def test_persistence_across_instances(self, temp_vault_path):
        """Test credentials persist when creating a new backend instance"""
        # First instance: store a credential
        backend1 = FileBasedCredentialBackend(vault_path=temp_vault_path)
        backend1.set_credential(Credential("persist_test", "key", "persist_secret"))

        # Second instance: should read the same data
        backend2 = FileBasedCredentialBackend(vault_path=temp_vault_path)
        retrieved = backend2.get_credential("persist_test", "key")

        assert retrieved is not None
        assert retrieved.secret == "persist_secret"

    def test_get_storage_info(self, backend):
        """Test storage info returns correct data"""
        backend.set_credential(Credential("info_test", "user", "secret"))

        info = backend.get_storage_info()

        assert "vault_path" in info
        assert info["credential_count"] == 1
        assert info["vault_exists"] is True
        assert "encryption" in info
        assert "services" in info
        assert "info_test" in info["services"]

    def test_get_storage_info_empty(self, backend):
        """Test storage info for empty vault"""
        info = backend.get_storage_info()

        assert info["credential_count"] == 0
        assert info["services"] == []

    def test_encryption_changes_secret_on_disk(self, temp_vault_path):
        """Test that the stored secret on disk differs from plaintext"""
        backend = FileBasedCredentialBackend(vault_path=temp_vault_path)

        test_secret = "my_super_secret_api_key_12345"
        backend.set_credential(Credential("enc_test", "key", test_secret))

        # Read the raw file and confirm the secret is not in plaintext
        with open(temp_vault_path, "r") as f:
            raw_data = json.load(f)

        stored_entry = raw_data["credentials"].get("enc_test:key")
        assert stored_entry is not None

        # The stored secret should NOT be the plaintext value
        stored_secret = stored_entry["secret"]
        assert test_secret not in stored_secret, "Secret should not be stored in plaintext"

        # But the backend can decrypt it back
        retrieved = backend.get_credential("enc_test", "key")
        assert retrieved is not None
        assert retrieved.secret == test_secret

    def test_overwrite_existing_credential(self, backend):
        """Test overwriting an existing credential"""
        backend.set_credential(Credential("overwrite", "key", "original_secret"))
        backend.set_credential(Credential("overwrite", "key", "new_secret"))

        retrieved = backend.get_credential("overwrite", "key")
        assert retrieved is not None
        assert retrieved.secret == "new_secret"

    def test_vault_file_format(self, temp_vault_path):
        """Test the vault file structure"""
        backend = FileBasedCredentialBackend(vault_path=temp_vault_path)
        backend.set_credential(Credential("format_test", "user", "secret_value"))

        with open(temp_vault_path, "r") as f:
            data = json.load(f)

        assert "version" in data
        assert data["version"] == 1
        assert "created_at" in data
        assert "credentials" in data
        assert "format_test:user" in data["credentials"]

        entry = data["credentials"]["format_test:user"]
        assert "secret" in entry
        assert "description" in entry
        assert "metadata" in entry
        assert "updated_at" in entry

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_different_machine_keys_produce_different_ciphertext(self, temp_vault_path):
        """Test that different encryption keys produce different ciphertext for same secret"""
        import base64
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.backends import default_backend

        # Derive two different keys
        seed1 = "machine-1|/home/user1|user1"
        seed2 = "machine-2|/home/user2|user2"

        kdf1 = PBKDF2HMAC(
            algorithm=hashes.SHA256(), length=32, salt=b"xencode-vault-salt",
            iterations=600_000, backend=default_backend(),
        )
        kdf2 = PBKDF2HMAC(
            algorithm=hashes.SHA256(), length=32, salt=b"xencode-vault-salt",
            iterations=600_000, backend=default_backend(),
        )

        key1 = base64.urlsafe_b64encode(kdf1.derive(seed1.encode()))
        key2 = base64.urlsafe_b64encode(kdf2.derive(seed2.encode()))

        fernet1 = Fernet(key1)
        fernet2 = Fernet(key2)

        test_secret = "same_secret_value"
        c1 = fernet1.encrypt(test_secret.encode())
        c2 = fernet2.encrypt(test_secret.encode())

        # Different keys should produce different ciphertexts
        assert c1 != c2, "Different keys should produce different ciphertexts"
        assert c1 != test_secret.encode(), "Ciphertext should differ from plaintext"
        assert c2 != test_secret.encode(), "Ciphertext should differ from plaintext"


class TestFileBasedBackendWithVault:
    """Tests for FileBasedCredentialBackend integrated with CredentialVault"""

    @pytest.fixture
    def temp_vault_path(self):
        """Create a temporary vault file path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_vault.json"

    @pytest.fixture
    def vault(self, temp_vault_path):
        """Create vault with only file-based backend (mock Windows as unavailable)"""
        with patch.object(WindowsCredentialManagerBackend, 'is_available', return_value=False):
            return CredentialVault(prefer_windows=True, vault_path=temp_vault_path)

    def test_vault_has_file_backend(self, vault):
        """Test vault includes file-based backend"""
        backends = [type(b).__name__ for b in vault.backends]
        assert "FileBasedCredentialBackend" in backends

    def test_vault_set_and_get(self, vault):
        """Test vault-level set/get uses file backend"""
        cred = Credential("vault_file_test", "api_key", "vault_secret_456")
        assert vault.set(cred) is True

        retrieved = vault.get("vault_file_test", "api_key")
        assert retrieved is not None
        assert retrieved.secret == "vault_secret_456"

        vault.delete("vault_file_test", "api_key")

    def test_vault_backend_order(self, temp_vault_path):
        """Test backend priority: FileBased comes before Environment"""
        with patch.object(WindowsCredentialManagerBackend, 'is_available', return_value=False):
            vault = CredentialVault(prefer_windows=True, vault_path=temp_vault_path)
            backend_names = [type(b).__name__ for b in vault.backends]

            # Order should be: FileBasedCredentialBackend, EnvironmentBackend
            assert backend_names == ["FileBasedCredentialBackend", "EnvironmentBackend"]

    def test_vault_persistence(self, temp_vault_path):
        """Test credentials persist across vault instances"""
        with patch.object(WindowsCredentialManagerBackend, 'is_available', return_value=False):
            vault1 = CredentialVault(prefer_windows=True, vault_path=temp_vault_path)
            vault1.set(Credential("persist", "key", "stored_secret"))

            vault2 = CredentialVault(prefer_windows=True, vault_path=temp_vault_path)
            retrieved = vault2.get("persist", "key")

            assert retrieved is not None
            assert retrieved.secret == "stored_secret"

            vault2.delete("persist", "key")

    def test_get_secret_via_vault(self, vault):
        """Test get_secret convenience method"""
        vault.set(Credential("get_secret_test", "key", "my_secret_here"))
        secret = vault.get_secret("get_secret_test", "key")
        assert secret == "my_secret_here"
        vault.delete("get_secret_test", "key")

    def test_vault_status(self, vault):
        """Test vault status includes file backend"""
        status = vault.get_status()
        assert "FileBasedCredentialBackend" in status["backends"]

    def test_vault_delete_from_file(self, vault):
        """Test delete removes credential from file backend"""
        vault.set(Credential("del_test", "key", "secret"))
        assert vault.get("del_test", "key") is not None

        vault.delete("del_test", "key")
        assert vault.get("del_test", "key") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
