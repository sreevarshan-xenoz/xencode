#!/usr/bin/env python3
"""
Unit tests for JsonFileCredentialVault
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from xencode.auth.credential_vault import (
    Credential,
    WindowsCredentialManagerBackend,
)
from xencode.auth.json_file_vault import JsonFileCredentialVault


class TestJsonFileCredentialVault:
    """Tests for JsonFileCredentialVault"""

    @pytest.fixture
    def temp_vault_path(self):
        """Create a temporary vault file path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "vault.json"

    @pytest.fixture
    def vault(self, temp_vault_path):
        """Create JsonFileCredentialVault with temp storage"""
        return JsonFileCredentialVault(vault_path=temp_vault_path)

    # --- Basic operations ---

    def test_initialization(self, temp_vault_path):
        """Test vault initializes with only file-based backend"""
        vault = JsonFileCredentialVault(vault_path=temp_vault_path)
        assert len(vault.backends) == 1
        assert vault.backends[0].__class__.__name__ == "FileBasedCredentialBackend"

    def test_set_and_get(self, vault):
        """Test storing and retrieving a credential"""
        cred = Credential("test_service", "api_key", "sk-test-secret")
        assert vault.set(cred) is True

        retrieved = vault.get("test_service", "api_key")
        assert retrieved is not None
        assert retrieved.service == "test_service"
        assert retrieved.username == "api_key"
        assert retrieved.secret == "sk-test-secret"

    def test_get_nonexistent(self, vault):
        """Test getting a credential that doesn't exist"""
        result = vault.get("nonexistent", "key")
        assert result is None

    def test_get_secret(self, vault):
        """Test get_secret convenience method"""
        vault.set(Credential("svc", "key", "secret_value"))
        secret = vault.get_secret("svc", "key")
        assert secret == "secret_value"

    def test_delete(self, vault):
        """Test deleting a credential"""
        vault.set(Credential("del_test", "key", "secret"))
        assert vault.get("del_test", "key") is not None
        assert vault.delete("del_test", "key") is True
        assert vault.get("del_test", "key") is None

    def test_list_services(self, vault):
        """Test listing services"""
        vault.set(Credential("svc1", "user", "secret"))
        vault.set(Credential("svc2", "user", "secret"))

        services = vault.list_services()
        assert "svc1" in services
        assert "svc2" in services

    def test_has_credentials(self, vault):
        """Test has_credentials check"""
        assert vault.has_credentials() is False
        vault.set(Credential("test", "user", "secret"))
        assert vault.has_credentials() is True

    def test_overwrite(self, vault):
        """Test overwriting an existing credential"""
        vault.set(Credential("overwrite", "key", "original"))
        vault.set(Credential("overwrite", "key", "updated"))
        retrieved = vault.get("overwrite", "key")
        assert retrieved is not None
        assert retrieved.secret == "updated"

    # --- Vault path ---

    def test_default_vault_path(self):
        """Test default vault path is ~/.xencode/vault.json"""
        vault_path = Path.home() / ".xencode" / "vault.json"
        vault = JsonFileCredentialVault()
        assert vault.get_vault_path() == vault_path

    def test_custom_vault_path(self, temp_vault_path):
        """Test custom vault path is respected"""
        vault = JsonFileCredentialVault(vault_path=temp_vault_path)
        assert vault.get_vault_path() == temp_vault_path

    # --- Encryption ---

    def test_is_encrypted(self, vault):
        """Test encryption status is a boolean"""
        assert isinstance(vault.is_encrypted(), bool)

    def test_encrypted_on_disk(self, temp_vault_path):
        """Test that stored secrets are not in plaintext on disk"""
        vault = JsonFileCredentialVault(vault_path=temp_vault_path)
        vault.set(Credential("disk_test", "key", "my_super_secret_value"))

        with open(temp_vault_path, "r") as f:
            data = json.load(f)

        stored = data["credentials"]["disk_test:key"]["secret"]
        assert "my_super_secret_value" not in stored

    # --- Storage info ---

    def test_get_storage_info(self, vault):
        """Test storage info returns expected keys"""
        vault.set(Credential("info_test", "user", "secret"))
        info = vault.get_storage_info()

        assert "vault_path" in info
        assert "credential_count" in info
        assert info["credential_count"] == 1
        assert "vault_exists" in info
        assert info["vault_exists"] is True
        assert "encryption" in info
        assert "services" in info
        assert "info_test" in info["services"]
        assert "file_size_bytes" in info
        assert "master_key_provided" in info
        assert info["master_key_provided"] is False

    def test_storage_info_empty(self, temp_vault_path):
        """Test storage info for empty vault (not yet created)"""
        vault = JsonFileCredentialVault(vault_path=temp_vault_path)
        info = vault.get_storage_info()
        assert info["credential_count"] == 0
        assert info["services"] == []

    def test_storage_info_with_master_key(self, temp_vault_path):
        """Test master_key_provided is True when key is given"""
        vault = JsonFileCredentialVault(
            vault_path=temp_vault_path, master_key="test-master-key"
        )
        info = vault.get_storage_info()
        assert info["master_key_provided"] is True

    # --- Clear vault ---

    def test_clear_vault(self, vault):
        """Test clearing all credentials"""
        vault.set(Credential("svc1", "user", "secret"))
        vault.set(Credential("svc2", "user", "secret"))
        assert vault.has_credentials() is True

        assert vault.clear_vault() is True
        assert vault.has_credentials() is False
        assert vault.list_services() == []

    # --- Export ---

    def test_export_vault_to_string(self, vault):
        """Test exporting vault as JSON string"""
        vault.set(Credential("export_test", "key", "secret_value"))
        exported = vault.export_vault()

        assert exported is not None
        data = json.loads(exported)
        assert "exported_at" in data
        assert "encryption_active" in data
        assert data["credential_count"] == 1
        assert "export_test" in data["services"]

    def test_export_vault_to_file(self, vault, tmp_path):
        """Test exporting vault to a file"""
        vault.set(Credential("export_file", "key", "secret"))
        export_path = tmp_path / "export.json"

        result = vault.export_vault(export_path=export_path)
        assert result == str(export_path)
        assert export_path.exists()

        with open(export_path, "r") as f:
            data = json.load(f)
        assert data["credential_count"] == 1

    def test_export_empty_vault(self, temp_vault_path):
        """Test exporting empty vault returns JSON"""
        vault = JsonFileCredentialVault(vault_path=temp_vault_path)
        exported = vault.export_vault()
        assert exported is not None
        data = json.loads(exported)
        assert data["credential_count"] == 0

    # --- Import ---

    def test_import_vault(self, vault, tmp_path):
        """Test importing credentials from another vault"""
        # Create a source vault with some credentials
        src_path = tmp_path / "source_vault.json"
        src_vault = JsonFileCredentialVault(vault_path=src_path)
        src_vault.set(Credential("import_svc1", "key", "secret1"))
        src_vault.set(Credential("import_svc2", "key", "secret2"))

        # Import into the test vault
        count = vault.import_vault(src_path)
        assert count == 2

        # Verify credentials were imported
        cred1 = vault.get("import_svc1", "key")
        assert cred1 is not None
        assert cred1.secret == "secret1"

        cred2 = vault.get("import_svc2", "key")
        assert cred2 is not None
        assert cred2.secret == "secret2"

    def test_import_merges_with_existing(self, vault, tmp_path):
        """Test import merges credentials, overwriting existing ones"""
        vault.set(Credential("existing", "key", "original"))

        src_path = tmp_path / "merge_vault.json"
        src_vault = JsonFileCredentialVault(vault_path=src_path)
        src_vault.set(Credential("new_svc", "key", "new_secret"))
        src_vault.set(Credential("existing", "key", "updated_secret"))

        count = vault.import_vault(src_path)
        assert count == 2

        # Existing credential was overwritten
        assert vault.get("existing", "key").secret == "updated_secret"
        # New credential was added
        assert vault.get("new_svc", "key").secret == "new_secret"

    def test_import_nonexistent_file(self, vault):
        """Test importing from nonexistent file raises"""
        with pytest.raises(FileNotFoundError):
            vault.import_vault(Path("/nonexistent/vault.json"))

    # --- Status ---

    def test_get_status(self, vault):
        """Test get_status returns expected keys"""
        vault.set(Credential("status_test", "user", "secret"))
        status = vault.get_status()

        assert status["type"] == "JsonFileCredentialVault"
        assert "FileBasedCredentialBackend" in status["backends"]
        assert status["has_credentials"] is True
        assert isinstance(status["encrypted"], bool)
        assert "vault_path" in status
        assert status["credential_count"] == 1

    def test_get_status_empty(self, temp_vault_path):
        """Test status for empty vault"""
        vault = JsonFileCredentialVault(vault_path=temp_vault_path)
        status = vault.get_status()

        assert status["has_credentials"] is False
        assert status["credential_count"] == 0

    # --- Health check ---

    def test_health_check_no_vault(self, temp_vault_path):
        """Test health check when vault doesn't exist yet"""
        # Delete the file if it was created during init
        if temp_vault_path.exists():
            temp_vault_path.unlink()

        health = JsonFileCredentialVault.health_check(vault_path=temp_vault_path)
        assert health["vault_exists"] is False
        assert health["is_readable"] is True  # non-existent = not an error

    def test_health_check_with_vault(self, vault, temp_vault_path):
        """Test health check with a populated vault"""
        vault.set(Credential("health", "key", "secret"))
        health = JsonFileCredentialVault.health_check(vault_path=temp_vault_path)

        assert health["vault_exists"] is True
        assert health["is_readable"] is True
        assert health["is_valid_json"] is True
        assert health["credential_count"] == 1
        assert isinstance(health["encryption_available"], bool)

    def test_health_check_default_path(self):
        """Test health check uses default path"""
        health = JsonFileCredentialVault.health_check()
        assert "vault_path" in health
        assert health["vault_path"] == str(Path.home() / ".xencode" / "vault.json")

    # --- Persistence ---

    def test_persistence_across_instances(self, temp_vault_path):
        """Test credentials persist when creating a new vault instance"""
        vault1 = JsonFileCredentialVault(vault_path=temp_vault_path)
        vault1.set(Credential("persist", "key", "persistent_secret"))

        vault2 = JsonFileCredentialVault(vault_path=temp_vault_path)
        retrieved = vault2.get("persist", "key")
        assert retrieved is not None
        assert retrieved.secret == "persistent_secret"

    # --- Master key ---

    def test_with_master_key(self, temp_vault_path):
        """Test vault works with explicit master key"""
        vault = JsonFileCredentialVault(
            vault_path=temp_vault_path, master_key="my-custom-master-key"
        )
        assert vault.is_encrypted()  # Should always be encrypted with key provided

        vault.set(Credential("key_test", "user", "secret_with_key"))
        retrieved = vault.get("key_test", "user")
        assert retrieved is not None
        assert retrieved.secret == "secret_with_key"

    # --- Async operations ---

    @pytest.mark.asyncio
    async def test_async_set_and_get(self, vault):
        """Test async set and get operations"""
        result = await vault.set_async(Credential("async_test", "key", "async_secret"))
        assert result is True

        retrieved = await vault.get_async("async_test", "key")
        assert retrieved is not None
        assert retrieved.secret == "async_secret"

    @pytest.mark.asyncio
    async def test_async_delete(self, vault):
        """Test async delete operation"""
        vault.set(Credential("async_del", "key", "secret"))
        deleted = await vault.delete_async("async_del", "key")
        assert deleted is True
        assert vault.get("async_del", "key") is None

    @pytest.mark.asyncio
    async def test_async_clear_vault(self, vault):
        """Test async clear vault"""
        vault.set(Credential("async_clear", "user", "secret"))
        assert await vault.clear_vault_async() is True
        assert vault.has_credentials() is False

    @pytest.mark.asyncio
    async def test_async_storage_info(self, vault):
        """Test async storage info"""
        vault.set(Credential("async_info", "user", "secret"))
        info = await vault.get_storage_info_async()
        assert info["credential_count"] >= 1

    @pytest.mark.asyncio
    async def test_async_export(self, vault):
        """Test async export"""
        vault.set(Credential("async_export", "key", "secret"))
        exported = await vault.export_vault_async()
        assert exported is not None
        data = json.loads(exported)
        assert data["credential_count"] == 1

    @pytest.mark.asyncio
    async def test_async_import(self, vault, tmp_path):
        """Test async import"""
        src_path = tmp_path / "async_source.json"
        src_vault = JsonFileCredentialVault(vault_path=src_path)
        src_vault.set(Credential("async_imp", "key", "imported_secret"))

        count = await vault.import_vault_async(src_path)
        assert count == 1
        assert vault.get("async_imp", "key").secret == "imported_secret"

    # --- Edge cases ---

    def test_empty_string_secret(self, vault):
        """Test storing credential with empty string secret"""
        vault.set(Credential("empty_test", "key", ""))
        retrieved = vault.get("empty_test", "key")
        assert retrieved is not None
        assert retrieved.secret == ""

    def test_special_chars_in_secret(self, vault):
        """Test storing credential with special characters"""
        special_secret = '!@#$%^&*()_+-=[]{}|;:\'",.<>?/~`\n\t\\'
        vault.set(Credential("special", "key", special_secret))
        retrieved = vault.get("special", "key")
        assert retrieved is not None
        assert retrieved.secret == special_secret

    def test_unicode_secret(self, vault):
        """Test storing credential with unicode characters"""
        unicode_secret = "日本語 Español العربية 🔑 🚀"
        vault.set(Credential("unicode", "key", unicode_secret))
        retrieved = vault.get("unicode", "key")
        assert retrieved is not None
        assert retrieved.secret == unicode_secret

    def test_multiple_credentials_same_service(self, vault):
        """Test multiple usernames for same service"""
        vault.set(Credential("multi", "api_key", "key_secret"))
        vault.set(Credential("multi", "access_token", "token_secret"))

        key_cred = vault.get("multi", "api_key")
        assert key_cred is not None
        assert key_cred.secret == "key_secret"

        token_cred = vault.get("multi", "access_token")
        assert token_cred is not None
        assert token_cred.secret == "token_secret"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
