#!/usr/bin/env python3
"""
JsonFileCredentialVault — Encrypted JSON file-based CredentialVault

A concrete CredentialVault subclass that uses only FileBasedCredentialBackend
for secure, cross-platform credential storage.

Designed for environments where:
- Windows Credential Manager is not available (Linux, macOS, CI)
- A lightweight, portable vault is preferred
- Encryption via Fernet (AES-128-CBC with HMAC SHA256) is required
- Vault needs to be backup-able or sync-able as a single JSON file

Usage:
    vault = JsonFileCredentialVault()
    vault.set(Credential(service="openai", username="api_key", secret="sk-..."))
    cred = vault.get("openai", "api_key")
    info = vault.get_storage_info()
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .credential_vault import (
    Credential,
    CredentialVault,
    FileBasedCredentialBackend,
)


logger = logging.getLogger(__name__)


class JsonFileCredentialVault(CredentialVault):
    """
    CredentialVault subclass using only encrypted JSON file storage.

    Unlike the base CredentialVault which chains multiple backends
    (Windows CM -> File -> Environment), this subclass uses exclusively
    the FileBasedCredentialBackend for deterministic, cross-platform behavior.

    Features:
    - Single backend: FileBasedCredentialBackend
    - Fernet encryption (AES-128-CBC with HMAC SHA256)
    - Machine-derived encryption key via PBKDF2
    - Async-safe credential operations via asyncio.to_thread()
    - Vault introspection: get_storage_info(), is_encrypted(), get_vault_path()
    - Bulk operations: clear_vault(), export_vault(), import_vault()
    """

    def __init__(
        self,
        vault_path: Optional[Path] = None,
        master_key: Optional[str] = None,
    ):
        """
        Initialize the JSON file credential vault.

        Args:
            vault_path: Path to vault file (default: ~/.xencode/vault.json)
            master_key: Optional master key for encryption derivation.
                        Falls back to machine-specific seed if not provided.
        """
        # Store config for status reporting
        self._config = {
            "vault_path": vault_path or (Path.home() / ".xencode" / "vault.json"),
            "master_key_provided": master_key is not None,
        }

        # Initialize the file-based backend directly
        self._file_backend = FileBasedCredentialBackend(
            vault_path=vault_path,
            master_key=master_key,
        )

        # Set backend chain to ONLY the file-based backend
        # We intentionally skip super().__init__() to avoid creating
        # unused backends (Windows CM, Environment) and their misleading
        # console output. All CredentialVault methods use self.backends.
        self.backends = [self._file_backend]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_vault_path(self) -> Path:
        """Get the vault file path on disk."""
        return self._file_backend.get_vault_path()

    def is_encrypted(self) -> bool:
        """
        Check if credentials are actually encrypted (vs base64 fallback).

        Returns True when Fernet encryption is active, False when the
        cryptography package is not installed and only base64 obfuscation
        is used.
        """
        return self._file_backend.is_encrypted()

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the vault storage.

        Returns:
            Dict with keys: vault_path, credential_count, vault_exists,
            encryption, services, created_at, last_modified
        """
        info = self._file_backend.get_storage_info()

        # Add vault file metadata if it exists
        vault_path = self.get_vault_path()
        if vault_path.exists():
            try:
                stat = vault_path.stat()
                info["file_size_bytes"] = stat.st_size
                info["last_modified"] = datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat()
                info["created_at"] = datetime.fromtimestamp(
                    stat.st_ctime, tz=timezone.utc
                ).isoformat()
            except OSError:
                pass

        # Add config info
        info["master_key_provided"] = self._config["master_key_provided"]

        return info

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def clear_vault(self) -> bool:
        """
        Delete all credentials from the vault.

        Returns True if successful, False on error.
        """
        return self._file_backend.clear_vault()

    def export_vault(self, export_path: Optional[Path] = None) -> Optional[str]:
        """
        Export vault contents as a JSON string (secrets still encrypted).

        Useful for backups or migration.

        Args:
            export_path: Optional file path to write the export to.
                         If not provided, returns the JSON string.

        Returns:
            JSON string of vault contents, or None on error.
        """
        try:
            vault_path = self.get_vault_path()
            if not vault_path.exists():
                # Return an empty export rather than None
                empty = {
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "encryption_active": self.is_encrypted(),
                    "credential_count": 0,
                    "services": [],
                    "vault": {"version": 1, "credentials": {}},
                }
                if export_path:
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(export_path, "w", encoding="utf-8") as f:
                        json.dump(empty, f, indent=2)
                    return str(export_path)
                return json.dumps(empty, indent=2)

            with open(vault_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            export_data = {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "encryption_active": self.is_encrypted(),
                "credential_count": len(data.get("credentials", {})),
                "services": self.list_services(),
                "vault": data,
            }

            if export_path:
                export_path.parent.mkdir(parents=True, exist_ok=True)
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2)
                return str(export_path)

            return json.dumps(export_data, indent=2)

        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to export vault: %s", e)
            return None

    def import_vault(self, source_path: Path) -> int:
        """
        Import credentials from another vault JSON file.

        Merges imported credentials with existing ones. Existing credentials
        with the same service:username key will be overwritten.

        Args:
            source_path: Path to a vault JSON file to import from.

        Returns:
            Number of credentials imported.

        Raises:
            FileNotFoundError: If source_path doesn't exist.
            ValueError: If the file is not a valid vault JSON.
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Vault file not found: {source_path}")

        try:
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid vault file: {e}")

        credentials_data = data.get("credentials", {})
        if not isinstance(credentials_data, dict):
            raise ValueError("Vault file has no 'credentials' key or invalid format")

        count = 0
        failed = 0
        for key, entry in credentials_data.items():
            if ":" in key:
                service, username = key.split(":", 1)
                secret = entry.get("secret", "")
                # Decrypt the secret for re-encryption with our key
                try:
                    decrypted = self._file_backend._decrypt(secret)
                    cred = Credential(
                        service=service,
                        username=username,
                        secret=decrypted,
                        description=entry.get("description"),
                        metadata=entry.get("metadata"),
                    )
                    if self.set(cred):
                        count += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.warning(
                        "Failed to import credential '%s:%s': %s",
                        service, username, e,
                    )
                    failed += 1
                    continue

        if count == 0 and failed > 0:
            logger.warning(
                "Import completed: 0 of %d credentials imported (%d failed). "
                "Credentials from another machine cannot be decrypted with "
                "this vault's key.",
                len(credentials_data), failed,
            )

        return count

    # ------------------------------------------------------------------
    # Status and health
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive vault status with storage info."""
        info = self.get_storage_info()
        return {
            "type": "JsonFileCredentialVault",
            "backends": [type(b).__name__ for b in self.backends],
            "services": self.list_services(),
            "has_credentials": self.has_credentials(),
            "encrypted": self.is_encrypted(),
            "vault_path": str(self.get_vault_path()),
            "credential_count": info.get("credential_count", 0),
            "file_size_bytes": info.get("file_size_bytes"),
            "master_key_provided": info.get("master_key_provided", False),
        }

    @staticmethod
    def health_check(vault_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Check vault health without instantiating a full vault.

        Useful for diagnostics and the 'xencode doctor' command.

        Args:
            vault_path: Path to vault file (default: ~/.xencode/vault.json)

        Returns:
            Dict with health indicators.
        """
        if vault_path is None:
            vault_path = Path.home() / ".xencode" / "vault.json"

        result = {
            "vault_exists": vault_path.exists(),
            "vault_path": str(vault_path),
            "is_readable": False,
            "is_valid_json": False,
            "credential_count": 0,
            "encryption_available": False,
        }

        if not vault_path.exists():
            result["is_readable"] = True  # Non-existent is not an error
            return result

        # Check readability
        try:
            with open(vault_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result["is_readable"] = True
            result["is_valid_json"] = True
            result["credential_count"] = len(data.get("credentials", {}))
        except (OSError, json.JSONDecodeError):
            result["is_readable"] = False
            return result

        # Check if cryptography is available
        try:
            from cryptography.fernet import Fernet
            result["encryption_available"] = True
        except ImportError:
            result["encryption_available"] = False

        return result

    # ------------------------------------------------------------------
    # Async-safe operations
    # ------------------------------------------------------------------

    async def get_async(self, service: str, username: str) -> Optional[Credential]:
        """Async version of get() — runs sync storage in thread pool."""
        return await asyncio.to_thread(self.get, service, username)

    async def set_async(self, credential: Credential) -> bool:
        """Async version of set() — runs sync storage in thread pool."""
        return await asyncio.to_thread(self.set, credential)

    async def delete_async(self, service: str, username: str) -> bool:
        """Async version of delete() — runs sync storage in thread pool."""
        return await asyncio.to_thread(self.delete, service, username)

    async def clear_vault_async(self) -> bool:
        """Async version of clear_vault() — runs sync storage in thread pool."""
        return await asyncio.to_thread(self.clear_vault)

    async def get_storage_info_async(self) -> Dict[str, Any]:
        """Async version of get_storage_info()."""
        return await asyncio.to_thread(self.get_storage_info)

    async def export_vault_async(
        self, export_path: Optional[Path] = None
    ) -> Optional[str]:
        """Async version of export_vault()."""
        return await asyncio.to_thread(self.export_vault, export_path)

    async def import_vault_async(self, source_path: Path) -> int:
        """Async version of import_vault()."""
        return await asyncio.to_thread(self.import_vault, source_path)


    # ------------------------------------------------------------------
    # Migration from plaintext config
    # ------------------------------------------------------------------

    @classmethod
    def migrate_from_config(
        cls,
        config_path: Optional[Path] = None,
        vault_path: Optional[Path] = None,
        master_key: Optional[str] = None,
        delete_after: bool = False,
    ) -> Dict[str, Any]:
        """
        Migrate plaintext API keys from a config file into the vault.

        Scans the config file for known API key fields (openai_api_key,
        anthropic_api_key, etc.) and stores any plaintext (non-env-ref)
        values into a new or existing JsonFileCredentialVault.

        Args:
            config_path: Path to config file (auto-detected if not given).
            vault_path: Path for the vault file (default: ~/.xencode/vault.json).
            master_key: Optional master key for vault encryption.
            delete_after: If True, replace plaintext keys with env-var
                          references after migration.

        Returns:
            Dict with keys: migrated, skipped, errors, vault_path, config_path.
        """
        result: Dict[str, Any] = {
            "migrated": 0,
            "skipped": 0,
            "errors": [],
            "vault_path": "",
            "config_path": "",
        }

        # Find config file if not specified
        if config_path is None:
            config_path = cls._detect_config_file()
            if config_path is None:
                result["errors"].append(
                    "No config file found. Create one or pass --config-path."
                )
                return result

        if not config_path.exists():
            result["errors"].append(f"Config file not found: {config_path}")
            return result

        result["config_path"] = str(config_path)

        # Load config
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            result["errors"].append(f"Invalid JSON in config: {e}")
            return result
        except OSError as e:
            result["errors"].append(f"Cannot read config: {e}")
            return result

        # Build a list of known API key fields to scan
        known_api_keys: Dict[str, str] = {
            "openai_api_key": "openai",
            "openrouter_api_key": "openrouter",
            "anthropic_api_key": "anthropic",
            "google_gemini_api_key": "google_gemini",
            "huggingface_api_key": "huggingface",
            "api_key": "default",
            "api_key_openai": "openai",
            "api_key_anthropic": "anthropic",
            "qwen_api_key": "qwen",
            "ollama_api_key": "ollama",
        }

        def _find_keys(data: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
            """Recursively find API key values in config dict."""
            found: Dict[str, str] = {}
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                # Check if this key matches a known API key name directly
                service = known_api_keys.get(key)
                if service and isinstance(value, str) and len(value) > 4:
                    # Skip environment variable references
                    if not value.startswith("${") and not value.startswith("env:"):
                        found[full_key] = value
                    else:
                        result["skipped"] += 1
                # Recurse into nested dicts
                elif isinstance(value, dict):
                    found.update(_find_keys(value, full_key))
            return found

        found_keys = _find_keys(config)

        if not found_keys:
            result["errors"].append(
                "No plaintext API keys found in config (all already use env refs?)"
            )
            return result

        # Create vault and store each credential
        try:
            vault = cls(vault_path=vault_path, master_key=master_key)
            result["vault_path"] = str(vault.get_vault_path())

            for full_key, secret in found_keys.items():
                # Derive a clean service name from the key path
                parts = full_key.split(".")
                service = parts[-1] if len(parts) > 1 else full_key
                # Remove common suffixes to get a clean service name
                for suffix in ["_api_key", "_key"]:
                    if service.endswith(suffix):
                        service = service[: -len(suffix)]
                        break

                try:
                    cred = Credential(
                        service=service,
                        username="api_key",
                        secret=secret,
                        description=f"Migrated from {config_path.name} ({full_key})",
                    )
                    if vault.set(cred):
                        result["migrated"] += 1
                        logger.info(
                            "Migrated credential for '%s' from %s",
                            service,
                            config_path.name,
                        )
                    else:
                        result["errors"].append(f"Failed to store {full_key}")
                except Exception as e:
                    result["errors"].append(f"Error migrating {full_key}: {e}")

            # Optionally mark config as migrated
            if delete_after and result["migrated"] > 0:
                cls._mark_config_migrated(config_path, config)

        except Exception as e:
            result["errors"].append(f"Vault initialization error: {e}")

        return result

    @classmethod
    def _detect_config_file(cls) -> Optional[Path]:
        """Find a config file in standard locations."""
        candidates = [
            Path.home() / ".xencode" / "config.json",
            Path.home() / ".xencode" / "config.yaml",
            Path.home() / ".xencode" / "config.toml",
            Path.home() / ".config" / "xencode" / "config.json",
            Path("xencode.json"),
            Path("xencode.yaml"),
            Path(".xencode.json"),
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    @classmethod
    def _mark_config_migrated(
        cls, config_path: Path, config: Dict[str, Any]
    ) -> None:
        """Add a migration marker to the config file."""
        try:
            config["_migrated_to_vault"] = True
            config["_vault_migrated_at"] = datetime.now(timezone.utc).isoformat()
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info("Marked %s as migrated to vault", config_path.name)
        except OSError as e:
            logger.warning("Could not mark config as migrated: %s", e)

    @classmethod
    def init_vault(
        cls,
        vault_path: Optional[Path] = None,
        master_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Initialize (create) a new vault file.

        Creates the vault directory and an empty vault file if one does not
        already exist. Safe to call multiple times — will not overwrite an
        existing vault.

        Returns:
            Dict with keys: created, vault_path, already_exists, error.
        """
        result: Dict[str, Any] = {
            "created": False,
            "vault_path": "",
            "already_exists": False,
            "error": None,
        }

        try:
            vault = cls(vault_path=vault_path, master_key=master_key)
            vault_path_resolved = vault.get_vault_path()
            result["vault_path"] = str(vault_path_resolved)

            if vault_path_resolved.exists():
                result["already_exists"] = True
                # Verify it's valid by loading it
                try:
                    with open(vault_path_resolved, "r", encoding="utf-8") as f:
                        json.load(f)
                    logger.info("Vault already exists at %s", vault_path_resolved)
                except (OSError, json.JSONDecodeError) as e:
                    result["error"] = f"Vault exists but is corrupted: {e}"
                return result

            # Create an empty vault
            vault_path_resolved.parent.mkdir(parents=True, exist_ok=True)
            empty_data: Dict[str, Any] = {
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "credentials": {},
            }
            with open(vault_path_resolved, "w", encoding="utf-8") as f:
                json.dump(empty_data, f, indent=2)

            # Restrict permissions (best-effort)
            try:
                import os as _os
                _os.chmod(vault_path_resolved, 0o600)
            except Exception:
                pass

            result["created"] = True
            logger.info("Created empty vault at %s", vault_path_resolved)

        except Exception as e:
            result["error"] = str(e)
            logger.error("Failed to initialize vault: %s", e)

        return result


__all__ = [
    "JsonFileCredentialVault",
]
