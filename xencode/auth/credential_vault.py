#!/usr/bin/env python3
"""
Credential Vault Backend for Xencode

Secure credential storage with:
- Windows Credential Manager backend (primary on Windows)
- File-based encrypted storage backend (cross-platform)
- Environment provider backend (fallback)
- Migration from plaintext config
- Abstracted interface for cross-platform support
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

try:
    import pywintypes
    import win32cred
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

try:
    import base64

    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from rich.console import Console

console = Console()


@dataclass
class Credential:
    """Credential data structure"""
    service: str
    username: str
    secret: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CredentialBackend(ABC):
    """Abstract base class for credential backends"""

    @abstractmethod
    def get_credential(self, service: str, username: str) -> Optional[Credential]:
        """Retrieve credential from vault"""
        pass

    @abstractmethod
    def set_credential(self, credential: Credential) -> bool:
        """Store credential in vault"""
        pass

    @abstractmethod
    def delete_credential(self, service: str, username: str) -> bool:
        """Delete credential from vault"""
        pass

    @abstractmethod
    def list_services(self) -> List[str]:
        """List all services with stored credentials"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass


class WindowsCredentialManagerBackend(CredentialBackend):
    """
    Windows Credential Manager backend

    Uses Windows Credential Manager via win32cred or keyring
    """

    SERVICE_PREFIX = "Xencode_"

    def __init__(self):
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if Windows Credential Manager is available"""
        if not sys.platform == "win32":
            return False

        # Try win32cred first (more direct)
        if WIN32_AVAILABLE:
            try:
                # Test by reading a non-existent credential
                win32cred.CredRead(
                    TargetName=f"{self.SERVICE_PREFIX}Test",
                    Type=win32cred.CRED_TYPE_GENERIC,
                )
            except pywintypes.error as e:
                # ERROR_NOT_FOUND is expected for non-existent credentials
                if e.winerror == 1168:  # ERROR_NOT_FOUND
                    return True
                # Other errors indicate Cred API not available
                return False
            except Exception:
                return False

        # Fallback to keyring
        if KEYRING_AVAILABLE:
            try:
                keyring.get_password(f"{self.SERVICE_PREFIX}Test", "test")
                return True
            except Exception:
                return False

        return False

    def is_available(self) -> bool:
        """Check if Windows Credential Manager is available"""
        return self._available

    def _make_target_name(self, service: str, username: str) -> str:
        """Create Windows Credential Manager target name"""
        return f"{self.SERVICE_PREFIX}{service}_{username}"

    def get_credential(self, service: str, username: str) -> Optional[Credential]:
        """Retrieve credential from Windows Credential Manager"""
        if not self._available:
            return None

        target_name = self._make_target_name(service, username)

        # Try win32cred first
        if WIN32_AVAILABLE:
            try:
                cred = win32cred.CredRead(
                    TargetName=target_name,
                    Type=win32cred.CRED_TYPE_GENERIC,
                )

                # Decode credential blob (stored as UTF-16)
                secret_blob = cred['CredentialBlob']
                if isinstance(secret_blob, bytes):
                    try:
                        secret = secret_blob.decode('utf-16').rstrip('\x00')
                    except UnicodeDecodeError:
                        secret = secret_blob.decode('utf-8', errors='replace')
                else:
                    secret = str(secret_blob)

                return Credential(
                    service=service,
                    username=username,
                    secret=secret,
                    description=cred.get('Comment'),
                    metadata={
                        'target_name': target_name,
                        'persist_type': cred.get('Persist'),
                        'last_written': cred.get('LastWritten'),
                    }
                )
            except pywintypes.error as e:
                if e.winerror != 1168:  # ERROR_NOT_FOUND
                    console.print(f"[yellow]Warning: Credential read error: {e}[/yellow]")
                return None
            except Exception as e:
                console.print(f"[yellow]Warning: Unexpected error: {e}[/yellow]")
                return None

        # Fallback to keyring
        if KEYRING_AVAILABLE:
            try:
                secret = keyring.get_password(f"{self.SERVICE_PREFIX}{service}", username)
                if secret:
                    return Credential(
                        service=service,
                        username=username,
                        secret=secret,
                    )
            except Exception as e:
                console.print(f"[yellow]Warning: Keyring error: {e}[/yellow]")

        return None

    def set_credential(self, credential: Credential) -> bool:
        """Store credential in Windows Credential Manager"""
        if not self._available:
            return False

        target_name = self._make_target_name(credential.service, credential.username)

        # Try win32cred first
        if WIN32_AVAILABLE:
            try:
                win32cred.CredWrite({
                    'Type': win32cred.CRED_TYPE_GENERIC,
                    'TargetName': target_name,
                    'CredentialBlob': credential.secret,
                    'Comment': credential.description or f"Xencode credential for {credential.service}",
                    'Persist': win32cred.CRED_PERSIST_LOCAL_MACHINE,
                })
                console.print(f"[green]OK: Stored credential for {credential.service}[/green]")
                return True
            except Exception as e:
                console.print(f"[red]FAIL: Failed to store credential: {e}[/red]")
                return False

        # Fallback to keyring
        if KEYRING_AVAILABLE:
            try:
                keyring.set_password(
                    f"{self.SERVICE_PREFIX}{credential.service}",
                    credential.username,
                    credential.secret,
                )
                console.print(f"[green]OK: Stored credential for {credential.service}[/green]")
                return True
            except Exception as e:
                console.print(f"[red]FAIL: Keyring error: {e}[/red]")
                return False

        return False

    def delete_credential(self, service: str, username: str) -> bool:
        """Delete credential from Windows Credential Manager"""
        if not self._available:
            return False

        target_name = self._make_target_name(service, username)

        # Try win32cred first
        if WIN32_AVAILABLE:
            try:
                win32cred.CredDelete(
                    TargetName=target_name,
                    Type=win32cred.CRED_TYPE_GENERIC,
                )
                console.print(f"[green]OK: Deleted credential for {service}[/green]")
                return True
            except pywintypes.error as e:
                if e.winerror == 1168:  # ERROR_NOT_FOUND
                    console.print(f"[yellow]Warning: Credential not found: {service}[/yellow]")
                    return False
                console.print(f"[red]FAIL: Failed to delete credential: {e}[/red]")
                return False
            except Exception as e:
                console.print(f"[red]FAIL: Unexpected error: {e}[/red]")
                return False

        # Fallback to keyring
        if KEYRING_AVAILABLE:
            try:
                keyring.delete_password(f"{self.SERVICE_PREFIX}{service}", username)
                console.print(f"[green]OK: Deleted credential for {service}[/green]")
                return True
            except Exception as e:
                console.print(f"[red]FAIL: Keyring error: {e}[/red]")
                return False

        return False

    def list_services(self) -> List[str]:
        """List all services with stored credentials"""
        if not self._available:
            return []

        services = set()

        # Try win32cred to enumerate credentials
        if WIN32_AVAILABLE:
            try:
                # Read all credentials
                creds = win32cred.CredEnumerate(None, 0)

                for cred in creds:
                    target_name = cred.get('TargetName', '')
                    if target_name.startswith(self.SERVICE_PREFIX):
                        # Extract service from target name
                        remainder = target_name[len(self.SERVICE_PREFIX):]
                        if '_' in remainder:
                            service = remainder.split('_')[0]
                            services.add(service)
            except pywintypes.error as e:
                if e.winerror != 1168:  # ERROR_NOT_FOUND is OK (no credentials)
                    console.print(f"[yellow]Warning: Error enumerating credentials: {e}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Warning: Unexpected error: {e}[/yellow]")

        return list(services)


class EnvironmentBackend(CredentialBackend):
    """
    Environment variable backend (fallback)

    Stores credentials in environment variables (less secure, for development)
    """

    PREFIX = "XENCODE_"

    def __init__(self):
        self._storage: Dict[str, Credential] = {}

    def is_available(self) -> bool:
        """Environment backend is always available"""
        return True

    def _make_env_name(self, service: str, username: str) -> str:
        """Create environment variable name"""
        return f"{self.PREFIX}{service.upper()}_{username.upper()}"

    def get_credential(self, service: str, username: str) -> Optional[Credential]:
        """Retrieve credential from environment"""
        # Check in-memory storage first
        key = f"{service}:{username}"
        if key in self._storage:
            return self._storage[key]

        # Check environment variables
        env_name = self._make_env_name(service, username)
        secret = os.environ.get(env_name)

        if secret:
            return Credential(
                service=service,
                username=username,
                secret=secret,
                metadata={'source': 'environment'},
            )

        return None

    def set_credential(self, credential: Credential) -> bool:
        """Store credential in memory (and optionally environment)"""
        key = f"{credential.service}:{credential.username}"
        self._storage[key] = credential

        # Also set in environment for compatibility
        env_name = self._make_env_name(credential.service, credential.username)
        os.environ[env_name] = credential.secret

        console.print(f"[green]OK: Stored credential for {credential.service} (environment)[/green]")
        return True

    def delete_credential(self, service: str, username: str) -> bool:
        """Delete credential from memory and environment"""
        key = f"{service}:{username}"

        # Remove from memory
        if key in self._storage:
            del self._storage[key]

        # Remove from environment
        env_name = self._make_env_name(service, username)
        if env_name in os.environ:
            del os.environ[env_name]

        console.print(f"[green]OK: Deleted credential for {service}[/green]")
        return True

    def list_services(self) -> List[str]:
        """List all services with stored credentials"""
        services = set()

        # Check memory storage
        for key in self._storage.keys():
            service = key.split(':')[0]
            services.add(service)

        # Check environment variables
        for env_name in os.environ.keys():
            if env_name.startswith(self.PREFIX):
                remainder = env_name[len(self.PREFIX):]
                if '_' in remainder:
                    service = remainder.split('_')[0]
                    services.add(service.lower())

        return list(services)


class FileBasedCredentialBackend(CredentialBackend):
    """
    File-based encrypted credential backend (cross-platform)

    Encrypts credentials using Fernet (AES-128-CBC with HMAC) and stores
    them in a JSON file on disk. Works on all platforms.

    Security model:
    - Encryption key derived from a machine-specific seed using PBKDF2
    - Each credential encrypted individually with Fernet
    - Vault file stored at ~/.xencode/vault.json with restricted permissions
    - Falls back to base64 obfuscation if cryptography is not installed
    """

    DEFAULT_VAULT_DIR = ".xencode"
    DEFAULT_VAULT_FILE = "vault.json"

    def __init__(
        self,
        vault_path: Optional[Path] = None,
        master_key: Optional[str] = None,
    ):
        """
        Initialize file-based credential backend.

        Args:
            vault_path: Path to vault file (default: ~/.xencode/vault.json)
            master_key: Optional master key for encryption. If not provided,
                       a key is derived from machine-specific identifiers.
        """
        self._vault_path = vault_path or (Path.home() / self.DEFAULT_VAULT_DIR / self.DEFAULT_VAULT_FILE)
        self._fernet: Optional[Any] = None
        self._plaintext_fallback = False

        # Initialize encryption
        self._init_encryption(master_key)

        # Ensure vault directory exists
        self._vault_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory cache (loaded from disk on first access)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._loaded = False

    def _init_encryption(self, master_key: Optional[str] = None) -> None:
        """Initialize Fernet encryption or fallback"""
        if CRYPTO_AVAILABLE:
            try:
                # Derive a stable key from machine-specific identifiers
                seed = master_key or self._get_machine_seed()
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b"xencode-vault-salt",
                    iterations=600_000,
                    backend=default_backend(),
                )
                key = base64.urlsafe_b64encode(kdf.derive(seed.encode("utf-8")))
                self._fernet = Fernet(key)
            except Exception:
                self._plaintext_fallback = True
        else:
            self._plaintext_fallback = True

    def _get_machine_seed(self) -> str:
        """
        Derive a stable machine-specific seed for key derivation.

        Combines OS-level identifiers to create a consistent per-machine key.
        """
        seed_parts = []

        # Machine ID (Linux/macOS)
        machine_id_path = Path("/etc/machine-id")
        if machine_id_path.exists():
            seed_parts.append(machine_id_path.read_text().strip())
        else:
            # Windows: use MachineGUID from registry
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                    guid, _ = winreg.QueryValueEx(key, "MachineGuid")
                    seed_parts.append(guid)
            except Exception:
                pass

        # Fallback: hostname + home directory
        seed_parts.append(os.environ.get("COMPUTERNAME", "") or os.uname().nodename if hasattr(os, "uname") else "")
        seed_parts.append(str(Path.home()))
        seed_parts.append(os.environ.get("USER", "") or os.environ.get("USERNAME", ""))

        return "|".join(seed_parts)

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt a string"""
        if self._fernet:
            return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")
        # Plaintext fallback: base64 encode (obfuscation only, not real encryption)
        return f"b64:{base64.b64encode(plaintext.encode('utf-8')).decode('utf-8')}"

    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt a string"""
        if self._fernet:
            try:
                return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
            except Exception:
                raise  # Let the caller handle decryption failures
        # Plaintext fallback
        if ciphertext.startswith("b64:"):
            try:
                return base64.b64decode(ciphertext[4:]).decode("utf-8")
            except Exception:
                raise
        return ciphertext

    def _load_vault(self) -> Dict[str, Dict[str, Any]]:
        """Load vault data from disk"""
        if self._loaded:
            return self._cache

        self._loaded = True
        if not self._vault_path.exists():
            self._cache = {}
            return self._cache

        try:
            with open(self._vault_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._cache = raw.get("credentials", {})
        except (json.JSONDecodeError, OSError):
            self._cache = {}

        return self._cache

    def _save_vault(self) -> bool:
        """Save vault data to disk"""
        try:
            data = {
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "credentials": self._cache,
            }
            with open(self._vault_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Restrict file permissions (best effort)
            try:
                os.chmod(self._vault_path, 0o600)
            except Exception:
                pass  # Permission changes may fail on some platforms

            self._dirty = False
            return True
        except OSError as e:
            console.print(f"[red]FAIL: Failed to save vault: {e}[/red]")
            return False

    def _make_key(self, service: str, username: str) -> str:
        """Create internal storage key"""
        return f"{service}:{username}"

    def is_available(self) -> bool:
        """File-based backend is always available (can always write to disk)"""
        return True

    def is_encrypted(self) -> bool:
        """Check if credentials are actually encrypted (vs plaintext fallback)"""
        return self._fernet is not None

    def get_vault_path(self) -> Path:
        """Get the vault file path"""
        return self._vault_path

    def get_credential(self, service: str, username: str) -> Optional[Credential]:
        """Retrieve credential from encrypted file storage"""
        vault = self._load_vault()
        key = self._make_key(service, username)
        entry = vault.get(key)

        if not entry:
            return None

        try:
            secret = self._decrypt(entry["secret"])
            return Credential(
                service=service,
                username=username,
                secret=secret,
                description=entry.get("description"),
                metadata={
                    **(entry.get("metadata", {})),
                    "source": "file_vault",
                    "vault_path": str(self._vault_path),
                },
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to decrypt credential for {service}: {e}[/yellow]")
            return None

    def set_credential(self, credential: Credential) -> bool:
        """Store credential in encrypted file storage"""
        vault = self._load_vault()
        key = self._make_key(credential.service, credential.username)

        # Encrypt the secret
        encrypted_secret = self._encrypt(credential.secret)

        vault[key] = {
            "secret": encrypted_secret,
            "description": credential.description or "",
            "metadata": credential.metadata or {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        self._dirty = True
        result = self._save_vault()

        if result:
            enc_status = "encrypted" if self._fernet else "base64 encoded"
            console.print(f"[green]OK: Stored credential for {credential.service} ({enc_status})[/green]")
        return result

    def delete_credential(self, service: str, username: str) -> bool:
        """Delete credential from file storage"""
        vault = self._load_vault()
        key = self._make_key(service, username)

        if key in vault:
            del vault[key]
            self._dirty = True
            result = self._save_vault()
            if result:
                console.print(f"[green]OK: Deleted credential for {service} from file vault[/green]")
            return result

        console.print(f"[yellow]Warning: Credential not found: {service}[/yellow]")
        return False

    def list_services(self) -> List[str]:
        """List all services with stored credentials"""
        vault = self._load_vault()
        services = set()

        for key in vault:
            service = key.split(":")[0]
            services.add(service)

        return list(services)

    def clear_vault(self) -> bool:
        """Delete all credentials from the vault file"""
        self._cache = {}
        self._dirty = True
        result = self._save_vault()
        if result:
            console.print("[green]OK: Vault cleared[/green]")
        return result

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the file-based storage"""
        vault = self._load_vault()
        return {
            "vault_path": str(self._vault_path),
            "credential_count": len(vault),
            "vault_exists": self._vault_path.exists(),
            "encryption": "Fernet (AES-128)" if self._fernet else "base64 (obfuscation only)",
            "services": self.list_services(),
        }


class CredentialVault:
    """
    Main credential vault interface

    Provides unified access to credentials with automatic backend selection
    and migration support.

    Usage:
        vault = CredentialVault()
        cred = vault.get("openai", "api_key")
        vault.set(Credential(service="openai", username="api_key", secret="sk-..."))
    """

    # Known services
    SERVICE_QWEN = "qwen"
    SERVICE_OPENROUTER = "openrouter"
    SERVICE_OPENAI = "openai"
    SERVICE_ANTHROPIC = "anthropic"
    SERVICE_OLLAMA = "ollama"  # Usually no credentials needed

    def __init__(self, prefer_windows: bool = True, vault_path: Optional[Path] = None):
        """
        Initialize credential vault

        Args:
            prefer_windows: If True, prefer Windows Credential Manager
            vault_path: Optional custom path for file-based vault (default: ~/.xencode/vault.json)
        """
        self.backends: List[CredentialBackend] = []

        # Initialize backends in priority order

        # 1. Windows Credential Manager (highest security, Windows only)
        if prefer_windows and sys.platform == "win32":
            windows_backend = WindowsCredentialManagerBackend()
            if windows_backend.is_available():
                self.backends.append(windows_backend)
                console.print("[green]OK: Windows Credential Manager available[/green]")

        # 2. File-based encrypted storage (cross-platform, persistent)
        file_backend = FileBasedCredentialBackend(vault_path=vault_path)
        self.backends.append(file_backend)

        if file_backend.is_encrypted():
            console.print("[green]OK: Encrypted file vault ready[/green]")
        else:
            console.print("[yellow]Warning: cryptography not available — file vault using base64 encoding only (less secure)[/yellow]")
            console.print("[yellow]  Install with: pip install cryptography[/yellow]")

        # 3. Environment backend (always last, fallback for development)
        env_backend = EnvironmentBackend()
        self.backends.append(env_backend)

        if not self.backends:
            console.print("[yellow]Warning: No credential backends available[/yellow]")

    def get(self, service: str, username: str) -> Optional[Credential]:
        """
        Get credential from vault

        Args:
            service: Service name (e.g., "openai", "qwen")
            username: Username/identifier (e.g., "api_key", "access_token")

        Returns:
            Credential or None if not found
        """
        for backend in self.backends:
            cred = backend.get_credential(service, username)
            if cred:
                return cred

        return None

    def get_secret(self, service: str, username: str) -> Optional[str]:
        """
        Get secret value from vault

        Args:
            service: Service name
            username: Username/identifier

        Returns:
            Secret string or None
        """
        cred = self.get(service, username)
        return cred.secret if cred else None

    def set(self, credential: Credential) -> bool:
        """
        Store credential in vault

        Args:
            credential: Credential to store

        Returns:
            True if successful
        """
        # Store in first available backend (highest priority)
        for backend in self.backends:
            if backend.is_available():
                return backend.set_credential(credential)

        return False

    def delete(self, service: str, username: str) -> bool:
        """
        Delete credential from vault

        Args:
            service: Service name
            username: Username/identifier

        Returns:
            True if deleted
        """
        success = False

        # Delete from all backends
        for backend in self.backends:
            if backend.delete_credential(service, username):
                success = True

        return success

    def list_services(self) -> List[str]:
        """List all services with stored credentials"""
        services = set()

        for backend in self.backends:
            for service in backend.list_services():
                services.add(service)

        return list(services)

    def has_credentials(self) -> bool:
        """Check if any credentials are stored"""
        return len(self.list_services()) > 0

    def migrate_from_config(
        self,
        config_path: Optional[Path] = None,
        delete_after: bool = False,
    ) -> Dict[str, Any]:
        """
        Migrate credentials from plaintext config file to vault

        Args:
            config_path: Path to config file (default: ~/.xencode/config.json)
            delete_after: If True, delete plaintext credentials after migration

        Returns:
            Migration result with counts and errors
        """
        if config_path is None:
            config_path = Path.home() / ".xencode" / "config.json"

        result = {
            'migrated': 0,
            'failed': 0,
            'errors': [],
        }

        if not config_path.exists():
            result['errors'].append(f"Config file not found: {config_path}")
            return result

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Extract credentials from config
            providers = config.get('providers', {})

            for provider_name, provider_config in providers.items():
                api_key = provider_config.get('api_key')

                if api_key and api_key.startswith('${'):
                    # Environment variable reference, skip
                    continue

                if api_key:
                    try:
                        credential = Credential(
                            service=provider_name,
                            username='api_key',
                            secret=api_key,
                            description=f"Migrated from {config_path}",
                        )

                        if self.set(credential):
                            result['migrated'] += 1

                            # Optionally remove from config
                            if delete_after:
                                provider_config['api_key'] = f"${{{provider_name.upper()}_API_KEY}}"
                        else:
                            result['failed'] += 1
                            result['errors'].append(f"Failed to migrate {provider_name}")
                    except Exception as e:
                        result['failed'] += 1
                        result['errors'].append(f"Error migrating {provider_name}: {e}")

            # Save updated config if deleting plaintext credentials
            if delete_after and result['migrated'] > 0:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)

                console.print(f"[green]OK: Migrated {result['migrated']} credentials from config[/green]")

        except json.JSONDecodeError as e:
            result['errors'].append(f"Invalid JSON in config: {e}")
        except Exception as e:
            result['errors'].append(f"Migration error: {e}")

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get vault status summary"""
        return {
            'backends': [type(b).__name__ for b in self.backends],
            'services': self.list_services(),
            'has_credentials': self.has_credentials(),
            'primary_backend': type(self.backends[0]).__name__ if self.backends else None,
        }


# Global vault instance
_vault: Optional[CredentialVault] = None


def get_vault(prefer_windows: bool = True, vault_path: Optional[Path] = None) -> CredentialVault:
    """Get or create global credential vault"""
    global _vault
    if _vault is None:
        _vault = CredentialVault(prefer_windows=prefer_windows, vault_path=vault_path)
    return _vault


# Convenience functions
def get_credential(service: str, username: str) -> Optional[Credential]:
    """Get credential from global vault"""
    return get_vault().get(service, username)


def get_secret(service: str, username: str) -> Optional[str]:
    """Get secret from global vault"""
    return get_vault().get_secret(service, username)


def set_credential(credential: Credential) -> bool:
    """Store credential in global vault"""
    return get_vault().set(credential)


def delete_credential(service: str, username: str) -> bool:
    """Delete credential from global vault"""
    return get_vault().delete(service, username)


if __name__ == "__main__":
    # Credential Vault - Run with --demo flag for testing
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        async def demo():
            console.print("[bold blue]Credential Vault Demo[/bold blue]\n")

            vault = CredentialVault()

            # Show status
            status = vault.get_status()
            console.print("[bold]Vault Status:[/bold]")
            console.print(f"  Backends: {', '.join(status['backends'])}")
            console.print(f"  Services: {status['services'] or 'None'}")
            console.print(f"  Has credentials: {status['has_credentials']}")

            # Show file vault info if available
            for backend in vault.backends:
                if isinstance(backend, FileBasedCredentialBackend):
                    info = backend.get_storage_info()
                    console.print(f"  Vault path: {info['vault_path']}")
                    console.print(f"  Encryption: {info['encryption']}")

            # Demo: Store and retrieve a credential
            console.print("\n[bold]Testing credential storage...[/bold]")

            test_cred = Credential(
                service="demo_service",
                username="demo_user",
                secret="demo_secret_value",
                description="Demo credential",
            )

            if vault.set(test_cred):
                console.print("[green]OK: Credential stored[/green]")

                retrieved = vault.get("demo_service", "demo_user")
                if retrieved:
                    console.print("[green]OK: Credential retrieved[/green]")

                # Clean up
                vault.delete("demo_service", "demo_user")
                console.print("[green]OK: Demo credential deleted[/green]")

        import asyncio
        asyncio.run(demo())
    else:
        print("Credential Vault module")
        print("Usage: python -m xencode.auth.credential_vault --demo")
