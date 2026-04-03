#!/usr/bin/env python3
"""
Credential Vault Backend for Xencode

Secure credential storage with:
- Windows Credential Manager backend (primary)
- Environment provider backend (fallback)
- Migration from plaintext config
- Abstracted interface for cross-platform support
"""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

try:
    import win32cred
    import pywintypes
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

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
    
    def __init__(self, prefer_windows: bool = True):
        """
        Initialize credential vault
        
        Args:
            prefer_windows: If True, prefer Windows Credential Manager
        """
        self.backends: List[CredentialBackend] = []
        
        # Initialize backends in priority order
        if prefer_windows and sys.platform == "win32":
            windows_backend = WindowsCredentialManagerBackend()
            if windows_backend.is_available():
                self.backends.append(windows_backend)
                console.print("[green]OK: Windows Credential Manager available[/green]")
        
        # Always add environment backend as fallback
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
        import json
        
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


def get_vault(prefer_windows: bool = True) -> CredentialVault:
    """Get or create global credential vault"""
    global _vault
    if _vault is None:
        _vault = CredentialVault(prefer_windows=prefer_windows)
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
            console.print(f"[bold]Vault Status:[/bold]")
            console.print(f"  Backends: {', '.join(status['backends'])}")
            console.print(f"  Services: {status['services'] or 'None'}")
            console.print(f"  Has credentials: {status['has_credentials']}")
            
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
                    console.print(f"[green]OK: Credential retrieved[/green]")
                
                # Clean up
                vault.delete("demo_service", "demo_user")
                console.print("[green]OK: Demo credential deleted[/green]")
        
        import asyncio
        asyncio.run(demo())
    else:
        print("Credential Vault module")
        print("Usage: python -m xencode.auth.credential_vault --demo")
