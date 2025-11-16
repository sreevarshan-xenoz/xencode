"""Environment variable resolver for configuration."""

import os
import re
import subprocess
from typing import Optional


class ResolverError(Exception):
    """Environment variable resolution error."""
    pass


class EnvironmentResolver:
    """Resolves environment variables and shell commands in configuration values."""
    
    # Regex patterns for variable expansion
    SIMPLE_VAR_PATTERN = re.compile(r'\$([A-Za-z_][A-Za-z0-9_]*)')
    BRACED_VAR_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')
    COMMAND_PATTERN = re.compile(r'\$\(([^)]+)\)')
    
    def __init__(self, allow_command_expansion: bool = True):
        """Initialize environment resolver.
        
        Args:
            allow_command_expansion: Whether to allow $(command) expansion
        """
        self.allow_command_expansion = allow_command_expansion
    
    def resolve(self, value: str) -> str:
        """Resolve environment variables and commands in a string.
        
        Supports:
        - $VAR - Simple variable expansion
        - ${VAR} - Braced variable expansion
        - $(command) - Command substitution (if enabled)
        
        Args:
            value: String value to resolve
            
        Returns:
            Resolved string value
            
        Raises:
            ResolverError: If resolution fails
        """
        if not isinstance(value, str):
            return value
        
        # First resolve command substitutions
        if self.allow_command_expansion:
            value = self._resolve_commands(value)
        
        # Then resolve braced variables ${VAR}
        value = self._resolve_braced_vars(value)
        
        # Finally resolve simple variables $VAR
        value = self._resolve_simple_vars(value)
        
        return value
    
    def _resolve_simple_vars(self, value: str) -> str:
        """Resolve simple $VAR style variables.
        
        Args:
            value: String to resolve
            
        Returns:
            String with resolved variables
        """
        def replace_var(match):
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                # Leave unresolved if variable doesn't exist
                return match.group(0)
            return env_value
        
        return self.SIMPLE_VAR_PATTERN.sub(replace_var, value)
    
    def _resolve_braced_vars(self, value: str) -> str:
        """Resolve braced ${VAR} style variables.
        
        Args:
            value: String to resolve
            
        Returns:
            String with resolved variables
        """
        def replace_var(match):
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                # Leave unresolved if variable doesn't exist
                return match.group(0)
            return env_value
        
        return self.BRACED_VAR_PATTERN.sub(replace_var, value)
    
    def _resolve_commands(self, value: str) -> str:
        """Resolve $(command) style command substitutions.
        
        Args:
            value: String to resolve
            
        Returns:
            String with resolved commands
            
        Raises:
            ResolverError: If command execution fails
        """
        def replace_command(match):
            command = match.group(1)
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True
                )
                # Strip trailing newline
                return result.stdout.rstrip('\n')
            except subprocess.TimeoutExpired:
                raise ResolverError(
                    f"Command timed out: {command}"
                )
            except subprocess.CalledProcessError as e:
                raise ResolverError(
                    f"Command failed with exit code {e.returncode}: {command}\n"
                    f"stderr: {e.stderr}"
                )
            except Exception as e:
                raise ResolverError(
                    f"Failed to execute command: {command}\n"
                    f"Error: {e}"
                )
        
        return self.COMMAND_PATTERN.sub(replace_command, value)
    
    def validate_resolved(self, value: str, field_name: str) -> None:
        """Validate that a value has been fully resolved.
        
        Checks for any remaining unresolved variables or commands.
        
        Args:
            value: Resolved value to validate
            field_name: Name of the field (for error messages)
            
        Raises:
            ResolverError: If value contains unresolved variables
        """
        if not isinstance(value, str):
            return
        
        # Check for unresolved variables
        if self.SIMPLE_VAR_PATTERN.search(value):
            raise ResolverError(
                f"Unresolved environment variable in {field_name}: {value}"
            )
        
        if self.BRACED_VAR_PATTERN.search(value):
            raise ResolverError(
                f"Unresolved environment variable in {field_name}: {value}"
            )
        
        if self.allow_command_expansion and self.COMMAND_PATTERN.search(value):
            raise ResolverError(
                f"Unresolved command substitution in {field_name}: {value}"
            )
    
    def get_env(self, var_name: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value.
        
        Args:
            var_name: Environment variable name
            default: Default value if variable is not set
            
        Returns:
            Environment variable value or default
        """
        return os.environ.get(var_name, default)
    
    def set_env(self, var_name: str, value: str) -> None:
        """Set environment variable.
        
        Args:
            var_name: Environment variable name
            value: Value to set
        """
        os.environ[var_name] = value
    
    def has_env(self, var_name: str) -> bool:
        """Check if environment variable is set.
        
        Args:
            var_name: Environment variable name
            
        Returns:
            True if variable is set, False otherwise
        """
        return var_name in os.environ
