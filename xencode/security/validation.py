"""
Security utilities module for Xencode
Provides input sanitization, validation, and other security measures
"""
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from rich.console import Console
from rich.panel import Panel

console = Console()

# Dangerous patterns that should be filtered out
DANGEROUS_PATTERNS = [
    r"(?i)(system|exec|eval|import|open|read|write|delete|remove|unlink|chmod|chown)\s*\(",
    r"(?i)(rm\s+-rf|rm\s+--no-preserve-root|mkfs|dd|shred)\s+",
    r"(?i)(sudo|su|doas)\s+",
    r"(?i)(cat|echo|printf)\s+/etc/",
    r"(?i)(curl|wget|fetch)\s+--(insecure|no-check-certificate)",
    r"(?i)(bash|sh|zsh|fish)\s+-c\s+",
    r"(?i)(python|perl|ruby|php|node)\s+(-c|--command)\s+",
    r"(?i)(alias|export|set)\s+[A-Z_]+\s*=",
    r"(?i)(crontab|at)\s+",
    r"(?i)(mount|umount|swapon|swapoff)\s+",
    r"(?i)(kill|killall|pkill|killpg)\s+",
    r"(?i)(iptables|firewall-cmd|ufw)\s+",
    r"(?i)(passwd|shadow|group)\s+",
    r"(?i)(ssh|scp|rsync)\s+",
    r"(?i)(nc|netcat|socat|telnet)\s+",
    r"(?i)(tcpdump|wireshark|tshark)\s+",
    r"(?i)(strace|ltrace|gdb)\s+",
    r"(?i)(chattr|lsattr)\s+",
    r"(?i)(chroot|pivot_root)\s+",
    r"(?i)(insmod|rmmod|lsmod|modprobe)\s+",
    r"(?i)(semanage|setsebool|restorecon)\s+",
    r"(?i)(auditctl|ausearch|autrace)\s+",
    r"(?i)(sysctl|insmod|rmmod)\s+",
    r"(?i)(>/dev/null|2>&1|&)\s*$",
    r"(?i)(\|\||&&)\s+",
    r"(?i)(\$\(|`).*(`|\))",
    r"(?i)(\$\{.*\})",
    r"(?i)(\$\w+)",
    r"(?i)(\${.*})",
    r"(?i)(\\n|\\r|\\t|\\\\)",
    r"(?i)(\.\./)+",
    r"(?i)(\.\.\\)+",
    r"(?i)(~/.*)",
    r"(?i)(/\s*[a-zA-Z]:)",  # Windows drive paths in Unix context
    r"(?i)([a-zA-Z]:\\s*\\)",  # Windows paths
    r"(?i)(\\\\.*\\\\)",  # UNC paths
    r"(?i)(COM\d|LPT\d|CON|PRN|AUX|NUL)",  # Windows reserved names
    r"(?i)(<|>|;|`|\\|\||&)",  # Shell metacharacters
    r"(?i)(\$\(|`).*?(;|`|\))",  # Command substitution with semicolon
    r"(?i)(\$\{.*?};)",  # Variable expansion with semicolon
    r"(?i)(\$\w+;)",  # Variable reference with semicolon
    r"(?i)(\${.*?};)",  # Variable reference with semicolon
    r"(?i)(\$\((.*?)\);)",  # Command substitution with semicolon
]

class InputValidator:
    """Class to handle input validation and sanitization"""
    
    @staticmethod
    def sanitize_input(input_text: str) -> str:
        """
        Sanitize user input by removing dangerous patterns
        
        Args:
            input_text: The raw input text to sanitize
            
        Returns:
            Sanitized input text
        """
        if not input_text:
            return input_text
            
        # Remove dangerous patterns
        sanitized = input_text
        for pattern in DANGEROUS_PATTERNS:
            sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)
            
        return sanitized
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """
        Validate file path to prevent directory traversal and other attacks
        
        Args:
            file_path: The file path to validate
            
        Returns:
            True if the path is valid, False otherwise
        """
        if not file_path:
            return False
            
        # Normalize the path
        import os
        normalized_path = os.path.normpath(file_path)
        
        # Check for directory traversal attempts
        if '..' in normalized_path.replace('\\', '/').split('/'):
            return False
            
        # Check for absolute paths that might be dangerous
        if os.path.isabs(normalized_path):
            # Allow only paths within safe directories
            home_dir = os.path.expanduser("~")
            if not normalized_path.startswith(home_dir):
                # Check if it's in the current working directory
                cwd = os.getcwd()
                if not normalized_path.startswith(cwd):
                    return False
                    
        return True
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL to prevent SSRF and other attacks
        
        Args:
            url: The URL to validate
            
        Returns:
            True if the URL is valid, False otherwise
        """
        if not url:
            return False
            
        try:
            parsed = urlparse(url)
            if not parsed.scheme or parsed.scheme not in ['http', 'https']:
                return False
                
            # Block private IP ranges to prevent SSRF
            hostname = parsed.hostname
            if hostname:
                # Check for private IP addresses
                if hostname.startswith(('10.', '172.', '192.168.')):
                    return False
                # Check for localhost variations
                if hostname in ['localhost', '127.0.0.1', '::1']:
                    return False
                # Check for internal hostnames
                if hostname.endswith(('.internal', '.local', '.lan')):
                    return False
                    
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_model_name(model_name: str) -> bool:
        """
        Validate model name to prevent injection attacks
        
        Args:
            model_name: The model name to validate
            
        Returns:
            True if the model name is valid, False otherwise
        """
        if not model_name:
            return False
            
        # Only allow alphanumeric characters, dots, colons, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9._:-]+$', model_name):
            return False
            
        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, model_name, re.IGNORECASE):
                return False
                
        return True


def sanitize_user_input(user_input: str) -> str:
    """
    Convenience function to sanitize user input
    
    Args:
        user_input: Raw user input to sanitize
        
    Returns:
        Sanitized user input
    """
    validator = InputValidator()
    return validator.sanitize_input(user_input)


def validate_file_operation(file_path: str, operation: str = "read") -> bool:
    """
    Validate file operations to prevent unauthorized access
    
    Args:
        file_path: Path to the file
        operation: Type of operation ('read', 'write', 'delete')
        
    Returns:
        True if the operation is valid, False otherwise
    """
    validator = InputValidator()
    
    # Validate the file path
    if not validator.validate_file_path(file_path):
        return False
        
    # Additional checks based on operation
    import os
    if operation == "read":
        # For read operations, check if file exists and is readable
        return os.path.isfile(file_path) and os.access(file_path, os.R_OK)
    elif operation == "write":
        # For write operations, check if directory is writable
        directory = os.path.dirname(file_path) or os.getcwd()
        return os.access(directory, os.W_OK)
    elif operation == "delete":
        # For delete operations, check if file exists and is writable
        return os.path.isfile(file_path) and os.access(file_path, os.W_OK)
    
    return True