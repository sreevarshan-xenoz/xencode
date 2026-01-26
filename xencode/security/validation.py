"""
Input validation utilities for Xencode
Provides security validation and sanitization for user inputs
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from urllib.parse import urlparse


def validate_file_path(file_path: str, allowed_base_paths: Optional[List[Path]] = None) -> bool:
    """
    Validate file path to prevent directory traversal attacks.
    
    Args:
        file_path: Path to validate
        allowed_base_paths: List of allowed base paths for file access
        
    Returns:
        True if the path is valid, False otherwise
    """
    try:
        # Convert to Path object
        path = Path(file_path)
        
        # Resolve the path to its absolute form
        resolved_path = path.resolve()
        
        # Check for directory traversal
        if ".." in path.parts or str(resolved_path) != str(path.resolve()):
            # Double-check by comparing with original path
            if ".." in str(path) or "../" in str(path):
                return False
        
        # If allowed base paths are specified, check if the resolved path is within them
        if allowed_base_paths:
            for base_path in allowed_base_paths:
                try:
                    resolved_path.relative_to(base_path.resolve())
                    return True  # Path is within allowed base path
                except ValueError:
                    continue  # Path is not within this base path
            return False  # Path is not within any allowed base path
        
        return True
    except Exception:
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent malicious file operations.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Remove dangerous characters and sequences
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Prevent directory traversal
    sanitized = sanitized.replace('../', '').replace('..\\', '')
    
    # Limit length to prevent buffer overflow attempts
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized


def validate_model_name(model_name: str) -> bool:
    """
    Validate model name to prevent injection attacks.
    
    Args:
        model_name: Model name to validate
        
    Returns:
        True if the model name is valid, False otherwise
    """
    # Allow alphanumeric characters, hyphens, underscores, colons, dots, and slashes
    # This supports model names like "llama2:7b", "openai/gpt-4", etc.
    pattern = r'^[a-zA-Z0-9_\-:./]+$'
    return bool(re.match(pattern, model_name))


def validate_prompt(prompt: str, max_length: int = 10000) -> bool:
    """
    Validate prompt to prevent excessively long inputs or injection attempts.

    Args:
        prompt: Prompt to validate
        max_length: Maximum allowed length for the prompt

    Returns:
        True if the prompt is valid, False otherwise
    """
    if not isinstance(prompt, str):
        return False

    if len(prompt) > max_length:
        return False

    # Check for potential injection patterns
    injection_patterns = [
        r'\$\(.*\)',  # Command substitution
        r'`.*`',       # Backtick command execution
        r';\s*\w+',    # Semicolon followed by command
        r'&&\s*\w+',   # Logical AND followed by command
        r'\|\s*\w+',   # Pipe followed by command
        r'<script[^>]*>',  # Potential XSS
        r'eval\s*\(',  # JavaScript eval
        r'exec\s*\(',  # Python exec
        r'import\s+\w+',  # Python import statements
        r'os\.\w+',    # OS module access
        r'subprocess\.\w+',  # Subprocess module access
        r'__import__', # Import magic method
        r'globals\(\)', # Globals access
        r'locals\(\)',  # Locals access
        r'compile\(',   # Code compilation
        r'open\([^)]*["\']\w+["\']\s*,\s*["\'][rwax][tb]?',  # File operations
    ]

    for pattern in injection_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return False

    return True


def detect_prompt_injection(prompt: str) -> List[str]:
    """
    Detect potential prompt injection techniques in the input.

    Args:
        prompt: Prompt to analyze

    Returns:
        List of detected injection patterns
    """
    injection_indicators = []

    # Patterns that suggest prompt injection attempts
    injection_patterns = {
        'command_separation': r'[;&|]',
        'script_tags': r'<script[^>]*>.*?</script>',
        'html_entities': r'&#?\w+;',
        'javascript_urls': r'javascript:',
        'data_urls': r'data:text/',
        'css_expressions': r'expression\s*\(',
        'sql_keywords': r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|OR|AND)\b',
        'shell_commands': r'\$\(.*\)|`.*`',
        'programming_constructs': r'eval\(|exec\(|import\s+\w+|__import__|globals\(\)|locals\(\)',
        'file_operations': r'open\(|os\.\w+|subprocess\.',
        'escape_sequences': r'\\[nrtbfav\\\'\"0-7]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}',
        'format_strings': r'%[sdxfgo]',
        'template_syntax': r'\{\{.*?\}\}|\{\%.*?\%\}',
    }

    for name, pattern in injection_patterns.items():
        if re.search(pattern, prompt, re.IGNORECASE):
            injection_indicators.append(name)

    return injection_indicators


def validate_config_setting(setting_name: str, setting_value: Any) -> bool:
    """
    Validate configuration settings to prevent unsafe configurations.

    Args:
        setting_name: Name of the setting
        setting_value: Value of the setting

    Returns:
        True if the configuration is valid, False otherwise
    """
    # Block dangerous configuration options
    dangerous_settings = [
        'exec_mode',
        'eval_enabled',
        'unsafe_imports',
        'disable_security',
        'bypass_validation',
        'allow_arbitrary_code_execution'
    ]

    if setting_name.lower() in dangerous_settings:
        return False

    # Validate setting values based on expected types
    if setting_name.endswith('_path'):
        # Validate file paths
        if isinstance(setting_value, str):
            return validate_file_path(setting_value)
        else:
            return False

    if setting_name.endswith('_url') or setting_name.endswith('_endpoint'):
        # Validate URLs
        if isinstance(setting_value, str):
            return validate_url(setting_value)
        else:
            return False

    return True


def validate_api_request(data: Dict[str, Any], allowed_fields: List[str],
                        required_fields: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate API request data against allowed fields.

    Args:
        data: Request data to validate
        allowed_fields: List of allowed field names
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if required_fields:
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

    for field in data:
        if field not in allowed_fields:
            errors.append(f"Unauthorized field: {field}")

    # Validate individual field values
    for field, value in data.items():
        if field in allowed_fields:
            # Apply specific validation based on field name
            if 'model' in field.lower():
                if not validate_model_name(str(value)):
                    errors.append(f"Invalid model name in field {field}")
            elif 'prompt' in field.lower() or 'input' in field.lower():
                if not validate_prompt(str(value)):
                    errors.append(f"Invalid prompt in field {field}")
            elif 'file' in field.lower() or 'path' in field.lower():
                if not validate_file_path(str(value)):
                    errors.append(f"Invalid file path in field {field}")

    return len(errors) == 0, errors


def sanitize_prompt(prompt: str) -> str:
    """
    Sanitize prompt to remove potentially harmful content.
    
    Args:
        prompt: Prompt to sanitize
        
    Returns:
        Sanitized prompt
    """
    # Remove potential command injection patterns
    sanitized = re.sub(r'\$\([^)]*\)', '', prompt)  # Remove $(...)
    sanitized = re.sub(r'`[^`]*`', '', sanitized)    # Remove `...`
    
    # Remove potential script tags
    sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE)
    
    # Remove potential HTML tags (basic sanitization)
    sanitized = re.sub(r'<[^>]+>', '', sanitized)
    
    return sanitized.strip()


def sanitize_user_input(input_str: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    Wrapper around sanitize_prompt for general user input.

    Args:
        input_str: User input to sanitize

    Returns:
        Sanitized input
    """
    return sanitize_prompt(input_str)


def validate_url(url: str) -> bool:
    """
    Validate URL to ensure it's properly formatted and safe.
    
    Args:
        url: URL to validate
        
    Returns:
        True if the URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        # Check if scheme and netloc are present
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_json_payload(payload: Union[str, Dict], max_depth: int = 10) -> bool:
    """
    Validate JSON payload to prevent deep nesting and oversized payloads.
    
    Args:
        payload: JSON payload to validate (as string or dict)
        max_depth: Maximum allowed nesting depth
        
    Returns:
        True if the payload is valid, False otherwise
    """
    import json
    
    try:
        if isinstance(payload, str):
            data = json.loads(payload)
        else:
            data = payload
        
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                return False
            if isinstance(obj, dict):
                return all(check_depth(v, current_depth + 1) for v in obj.values())
            elif isinstance(obj, list):
                return all(check_depth(item, current_depth + 1) for item in obj)
            return True
        
        return check_depth(data)
    except (json.JSONDecodeError, TypeError, RecursionError):
        return False


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format (basic validation).
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if the API key format is valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Basic format check - most API keys are longer than 10 characters
    if len(api_key) < 10:
        return False
    
    # Check for common API key patterns (alphanumeric with possible special chars)
    pattern = r'^[a-zA-Z0-9_\-+=]+$'
    return bool(re.match(pattern, api_key))


def is_safe_string(input_str: str, allowed_chars: Optional[str] = None) -> bool:
    """
    Check if a string contains only safe characters.
    
    Args:
        input_str: String to validate
        allowed_chars: Regex pattern of allowed characters (defaults to alphanumeric + common punctuation)
        
    Returns:
        True if the string is safe, False otherwise
    """
    if allowed_chars is None:
        # Default: alphanumeric, spaces, common punctuation
        allowed_chars = r'^[a-zA-Z0-9\s\-\_\.\,\!\?\;\:\(\)\[\]\{\}\<\>\=\+\*\/\&\|\@\$]+$'
    
    return bool(re.match(allowed_chars, input_str))


def validate_integer(value: Any, min_val: Optional[int] = None, max_val: Optional[int] = None) -> bool:
    """
    Validate integer value with optional bounds.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        True if the value is a valid integer within bounds, False otherwise
    """
    try:
        int_val = int(value)
        if min_val is not None and int_val < min_val:
            return False
        if max_val is not None and int_val > max_val:
            return False
        return True
    except (TypeError, ValueError):
        return False


def validate_float(value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> bool:
    """
    Validate float value with optional bounds.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        True if the value is a valid float within bounds, False otherwise
    """
    try:
        float_val = float(value)
        if min_val is not None and float_val < min_val:
            return False
        if max_val is not None and float_val > max_val:
            return False
        return True
    except (TypeError, ValueError):
        return False


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if the email format is valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_ip_address(ip: str) -> bool:
    """
    Validate IP address format (both IPv4 and IPv6).
    
    Args:
        ip: IP address to validate
        
    Returns:
        True if the IP address format is valid, False otherwise
    """
    import ipaddress
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


class InputValidator:
    """
    A comprehensive input validator class that combines multiple validation methods.
    """
    
    def __init__(self, allowed_base_paths: Optional[List[Path]] = None):
        """
        Initialize the validator.
        
        Args:
            allowed_base_paths: List of allowed base paths for file operations
        """
        self.allowed_base_paths = allowed_base_paths or []
    
    def validate_file_operation(self, file_path: str) -> bool:
        """
        Validate file operation parameters.
        
        Args:
            file_path: Path for file operation
            
        Returns:
            True if valid, False otherwise
        """
        return validate_file_path(file_path, self.allowed_base_paths)
    
    def validate_model(self, model_name: str) -> bool:
        """
        Validate model name.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            True if valid, False otherwise
        """
        return validate_model_name(model_name)
    
    def validate_user_prompt(self, prompt: str) -> bool:
        """
        Validate user prompt with comprehensive checks.
        
        Args:
            prompt: User prompt to validate
            
        Returns:
            True if valid, False otherwise
        """
        return validate_prompt(prompt)
    
    def sanitize_user_input(self, input_str: str) -> str:
        """
        Apply comprehensive sanitization to user input.
        
        Args:
            input_str: Input string to sanitize
            
        Returns:
            Sanitized string
        """
        # Apply multiple sanitization steps
        sanitized = sanitize_prompt(input_str)
        return sanitized
    
    def validate_json_safe(self, payload: Union[str, Dict]) -> bool:
        """
        Validate JSON payload for safety.
        
        Args:
            payload: JSON payload to validate
            
        Returns:
            True if safe, False otherwise
        """
        return validate_json_payload(payload)