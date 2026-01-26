"""
Sanitization utilities for Xencode
Provides input sanitization for user inputs
"""
import html
import re
from typing import Any, Dict, List, Union


def sanitize_input(input_data: Union[str, Dict, List, Any]) -> Union[str, Dict, List, Any]:
    """
    Sanitize input data recursively to remove potentially harmful content.
    
    Args:
        input_data: Data to sanitize (string, dict, list, or other)
        
    Returns:
        Sanitized data
    """
    if isinstance(input_data, str):
        return _sanitize_string(input_data)
    elif isinstance(input_data, dict):
        sanitized_dict = {}
        for key, value in input_data.items():
            sanitized_key = _sanitize_string(str(key)) if isinstance(key, str) else key
            sanitized_value = sanitize_input(value)
            sanitized_dict[sanitized_key] = sanitized_value
        return sanitized_dict
    elif isinstance(input_data, list):
        return [sanitize_input(item) for item in input_data]
    else:
        return input_data


def _sanitize_string(input_str: str) -> str:
    """
    Sanitize a string by removing or escaping potentially harmful content.
    
    Args:
        input_str: String to sanitize
        
    Returns:
        Sanitized string
    """
    if not isinstance(input_str, str):
        return input_str
    
    # HTML encode special characters
    sanitized = html.escape(input_str)
    
    # Remove potential command injection patterns
    sanitized = re.sub(r'\$\([^)]*\)', '', sanitized)  # Remove $()
    sanitized = re.sub(r'`[^`]*`', '', sanitized)      # Remove ``
    
    # Remove potential script tags (case insensitive)
    sanitized = re.sub(r'<\s*script[^>]*>.*?<\s*/\s*script\s*>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove potential iframe tags
    sanitized = re.sub(r'<\s*iframe[^>]*>.*?<\s*/\s*iframe\s*>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove javascript: and data: URIs in href/src attributes
    sanitized = re.sub(r'(href|src)\s*=\s*["\'][^"\']*javascript:[^"\']*["\']', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'(href|src)\s*=\s*["\'][^"\']*data:[^"\']*["\']', '', sanitized, flags=re.IGNORECASE)
    
    # Remove on* event handlers
    sanitized = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()


def strip_control_characters(input_str: str) -> str:
    """
    Remove control characters from a string that could be used maliciously.
    
    Args:
        input_str: String to clean
        
    Returns:
        String with control characters removed
    """
    if not isinstance(input_str, str):
        return input_str
    
    # Remove control characters (ASCII 0-31) except tab, newline, and carriage return
    cleaned = ''.join(char for char in input_str if ord(char) >= 32 or char in '\t\n\r')
    return cleaned


def normalize_whitespace(input_str: str) -> str:
    """
    Normalize whitespace in a string to prevent parsing issues.
    
    Args:
        input_str: String to normalize
        
    Returns:
        String with normalized whitespace
    """
    if not isinstance(input_str, str):
        return input_str
    
    # Replace multiple consecutive whitespace characters with a single space
    normalized = re.sub(r'\s+', ' ', input_str)
    return normalized.strip()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and other attacks.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    if not isinstance(filename, str):
        return filename
    
    # Remove dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Prevent directory traversal
    sanitized = sanitized.replace('../', '_').replace('..\\', '_')
    sanitized = sanitized.replace('./', '_').replace('.\\', '_')
    
    # Limit length to prevent buffer overflow attempts
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized


def sanitize_sql_like_input(input_str: str) -> str:
    """
    Sanitize input that might be used in SQL LIKE clauses to prevent wildcard abuse.
    
    Args:
        input_str: String to sanitize
        
    Returns:
        Sanitized string
    """
    if not isinstance(input_str, str):
        return input_str
    
    # Escape SQL LIKE wildcards
    sanitized = input_str.replace('%', '\\%').replace('_', '\\_')
    return sanitized


def sanitize_path(path: str) -> str:
    """
    Sanitize file path to prevent directory traversal.
    
    Args:
        path: Path to sanitize
        
    Returns:
        Sanitized path
    """
    if not isinstance(path, str):
        return path
    
    # Replace backslashes with forward slashes for consistency
    path = path.replace('\\', '/')
    
    # Split path into components and sanitize each part
    parts = path.split('/')
    sanitized_parts = []
    
    for part in parts:
        if part == '..' or part == '.':
            # Skip navigation components
            continue
        sanitized_part = sanitize_filename(part)
        if sanitized_part:  # Only add non-empty parts
            sanitized_parts.append(sanitized_part)
    
    return '/'.join(sanitized_parts)


def remove_potential_injections(input_str: str) -> str:
    """
    Remove potential injection patterns from input string.
    
    Args:
        input_str: String to clean
        
    Returns:
        Cleaned string
    """
    if not isinstance(input_str, str):
        return input_str
    
    # Remove potential NoSQL injection patterns
    patterns_to_remove = [
        r'\$where', r'\$eval', r'\$function',  # NoSQL
        r'\bOR\b\s+\d+=\d+', r'\bAND\b\s+\d+=\d+',  # SQL
        r'UNION\s+SELECT',  # SQL
        r'\bDROP\s+\w+',  # SQL
        r'\bDELETE\s+FROM',  # SQL
        r'\bINSERT\s+INTO',  # SQL
        r'\bUPDATE\s+\w+\s+SET',  # SQL
    ]
    
    sanitized = input_str
    for pattern in patterns_to_remove:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()


def sanitize_for_logging(input_str: str) -> str:
    """
    Sanitize input specifically for logging to prevent log injection.
    
    Args:
        input_str: String to sanitize for logging
        
    Returns:
        Sanitized string safe for logging
    """
    if not isinstance(input_str, str):
        return input_str
    
    # Remove newlines to prevent log forging
    sanitized = input_str.replace('\n', ' ').replace('\r', ' ')
    
    # Remove tab characters
    sanitized = sanitized.replace('\t', ' ')
    
    # Strip leading/trailing whitespace
    return sanitized.strip()


def sanitize_json_keys(json_obj: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
    """
    Sanitize keys in JSON objects to prevent injection through keys.
    
    Args:
        json_obj: JSON object (dict, list, or primitive) to sanitize
        
    Returns:
        Sanitized JSON object
    """
    if isinstance(json_obj, dict):
        sanitized_dict = {}
        for key, value in json_obj.items():
            # Sanitize the key if it's a string
            sanitized_key = _sanitize_string(str(key)) if isinstance(key, str) else key
            # Recursively sanitize the value
            sanitized_value = sanitize_json_keys(value)
            sanitized_dict[sanitized_key] = sanitized_value
        return sanitized_dict
    elif isinstance(json_obj, list):
        return [sanitize_json_keys(item) for item in json_obj]
    else:
        return json_obj


def is_valid_utf8(input_str: str) -> bool:
    """
    Check if a string contains valid UTF-8 sequences.
    
    Args:
        input_str: String to validate
        
    Returns:
        True if the string contains valid UTF-8, False otherwise
    """
    try:
        input_str.encode('utf-8').decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def sanitize_multiline_string(input_str: str) -> str:
    """
    Sanitize multiline string by normalizing line endings and removing control characters.
    
    Args:
        input_str: Multiline string to sanitize
        
    Returns:
        Sanitized multiline string
    """
    if not isinstance(input_str, str):
        return input_str
    
    # Normalize line endings to \n
    normalized = input_str.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove control characters except tab, newline, and carriage return
    cleaned = strip_control_characters(normalized)
    
    # Normalize whitespace
    return normalize_whitespace(cleaned)