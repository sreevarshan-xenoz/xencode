"""
API response validation module for Xencode
Provides validation for API responses to ensure they meet expected formats
"""
import json
import re
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.panel import Panel

console = Console()

class APIResponseValidator:
    """Class to handle validation of API responses"""
    
    @staticmethod
    def validate_ollama_response(response_data: Union[Dict[str, Any], str]) -> bool:
        """
        Validate Ollama API response format
        
        Args:
            response_data: The response data from Ollama API
            
        Returns:
            True if the response is valid, False otherwise
        """
        try:
            # If it's a string, try to parse it as JSON
            if isinstance(response_data, str):
                data = json.loads(response_data)
            else:
                data = response_data
                
            # Check if it's a streaming response (dict with 'response' key)
            if isinstance(data, dict):
                # Valid streaming responses should have certain keys
                if 'done' in data:  # This indicates a streaming response
                    # Check for required fields in streaming response
                    if 'response' in data or data.get('done', False):
                        return True
                    return False
                else:
                    # Non-streaming response should have 'response' key
                    return 'response' in data
            else:
                # If it's not a dict, it's probably invalid
                return False
        except (json.JSONDecodeError, TypeError):
            return False
    
    @staticmethod
    def validate_model_list_response(response_data: Union[Dict[str, Any], str]) -> bool:
        """
        Validate Ollama model list API response format
        
        Args:
            response_data: The response data from Ollama API
            
        Returns:
            True if the response is valid, False otherwise
        """
        try:
            # If it's a string, try to parse it as JSON
            if isinstance(response_data, str):
                data = json.loads(response_data)
            else:
                data = response_data
                
            # Model list response should have a 'models' key with a list value
            if isinstance(data, dict) and 'models' in data:
                models = data['models']
                if isinstance(models, list):
                    # Each model should be a dict with 'name' key
                    for model in models:
                        if not isinstance(model, dict) or 'name' not in model:
                            return False
                    return True
            return False
        except (json.JSONDecodeError, TypeError):
            return False
    
    @staticmethod
    def validate_json_response(response_text: str) -> Optional[Dict[str, Any]]:
        """
        Validate and parse JSON response
        
        Args:
            response_text: The response text to validate and parse
            
        Returns:
            Parsed JSON data if valid, None otherwise
        """
        try:
            data = json.loads(response_text)
            return data
        except (json.JSONDecodeError, TypeError):
            return None
    
    @staticmethod
    def sanitize_response_content(content: str) -> str:
        """
        Sanitize response content to remove potentially harmful elements
        
        Args:
            content: The content to sanitize
            
        Returns:
            Sanitized content
        """
        if not content:
            return content
            
        # Remove potentially dangerous patterns from responses
        # This is mainly for preventing XSS in displayed content
        sanitized = content
        
        # Remove script tags (for display safety)
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '[SCRIPT REMOVED]', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove javascript: urls
        sanitized = re.sub(r'javascript:', 'JAVASCRIPT_REMOVED:', sanitized, flags=re.IGNORECASE)
        
        # Remove data: urls that might contain executable content
        sanitized = re.sub(r'data:[^,]*?,', 'DATA_URL_REMOVED,', sanitized, flags=re.IGNORECASE)
        
        return sanitized


def validate_api_response(response_data: Union[Dict[str, Any], str], api_type: str = "ollama") -> bool:
    """
    Convenience function to validate API responses based on type
    
    Args:
        response_data: The response data to validate
        api_type: The type of API ('ollama', 'model_list', etc.)
        
    Returns:
        True if the response is valid, False otherwise
    """
    validator = APIResponseValidator()
    
    if api_type == "ollama":
        return validator.validate_ollama_response(response_data)
    elif api_type == "model_list":
        return validator.validate_model_list_response(response_data)
    else:
        # For unknown types, try basic JSON validation
        try:
            if isinstance(response_data, str):
                json.loads(response_data)
            return True
        except (json.JSONDecodeError, TypeError):
            return False


def sanitize_api_response(content: str) -> str:
    """
    Convenience function to sanitize API response content
    
    Args:
        content: The content to sanitize
        
    Returns:
        Sanitized content
    """
    validator = APIResponseValidator()
    return validator.sanitize_response_content(content)