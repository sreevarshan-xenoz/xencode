"""Message content validation for crush."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

try:
    import jsonschema
    from jsonschema import validate, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    ValidationError = Exception

logger = logging.getLogger(__name__)


class MessageValidator:
    """Validates message content parts against JSON schema."""
    
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize message validator.
        
        Args:
            schema_path: Path to JSON schema file. If None, uses default schema.
        """
        if schema_path is None:
            # Use default schema in same directory
            schema_path = Path(__file__).parent / "message_schema.json"
        
        self.schema_path = schema_path
        self._schema: Optional[Dict] = None
        self._load_schema()
    
    def _load_schema(self):
        """Load JSON schema from file."""
        try:
            with open(self.schema_path, 'r') as f:
                self._schema = json.load(f)
            logger.debug(f"Loaded message schema from {self.schema_path}")
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            self._schema = None
    
    def validate(self, content: Dict[str, Any]) -> bool:
        """Validate message content against schema.
        
        Args:
            content: Message content to validate (should have 'parts' key)
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If validation fails and strict mode is enabled
        """
        if not HAS_JSONSCHEMA:
            logger.warning("jsonschema not installed, skipping validation")
            return True
        
        if self._schema is None:
            logger.warning("Schema not loaded, skipping validation")
            return True
        
        try:
            validate(instance=content, schema=self._schema)
            return True
        except ValidationError as e:
            logger.error(f"Message validation failed: {e.message}")
            return False
    
    def validate_part(self, part: Dict[str, Any]) -> bool:
        """Validate a single message part.
        
        Args:
            part: Message part to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Wrap in parts array for validation
        content = {"parts": [part]}
        return self.validate(content)
    
    def validate_parts(self, parts: List[Dict[str, Any]]) -> bool:
        """Validate a list of message parts.
        
        Args:
            parts: List of message parts to validate
            
        Returns:
            True if all parts are valid, False otherwise
        """
        content = {"parts": parts}
        return self.validate(content)


# Content part type validators (for manual validation without jsonschema)

def is_text_part(part: Dict[str, Any]) -> bool:
    """Check if part is a valid text part.
    
    Args:
        part: Part to check
        
    Returns:
        True if valid text part
    """
    return (
        isinstance(part, dict) and
        part.get("type") == "text" and
        isinstance(part.get("text"), str)
    )


def is_tool_call_part(part: Dict[str, Any]) -> bool:
    """Check if part is a valid tool call part.
    
    Args:
        part: Part to check
        
    Returns:
        True if valid tool call part
    """
    return (
        isinstance(part, dict) and
        part.get("type") == "tool_call" and
        isinstance(part.get("tool"), str) and
        isinstance(part.get("input"), dict)
    )


def is_tool_result_part(part: Dict[str, Any]) -> bool:
    """Check if part is a valid tool result part.
    
    Args:
        part: Part to check
        
    Returns:
        True if valid tool result part
    """
    return (
        isinstance(part, dict) and
        part.get("type") == "tool_result" and
        isinstance(part.get("tool"), str) and
        isinstance(part.get("result"), str)
    )


def is_binary_part(part: Dict[str, Any]) -> bool:
    """Check if part is a valid binary part.
    
    Args:
        part: Part to check
        
    Returns:
        True if valid binary part
    """
    return (
        isinstance(part, dict) and
        part.get("type") == "binary" and
        isinstance(part.get("filename"), str) and
        isinstance(part.get("encoding"), str)
    )


def validate_part_type(part: Dict[str, Any]) -> bool:
    """Validate that a part has a recognized type.
    
    Args:
        part: Part to validate
        
    Returns:
        True if part type is valid
    """
    if not isinstance(part, dict):
        return False
    
    part_type = part.get("type")
    
    if part_type == "text":
        return is_text_part(part)
    elif part_type == "tool_call":
        return is_tool_call_part(part)
    elif part_type == "tool_result":
        return is_tool_result_part(part)
    elif part_type == "binary":
        return is_binary_part(part)
    else:
        logger.warning(f"Unknown part type: {part_type}")
        return False


# Global validator instance
_validator: Optional[MessageValidator] = None


def get_validator() -> MessageValidator:
    """Get global message validator instance.
    
    Returns:
        Message validator
    """
    global _validator
    
    if _validator is None:
        _validator = MessageValidator()
    
    return _validator


def validate_message_content(content: Dict[str, Any]) -> bool:
    """Validate message content using global validator.
    
    Args:
        content: Message content to validate
        
    Returns:
        True if valid
    """
    validator = get_validator()
    return validator.validate(content)
