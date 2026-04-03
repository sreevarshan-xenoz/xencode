#!/usr/bin/env python3
"""
Error Auto-Fix Suggestion Engine

Detects common failure signatures and suggests targeted fixes:
- Pattern-based error detection
- Context-aware fix suggestions
- Fix confidence scoring
- Common error library
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from rich.console import Console

console = Console()


class ErrorCategory(Enum):
    """Categories of errors"""
    SYNTAX = "syntax"
    NAME_ERROR = "name_error"
    TYPE_ERROR = "type_error"
    ATTRIBUTE_ERROR = "attribute_error"
    IMPORT_ERROR = "import_error"
    FILE_ERROR = "file_error"
    INDEX_ERROR = "index_error"
    KEY_ERROR = "key_error"
    VALUE_ERROR = "value_error"
    ZERO_DIVISION = "zero_division"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    PERMISSION = "permission"
    UNKNOWN = "unknown"


class FixConfidence(Enum):
    """Confidence level for fix suggestions"""
    HIGH = "high"  # >80% confident
    MEDIUM = "medium"  # 50-80% confident
    LOW = "low"  # <50% confident


@dataclass
class ErrorPattern:
    """Pattern for detecting specific error types"""
    pattern: str
    category: ErrorCategory
    description: str
    flags: int = re.IGNORECASE


@dataclass
class FixSuggestion:
    """Suggested fix for an error"""
    title: str
    description: str
    fix_code: Optional[str] = None
    explanation: str = ""
    confidence: FixConfidence = FixConfidence.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "description": self.description,
            "fix_code": self.fix_code,
            "explanation": self.explanation,
            "confidence": self.confidence.value,
            "category": self.category.value,
            "metadata": self.metadata,
        }


@dataclass
class ErrorAnalysis:
    """Complete analysis of an error"""
    original_error: str
    error_type: str
    category: ErrorCategory
    message: str
    traceback: Optional[str] = None
    suggestions: List[FixSuggestion] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_suggestions(self) -> bool:
        """Check if there are fix suggestions"""
        return len(self.suggestions) > 0
    
    @property
    def best_suggestion(self) -> Optional[FixSuggestion]:
        """Get highest confidence suggestion"""
        if not self.suggestions:
            return None
        
        # Sort by confidence
        confidence_order = {
            FixConfidence.HIGH: 0,
            FixConfidence.MEDIUM: 1,
            FixConfidence.LOW: 2,
        }
        
        sorted_suggestions = sorted(
            self.suggestions,
            key=lambda s: confidence_order.get(s.confidence, 3),
        )
        
        return sorted_suggestions[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_error": self.original_error,
            "error_type": self.error_type,
            "category": self.category.value,
            "message": self.message,
            "traceback": self.traceback,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "has_suggestions": self.has_suggestions,
            "context": self.context,
        }


class ErrorFixEngine:
    """
    Auto-fix suggestion engine
    
    Analyzes errors and suggests targeted fixes based on
    pattern matching and common error knowledge.
    
    Usage:
        engine = ErrorFixEngine()
        analysis = engine.analyze_error("NameError: name 'x' is not defined")
        print(analysis.best_suggestion)
    """
    
    # Error patterns library
    ERROR_PATTERNS: List[ErrorPattern] = [
        # Syntax errors
        ErrorPattern(
            r"SyntaxError:\s*(?:invalid syntax|unexpected EOF|EOL)",
            ErrorCategory.SYNTAX,
            "Syntax error in code",
        ),
        ErrorPattern(
            r"IndentationError:",
            ErrorCategory.SYNTAX,
            "Indentation error",
        ),
        
        # Name errors
        ErrorPattern(
            r"NameError:\s*name\s+'(\w+)'\s+is\s+not\s+defined",
            ErrorCategory.NAME_ERROR,
            "Undefined variable or function",
        ),
        
        # Type errors
        ErrorPattern(
            r"TypeError:\s*(?:unsupported operand|cannot concatenate|'NoneType')",
            ErrorCategory.TYPE_ERROR,
            "Type mismatch or incompatible operation",
        ),
        
        # Attribute errors
        ErrorPattern(
            r"AttributeError:\s*(?:'(\w+)' object has no attribute|module '\w+' has no attribute)",
            ErrorCategory.ATTRIBUTE_ERROR,
            "Missing attribute or method",
        ),
        
        # Import errors
        ErrorPattern(
            r"ImportError:\s*(?:No module named|cannot import name)",
            ErrorCategory.IMPORT_ERROR,
            "Missing module or import failure",
        ),
        ErrorPattern(
            r"ModuleNotFoundError:\s*No module named\s+'(\w+)'",
            ErrorCategory.IMPORT_ERROR,
            "Module not found",
        ),
        
        # File errors
        ErrorPattern(
            r"FileNotFoundError:\s*\[Errno 2\]",
            ErrorCategory.FILE_ERROR,
            "File not found",
        ),
        ErrorPattern(
            r"PermissionError:\s*\[Errno 13\]",
            ErrorCategory.PERMISSION,
            "Permission denied",
        ),
        
        # Index/Key errors
        ErrorPattern(
            r"IndexError:\s*list index out of range",
            ErrorCategory.INDEX_ERROR,
            "List index out of bounds",
        ),
        ErrorPattern(
            r"KeyError:\s*'(\w+)'",
            ErrorCategory.KEY_ERROR,
            "Dictionary key not found",
        ),
        
        # Value errors
        ErrorPattern(
            r"ValueError:\s*(?:invalid literal|too many values|not enough values)",
            ErrorCategory.VALUE_ERROR,
            "Invalid value or argument",
        ),
        
        # Zero division
        ErrorPattern(
            r"ZeroDivisionError:",
            ErrorCategory.ZERO_DIVISION,
            "Division by zero",
        ),
        
        # Timeout/Connection
        ErrorPattern(
            r"(?:asyncio\.exceptions\.)?TimeoutError",
            ErrorCategory.TIMEOUT,
            "Operation timed out",
        ),
        ErrorPattern(
            r"(?:ConnectionError|ConnectionRefusedError|ConnectionResetError)",
            ErrorCategory.CONNECTION,
            "Network connection error",
        ),
    ]
    
    # Fix suggestions library
    FIX_SUGGESTIONS: Dict[ErrorCategory, List[FixSuggestion]] = {
        ErrorCategory.SYNTAX: [
            FixSuggestion(
                title="Check Syntax",
                description="Review code for syntax errors",
                explanation="Common causes: missing colons, unmatched parentheses, incorrect indentation",
                confidence=FixConfidence.HIGH,
            ),
            FixSuggestion(
                title="Fix Indentation",
                description="Ensure consistent indentation (4 spaces recommended)",
                fix_code="# Use 4 spaces consistently\nfor i in range(10):\n    print(i)  # Indented with 4 spaces",
                confidence=FixConfidence.MEDIUM,
            ),
        ],
        
        ErrorCategory.NAME_ERROR: [
            FixSuggestion(
                title="Define Variable",
                description="Ensure the variable is defined before use",
                fix_code="# Define variable before using it\nx = 0  # Add this line\nprint(x)",
                confidence=FixConfidence.HIGH,
            ),
            FixSuggestion(
                title="Check Spelling",
                description="Verify variable/function name spelling",
                explanation="Python is case-sensitive: 'myVar' != 'myvar'",
                confidence=FixConfidence.MEDIUM,
            ),
            FixSuggestion(
                title="Check Scope",
                description="Ensure variable is in correct scope",
                explanation="Variables defined in functions are local by default",
                confidence=FixConfidence.MEDIUM,
            ),
        ],
        
        ErrorCategory.TYPE_ERROR: [
            FixSuggestion(
                title="Check Types",
                description="Ensure operands have compatible types",
                fix_code="# Convert types explicitly\nresult = str(42) + 'hello'  # '42hello'\nresult = int('42') + 8  # 50",
                confidence=FixConfidence.HIGH,
            ),
            FixSuggestion(
                title="Handle None",
                description="Check for None before operations",
                fix_code="if value is not None:\n    result = value + 1",
                confidence=FixConfidence.MEDIUM,
            ),
        ],
        
        ErrorCategory.ATTRIBUTE_ERROR: [
            FixSuggestion(
                title="Check Object Type",
                description="Verify the object has the expected attribute",
                fix_code="# Check attribute exists\nif hasattr(obj, 'method'):\n    obj.method()",
                confidence=FixConfidence.HIGH,
            ),
            FixSuggestion(
                title="Check Import",
                description="Ensure module is imported correctly",
                explanation="Use 'import module' or 'from module import attribute'",
                confidence=FixConfidence.MEDIUM,
            ),
        ],
        
        ErrorCategory.IMPORT_ERROR: [
            FixSuggestion(
                title="Install Package",
                description="Install the missing package",
                fix_code="pip install package_name",
                confidence=FixConfidence.HIGH,
            ),
            FixSuggestion(
                title="Check Import Path",
                description="Verify import statement syntax",
                fix_code="# Correct import syntax\nimport module\nfrom module import function\nfrom module import Class as Alias",
                confidence=FixConfidence.MEDIUM,
            ),
        ],
        
        ErrorCategory.FILE_ERROR: [
            FixSuggestion(
                title="Check File Path",
                description="Verify file path is correct",
                fix_code="from pathlib import Path\n\n# Use absolute path\nfile_path = Path('/absolute/path/to/file.txt')\nif file_path.exists():\n    with open(file_path) as f:\n        content = f.read()",
                confidence=FixConfidence.HIGH,
            ),
            FixSuggestion(
                title="Use Try-Except",
                description="Handle missing file gracefully",
                fix_code="try:\n    with open('file.txt') as f:\n        content = f.read()\nexcept FileNotFoundError:\n    print('File not found')",
                confidence=FixConfidence.MEDIUM,
            ),
        ],
        
        ErrorCategory.INDEX_ERROR: [
            FixSuggestion(
                title="Check List Length",
                description="Verify index is within bounds",
                fix_code="# Check length before accessing\nif len(my_list) > index:\n    value = my_list[index]\n\n# Or use safe access\nvalue = my_list[index] if index < len(my_list) else None",
                confidence=FixConfidence.HIGH,
            ),
        ],
        
        ErrorCategory.KEY_ERROR: [
            FixSuggestion(
                title="Use .get() Method",
                description="Safely access dictionary keys",
                fix_code="# Safe access with default\nvalue = my_dict.get('key', default_value)\n\n# Or check first\nif 'key' in my_dict:\n    value = my_dict['key']",
                confidence=FixConfidence.HIGH,
            ),
        ],
        
        ErrorCategory.ZERO_DIVISION: [
            FixSuggestion(
                title="Check Divisor",
                description="Ensure divisor is not zero",
                fix_code="# Check before dividing\nif divisor != 0:\n    result = dividend / divisor\nelse:\n    result = 0  # or handle appropriately",
                confidence=FixConfidence.HIGH,
            ),
        ],
        
        ErrorCategory.TIMEOUT: [
            FixSuggestion(
                title="Increase Timeout",
                description="Increase timeout value for slow operations",
                fix_code="# Increase timeout\nresult = await asyncio.wait_for(\n    operation(),\n    timeout=60.0  # Increase from default\n)",
                confidence=FixConfidence.MEDIUM,
            ),
            FixSuggestion(
                title="Optimize Operation",
                description="Consider optimizing the slow operation",
                explanation="Review algorithm complexity, add caching, or use async operations",
                confidence=FixConfidence.LOW,
            ),
        ],
        
        ErrorCategory.CONNECTION: [
            FixSuggestion(
                title="Check Connection",
                description="Verify network connectivity",
                fix_code="# Add retry logic\nimport asyncio\n\nasync def connect_with_retry(url, max_attempts=3):\n    for attempt in range(max_attempts):\n        try:\n            return await connect(url)\n        except ConnectionError:\n            if attempt == max_attempts - 1:\n                raise\n            await asyncio.sleep(2 ** attempt)",
                confidence=FixConfidence.MEDIUM,
            ),
        ],
        
        ErrorCategory.PERMISSION: [
            FixSuggestion(
                title="Check Permissions",
                description="Verify file/directory permissions",
                fix_code="# Run as administrator or check permissions\n# Windows: Run as Administrator\n# Linux/Mac: chmod +x file or use sudo",
                confidence=FixConfidence.HIGH,
            ),
        ],
        
        ErrorCategory.VALUE_ERROR: [
            FixSuggestion(
                title="Validate Input",
                description="Validate input before processing",
                fix_code="# Validate before conversion\ntry:\n    value = int(user_input)\nexcept ValueError:\n    print(f'Invalid number: {user_input}')",
                confidence=FixConfidence.HIGH,
            ),
        ],
    }
    
    def __init__(self):
        """Initialize error fix engine"""
        self._compiled_patterns: List[Tuple[re.Pattern, ErrorPattern]] = []
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        for error_pattern in self.ERROR_PATTERNS:
            try:
                compiled = re.compile(error_pattern.pattern, error_pattern.flags)
                self._compiled_patterns.append((compiled, error_pattern))
            except re.error as e:
                console.print(f"[yellow]Warning: Invalid pattern '{error_pattern.pattern}': {e}[/yellow]")
    
    def analyze_error(
        self,
        error_message: str,
        traceback: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorAnalysis:
        """
        Analyze error and generate fix suggestions
        
        Args:
            error_message: Full error message
            traceback: Optional traceback string
            context: Optional context (file path, line number, code snippet)
            
        Returns:
            ErrorAnalysis with suggestions
        """
        # Extract error type
        error_type = self._extract_error_type(error_message)
        
        # Categorize error
        category = self._categorize_error(error_message)
        
        # Extract message
        message = self._extract_message(error_message)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(category, error_message, context)
        
        return ErrorAnalysis(
            original_error=error_message,
            error_type=error_type,
            category=category,
            message=message,
            traceback=traceback,
            suggestions=suggestions,
            context=context or {},
        )
    
    def _extract_error_type(self, error_message: str) -> str:
        """Extract error type from message"""
        # Match patterns like "ErrorType:" at start
        match = re.match(r"(\w+Error|\w+Exception):\s*", error_message)
        if match:
            return match.group(1)
        return "UnknownError"
    
    def _categorize_error(self, error_message: str) -> ErrorCategory:
        """Categorize error using pattern matching"""
        for pattern, error_pattern in self._compiled_patterns:
            if pattern.search(error_message):
                return error_pattern.category
        
        return ErrorCategory.UNKNOWN
    
    def _extract_message(self, error_message: str) -> str:
        """Extract the error message portion"""
        # Remove error type prefix
        match = re.match(r"\w+Error:\s*(.+)", error_message)
        if match:
            return match.group(1).strip()
        return error_message.strip()
    
    def _generate_suggestions(
        self,
        category: ErrorCategory,
        error_message: str,
        context: Optional[Dict[str, Any]],
    ) -> List[FixSuggestion]:
        """Generate fix suggestions based on error category"""
        suggestions = []
        
        # Get base suggestions for category
        base_suggestions = self.FIX_SUGGESTIONS.get(category, [])
        
        for suggestion in base_suggestions:
            # Create a copy with updated context
            new_suggestion = FixSuggestion(
                title=suggestion.title,
                description=suggestion.description,
                fix_code=suggestion.fix_code,
                explanation=suggestion.explanation,
                confidence=suggestion.confidence,
                category=category,
                metadata={**suggestion.metadata},
            )
            
            # Add extracted info to metadata
            if context:
                new_suggestion.metadata.update(context)
            
            suggestions.append(new_suggestion)
        
        # Add context-specific suggestions
        if context:
            suggestions.extend(self._context_specific_suggestions(category, error_message, context))
        
        return suggestions
    
    def _context_specific_suggestions(
        self,
        category: ErrorCategory,
        error_message: str,
        context: Dict[str, Any],
    ) -> List[FixSuggestion]:
        """Generate suggestions based on context"""
        suggestions = []
        
        # File path context
        if "file_path" in context:
            suggestions.append(FixSuggestion(
                title="Verify File Exists",
                description=f"Check if '{context['file_path']}' exists",
                fix_code=f"from pathlib import Path\n\nif Path('{context['file_path']}').exists():\n    # File exists, proceed\nelse:\n    # Handle missing file",
                confidence=FixConfidence.HIGH,
                category=category,
            ))
        
        # Line number context
        if "line_number" in context and "code_snippet" in context:
            suggestions.append(FixSuggestion(
                title="Review Line {line_number}".format(line_number=context["line_number"]),
                description=f"Check code at line {context['line_number']}",
                fix_code=context["code_snippet"],
                confidence=FixConfidence.MEDIUM,
                category=category,
            ))
        
        # Variable name context (from NameError)
        var_match = re.search(r"name\s+'(\w+)'\s+is\s+not\s+defined", error_message)
        if var_match:
            var_name = var_match.group(1)
            suggestions.append(FixSuggestion(
                title=f"Define '{var_name}'",
                description=f"Variable '{var_name}' is not defined",
                fix_code=f"# Add definition before use\n{var_name} = None  # or appropriate value",
                confidence=FixConfidence.HIGH,
                category=ErrorCategory.NAME_ERROR,
            ))
        
        # Module name context (from ImportError)
        mod_match = re.search(r"(?:No module named|ModuleNotFoundError).*'(\w+)'", error_message)
        if mod_match:
            mod_name = mod_match.group(1)
            suggestions.append(FixSuggestion(
                title=f"Install '{mod_name}'",
                description=f"Module '{mod_name}' is not installed",
                fix_code=f"pip install {mod_name}",
                confidence=FixConfidence.HIGH,
                category=ErrorCategory.IMPORT_ERROR,
            ))
        
        return suggestions
    
    def get_error_summary(self, analysis: ErrorAnalysis) -> str:
        """Generate human-readable error summary"""
        lines = [
            f"[bold red]Error:[/bold red] {analysis.error_type}",
            f"[dim]{analysis.message}[/dim]",
            f"\n[bold]Category:[/bold] {analysis.category.value}",
        ]
        
        if analysis.suggestions:
            lines.append(f"\n[bold green]Suggested Fixes ({len(analysis.suggestions)}):[/bold green]")
            
            for i, suggestion in enumerate(analysis.suggestions[:3], 1):  # Show top 3
                confidence_icon = {
                    FixConfidence.HIGH: "✓",
                    FixConfidence.MEDIUM: "~",
                    FixConfidence.LOW: "?",
                }.get(suggestion.confidence, "?")
                
                lines.append(f"\n  {confidence_icon}. [bold]{suggestion.title}[/bold]")
                lines.append(f"     {suggestion.description}")
                
                if suggestion.explanation:
                    lines.append(f"     [dim]{suggestion.explanation}[/dim]")
        
        return "\n".join(lines)


# Global engine instance
_engine: Optional[ErrorFixEngine] = None


def get_fix_engine() -> ErrorFixEngine:
    """Get or create global error fix engine"""
    global _engine
    if _engine is None:
        _engine = ErrorFixEngine()
    return _engine


def analyze_error(
    error_message: str,
    traceback: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ErrorAnalysis:
    """
    Convenience function to analyze error
    
    Args:
        error_message: Error message string
        traceback: Optional traceback
        context: Optional context
        
    Returns:
        ErrorAnalysis with suggestions
    """
    return get_fix_engine().analyze_error(error_message, traceback, context)


def suggest_fix(
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[FixSuggestion]:
    """
    Get best fix suggestion for an error
    
    Args:
        error_message: Error message string
        context: Optional context
        
    Returns:
        Best FixSuggestion or None
    """
    analysis = analyze_error(error_message, context=context)
    return analysis.best_suggestion


if __name__ == "__main__":
    # Demo
    console.print("[bold blue]Error Auto-Fix Suggestion Engine Demo[/bold blue]\n")
    
    engine = ErrorFixEngine()
    
    # Test cases
    test_errors = [
        "NameError: name 'undefined_var' is not defined",
        "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        "FileNotFoundError: [Errno 2] No such file or directory: 'missing.txt'",
        "ZeroDivisionError: division by zero",
        "ModuleNotFoundError: No module named 'requests'",
        "IndexError: list index out of range",
        "KeyError: 'missing_key'",
    ]
    
    for error in test_errors:
        console.print(f"\n[bold]Error:[/bold] {error}")
        console.print("-" * 50)
        
        analysis = engine.analyze_error(error)
        console.print(engine.get_error_summary(analysis))
        console.print()
