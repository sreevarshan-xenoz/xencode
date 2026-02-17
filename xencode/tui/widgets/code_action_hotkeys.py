#!/usr/bin/env python3
"""
Refactor/Explain Hotkeys for Xencode TUI

Provides one-key actions for:
- Explain selected code
- Refactor selected code
- Generate tests for selected code
- Optimize selected code
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from rich.console import Console

console = Console()


class ActionType(Enum):
    """Available code action types"""
    EXPLAIN = "explain"
    REFACTOR = "refactor"
    TEST = "test"
    OPTIMIZE = "optimize"
    DOCUMENT = "document"
    DEBUG = "debug"


@dataclass
class CodeAction:
    """Represents a code action request"""
    action_type: ActionType
    code_snippet: str
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    language: str = "python"
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action_type": self.action_type.value,
            "code_snippet": self.code_snippet,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "language": self.language,
            "context": self.context,
        }


@dataclass
class CodeActionResult:
    """Result of a code action"""
    action_type: ActionType
    success: bool
    result: str = ""
    error: str = ""
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action_type": self.action_type.value,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }


class CodeActionPrompts:
    """Prompt templates for code actions"""
    
    EXPLAIN_TEMPLATE = """Explain the following {language} code clearly and concisely:

```{language}
{code}
```

Provide:
1. A brief summary of what the code does
2. Key operations or logic flow
3. Any potential issues or improvements
"""

    REFACTOR_TEMPLATE = """Refactor the following {language} code to improve readability, maintainability, and performance:

```{language}
{code}
```

Provide:
1. The refactored code
2. A list of changes made
3. Explanation of improvements

Keep the same functionality but make the code cleaner and more Pythonic.
"""

    TEST_TEMPLATE = """Generate comprehensive unit tests for the following {language} code:

```{language}
{code}
```

Provide:
1. Test cases covering normal operation
2. Test cases for edge cases
3. Test cases for error conditions
4. Use pytest framework

Include clear test names and assertions.
"""

    OPTIMIZE_TEMPLATE = """Optimize the following {language} code for better performance:

```{language}
{code}
```

Provide:
1. The optimized code
2. Explanation of performance improvements
3. Any trade-offs made

Focus on:
- Algorithm efficiency
- Memory usage
- Reducing redundant operations
"""

    DOCUMENT_TEMPLATE = """Generate documentation for the following {language} code:

```{language}
{code}
```

Provide:
1. Docstrings for functions/classes
2. Type hints if missing
3. Usage examples
4. Parameter descriptions

Follow Google-style docstring format.
"""

    DEBUG_TEMPLATE = """Analyze the following {language} code for potential bugs and issues:

```{language}
{code}
```

Provide:
1. List of potential bugs or issues
2. Explanation of each issue
3. Suggested fixes
4. Best practices to follow

Be thorough and check for:
- Logic errors
- Edge cases
- Resource leaks
- Security issues
"""

    @classmethod
    def get_prompt(cls, action_type: ActionType, code: str, language: str = "python") -> str:
        """Get prompt template for action type"""
        templates = {
            ActionType.EXPLAIN: cls.EXPLAIN_TEMPLATE,
            ActionType.REFACTOR: cls.REFACTOR_TEMPLATE,
            ActionType.TEST: cls.TEST_TEMPLATE,
            ActionType.OPTIMIZE: cls.OPTIMIZE_TEMPLATE,
            ActionType.DOCUMENT: cls.DOCUMENT_TEMPLATE,
            ActionType.DEBUG: cls.DEBUG_TEMPLATE,
        }
        
        template = templates.get(action_type, cls.EXPLAIN_TEMPLATE)
        return template.format(language=language, code=code)


class CodeActionHandler:
    """
    Handles code action requests
    
    Integrates with LLM to execute actions
    
    Usage:
        handler = CodeActionHandler(model_callback=my_llm_callback)
        result = await handler.execute_action(
            ActionType.EXPLAIN,
            "def hello(): print('world')"
        )
    """
    
    def __init__(
        self,
        model_callback: Optional[Callable] = None,
        default_language: str = "python",
    ):
        """
        Initialize code action handler
        
        Args:
            model_callback: Async callback for LLM calls
            default_language: Default programming language
        """
        self.model_callback = model_callback
        self.default_language = default_language
        self._history: List[CodeActionResult] = []
    
    async def execute_action(
        self,
        action_type: ActionType,
        code: str,
        file_path: Optional[str] = None,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CodeActionResult:
        """
        Execute a code action
        
        Args:
            action_type: Type of action to perform
            code: Code snippet to process
            file_path: Optional source file path
            line_start: Optional start line number
            line_end: Optional end line number
            language: Programming language (default: python)
            context: Optional additional context
            
        Returns:
            CodeActionResult with the action result
        """
        lang = language or self.default_language
        
        # Create action object
        action = CodeAction(
            action_type=action_type,
            code_snippet=code,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            language=lang,
            context=context or {},
        )
        
        try:
            # Generate prompt
            prompt = CodeActionPrompts.get_prompt(action_type, code, lang)
            
            # Add context to prompt if available
            if context:
                prompt += "\n\nAdditional context:\n"
                for key, value in context.items():
                    prompt += f"- {key}: {value}\n"
            
            # Call LLM if callback available
            if self.model_callback:
                result_text = await self.model_callback(prompt)
                
                return CodeActionResult(
                    action_type=action_type,
                    success=True,
                    result=result_text,
                )
            else:
                # No callback - return simulated response
                return self._simulate_action(action_type, code, lang)
                
        except Exception as e:
            return CodeActionResult(
                action_type=action_type,
                success=False,
                error=str(e),
            )
    
    def _simulate_action(
        self,
        action_type: ActionType,
        code: str,
        language: str,
    ) -> CodeActionResult:
        """Simulate action result when no LLM callback"""
        simulations = {
            ActionType.EXPLAIN: self._simulate_explain,
            ActionType.REFACTOR: self._simulate_refactor,
            ActionType.TEST: self._simulate_test,
            ActionType.OPTIMIZE: self._simulate_optimize,
            ActionType.DOCUMENT: self._simulate_document,
            ActionType.DEBUG: self._simulate_debug,
        }
        
        simulate_func = simulations.get(action_type, self._simulate_explain)
        return simulate_func(code, language)
    
    def _simulate_explain(self, code: str, language: str) -> CodeActionResult:
        """Simulate explain action"""
        lines = code.strip().split('\n')
        
        return CodeActionResult(
            action_type=ActionType.EXPLAIN,
            success=True,
            result=f"""**Code Explanation**

This {language} code contains {len(lines)} line(s).

**Summary:**
The code appears to define functionality related to the operations shown.

**Key Operations:**
- Code analysis would identify specific functions and logic flow
- Variable usage and control flow would be explained
- Dependencies and imports would be listed

**Note:** Connect an LLM for detailed explanations.
""",
            suggestions=[
                "Add docstrings for better clarity",
                "Consider adding type hints",
                "Break complex logic into smaller functions",
            ],
        )
    
    def _simulate_refactor(self, code: str, language: str) -> CodeActionResult:
        """Simulate refactor action"""
        return CodeActionResult(
            action_type=ActionType.REFACTOR,
            success=True,
            result=f"""**Refactored Code**

```{language}
{code}
# Refactored version would be generated by LLM
```

**Changes Made:**
- Code structure analysis pending LLM connection
- Improvements would be suggested based on best practices

**Common Refactoring Improvements:**
1. Extract methods for complex logic
2. Use list comprehensions where appropriate
3. Add error handling
4. Improve variable naming
5. Reduce nesting depth
""",
            suggestions=[
                "Extract repeated code into functions",
                "Use context managers for resources",
                "Add input validation",
            ],
        )
    
    def _simulate_test(self, code: str, language: str) -> CodeActionResult:
        """Simulate test generation"""
        return CodeActionResult(
            action_type=ActionType.TEST,
            success=True,
            result=f"""**Generated Tests**

```{language}
import pytest

# Test template - connect LLM for specific tests

def test_example():
    '''Example test structure'''
    # Add assertions based on code functionality
    assert True  # Replace with actual tests

def test_edge_cases():
    '''Test edge cases'''
    # Test boundary conditions
    pass

def test_error_handling():
    '''Test error conditions'''
    # Test that errors are raised appropriately
    with pytest.raises(Exception):
        pass
```

**Test Coverage Areas:**
- Normal operation tests
- Edge case tests  
- Error handling tests
- Integration tests
""",
            suggestions=[
                "Add tests for all public functions",
                "Include edge case coverage",
                "Mock external dependencies",
            ],
        )
    
    def _simulate_optimize(self, code: str, language: str) -> CodeActionResult:
        """Simulate optimization"""
        return CodeActionResult(
            action_type=ActionType.OPTIMIZE,
            success=True,
            result=f"""**Optimization Analysis**

**Current Code Analysis:**
- Lines of code: {len(code.split(chr(10)))}
- Complexity: Pending LLM analysis

**Potential Optimizations:**
1. **Algorithm**: Review for O(n) improvements
2. **Memory**: Check for unnecessary allocations
3. **Caching**: Identify repeated computations
4. **I/O**: Batch operations where possible

**Common Python Optimizations:**
- Use generators for large sequences
- Cache expensive computations with @lru_cache
- Use built-in functions (map, filter, sum)
- Avoid global variable lookups in loops
""",
            suggestions=[
                "Profile code to identify bottlenecks",
                "Use appropriate data structures",
                "Consider async for I/O operations",
            ],
        )
    
    def _simulate_document(self, code: str, language: str) -> CodeActionResult:
        """Simulate documentation"""
        return CodeActionResult(
            action_type=ActionType.DOCUMENT,
            success=True,
            result=f"""**Documentation Template**

```{language}
def function_name(param1: Type, param2: Type) -> ReturnType:
    \"\"\"
    Brief description of function purpose.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When this exception is raised

    Example:
        >>> function_name(arg1, arg2)
        expected_output
    \"\"\"
    pass
```

**Documentation Guidelines:**
- Use Google-style docstrings
- Include type hints
- Add usage examples
- Document all parameters and return values
""",
            suggestions=[
                "Add docstrings to all public functions",
                "Include type hints",
                "Add usage examples",
            ],
        )
    
    def _simulate_debug(self, code: str, language: str) -> CodeActionResult:
        """Simulate debug analysis"""
        return CodeActionResult(
            action_type=ActionType.DEBUG,
            success=True,
            result=f"""**Debug Analysis**

**Potential Issues to Check:**

1. **Logic Errors**
   - Review conditional statements
   - Check loop boundaries
   - Verify operator precedence

2. **Edge Cases**
   - Empty inputs
   - None/null values
   - Boundary conditions

3. **Resource Management**
   - File handles closed properly
   - Database connections released
   - Memory leaks

4. **Security**
   - Input validation
   - SQL injection prevention
   - Path traversal protection

**Recommended Debugging Steps:**
1. Add logging statements
2. Use debugger breakpoints
3. Write unit tests
4. Run static analysis tools
""",
            suggestions=[
                "Add input validation",
                "Use try-except for error handling",
                "Add logging for debugging",
            ],
        )
    
    def get_history(self, count: int = 10) -> List[CodeActionResult]:
        """Get recent action history"""
        return self._history[-count:]
    
    def clear_history(self):
        """Clear action history"""
        self._history = []


# Hotkey mappings
HOTKEY_MAP: Dict[str, ActionType] = {
    "e": ActionType.EXPLAIN,
    "r": ActionType.REFACTOR,
    "t": ActionType.TEST,
    "o": ActionType.OPTIMIZE,
    "d": ActionType.DOCUMENT,
    "b": ActionType.DEBUG,  # b for bug hunt
}

HOTKEY_HELP = """
**Code Action Hotkeys:**
  [E] Explain code
  [R] Refactor code
  [T] Generate tests
  [O] Optimize code
  [D] Generate documentation
  [B] Debug analysis
"""


# Global handler instance
_handler: Optional[CodeActionHandler] = None


def get_action_handler(model_callback: Optional[Callable] = None) -> CodeActionHandler:
    """Get or create global action handler"""
    global _handler
    if _handler is None:
        _handler = CodeActionHandler(model_callback=model_callback)
    return _handler


# Convenience functions
async def explain_code(code: str, language: str = "python") -> CodeActionResult:
    """Explain selected code"""
    handler = get_action_handler()
    return await handler.execute_action(ActionType.EXPLAIN, code, language=language)


async def refactor_code(code: str, language: str = "python") -> CodeActionResult:
    """Refactor selected code"""
    handler = get_action_handler()
    return await handler.execute_action(ActionType.REFACTOR, code, language=language)


async def generate_tests(code: str, language: str = "python") -> CodeActionResult:
    """Generate tests for selected code"""
    handler = get_action_handler()
    return await handler.execute_action(ActionType.TEST, code, language=language)


async def optimize_code(code: str, language: str = "python") -> CodeActionResult:
    """Optimize selected code"""
    handler = get_action_handler()
    return await handler.execute_action(ActionType.OPTIMIZE, code, language=language)


async def document_code(code: str, language: str = "python") -> CodeActionResult:
    """Generate documentation for selected code"""
    handler = get_action_handler()
    return await handler.execute_action(ActionType.DOCUMENT, code, language=language)


async def debug_code(code: str, language: str = "python") -> CodeActionResult:
    """Debug selected code"""
    handler = get_action_handler()
    return await handler.execute_action(ActionType.DEBUG, code, language=language)


if __name__ == "__main__":
    # Demo
    async def demo():
        console.print("[bold blue]Code Action Hotkeys Demo[/bold blue]\n")
        
        handler = CodeActionHandler()
        
        test_code = """
def calculate_sum(numbers):
    total = 0
    for n in numbers:
        total += n
    return total
"""
        
        console.print("[bold]Test Code:[/bold]")
        console.print(test_code)
        console.print("\n" + "="*50 + "\n")
        
        # Test each action
        for action_type in ActionType:
            console.print(f"[bold]{action_type.value.upper()}:[/bold]")
            result = await handler.execute_action(action_type, test_code)
            console.print(result.result[:200] + "..." if len(result.result) > 200 else result.result)
            console.print()
    
    import asyncio
    asyncio.run(demo())
