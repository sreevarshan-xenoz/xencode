#!/usr/bin/env python3
"""
Unit tests for Code Action Hotkeys
"""

import pytest

from xencode.tui.widgets.code_action_hotkeys import (
    ActionType,
    CodeAction,
    CodeActionResult,
    CodeActionPrompts,
    CodeActionHandler,
    HOTKEY_MAP,
    HOTKEY_HELP,
    get_action_handler,
    explain_code,
    refactor_code,
    generate_tests,
    optimize_code,
    document_code,
    debug_code,
)


class TestActionType:
    """Tests for ActionType enum"""
    
    def test_action_types(self):
        """Test all action types exist"""
        assert ActionType.EXPLAIN.value == "explain"
        assert ActionType.REFACTOR.value == "refactor"
        assert ActionType.TEST.value == "test"
        assert ActionType.OPTIMIZE.value == "optimize"
        assert ActionType.DOCUMENT.value == "document"
        assert ActionType.DEBUG.value == "debug"


class TestCodeAction:
    """Tests for CodeAction dataclass"""
    
    def test_action_creation(self):
        """Test creating code action"""
        action = CodeAction(
            action_type=ActionType.EXPLAIN,
            code_snippet="print('hello')",
            language="python",
        )
        
        assert action.action_type == ActionType.EXPLAIN
        assert action.code_snippet == "print('hello')"
        assert action.language == "python"
    
    def test_action_to_dict(self):
        """Test converting action to dictionary"""
        action = CodeAction(
            action_type=ActionType.REFACTOR,
            code_snippet="x = 1",
            file_path="test.py",
            line_start=1,
            line_end=2,
        )
        
        d = action.to_dict()
        
        assert d["action_type"] == "refactor"
        assert d["code_snippet"] == "x = 1"
        assert d["file_path"] == "test.py"
        assert d["line_start"] == 1


class TestCodeActionResult:
    """Tests for CodeActionResult dataclass"""
    
    def test_result_creation(self):
        """Test creating code action result"""
        result = CodeActionResult(
            action_type=ActionType.EXPLAIN,
            success=True,
            result="Explanation text",
        )
        
        assert result.action_type == ActionType.EXPLAIN
        assert result.success is True
        assert result.result == "Explanation text"
    
    def test_result_with_error(self):
        """Test creating failed result"""
        result = CodeActionResult(
            action_type=ActionType.TEST,
            success=False,
            error="LLM not available",
        )
        
        assert result.success is False
        assert result.error == "LLM not available"
    
    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        result = CodeActionResult(
            action_type=ActionType.OPTIMIZE,
            success=True,
            result="Optimized code",
            suggestions=["Use list comprehension"],
        )
        
        d = result.to_dict()
        
        assert d["action_type"] == "optimize"
        assert d["success"] is True
        assert len(d["suggestions"]) == 1


class TestCodeActionPrompts:
    """Tests for CodeActionPrompts"""
    
    def test_explain_prompt(self):
        """Test explain prompt template"""
        prompt = CodeActionPrompts.get_prompt(
            ActionType.EXPLAIN,
            "def hello(): pass",
            "python",
        )
        
        assert "Explain" in prompt
        assert "def hello(): pass" in prompt
        assert "python" in prompt
    
    def test_refactor_prompt(self):
        """Test refactor prompt template"""
        prompt = CodeActionPrompts.get_prompt(
            ActionType.REFACTOR,
            "x = 1",
            "python",
        )
        
        assert "Refactor" in prompt
        assert "readability" in prompt.lower()
    
    def test_test_prompt(self):
        """Test test generation prompt"""
        prompt = CodeActionPrompts.get_prompt(
            ActionType.TEST,
            "def add(a, b): return a + b",
            "python",
        )
        
        assert "test" in prompt.lower()
        assert "pytest" in prompt.lower()
    
    def test_optimize_prompt(self):
        """Test optimization prompt"""
        prompt = CodeActionPrompts.get_prompt(
            ActionType.OPTIMIZE,
            "for i in range(100): print(i)",
            "python",
        )
        
        assert "Optimize" in prompt
        assert "performance" in prompt.lower()
    
    def test_document_prompt(self):
        """Test documentation prompt"""
        prompt = CodeActionPrompts.get_prompt(
            ActionType.DOCUMENT,
            "def func(x): return x * 2",
            "python",
        )
        
        assert "documentation" in prompt.lower() or "Document" in prompt
        assert "Docstrings" in prompt or "docstring" in prompt.lower()
    
    def test_debug_prompt(self):
        """Test debug prompt"""
        prompt = CodeActionPrompts.get_prompt(
            ActionType.DEBUG,
            "result = 1 / 0",
            "python",
        )
        
        assert "bug" in prompt.lower() or "Debug" in prompt
        assert "issues" in prompt.lower()


class TestCodeActionHandler:
    """Tests for CodeActionHandler"""
    
    @pytest.fixture
    def handler(self):
        """Create action handler"""
        return CodeActionHandler()
    
    def test_handler_creation(self, handler):
        """Test handler initialization"""
        assert handler.default_language == "python"
        assert handler.model_callback is None
    
    @pytest.mark.asyncio
    async def test_execute_explain(self, handler):
        """Test explain action"""
        result = await handler.execute_action(
            ActionType.EXPLAIN,
            "print('hello')",
        )
        
        assert result.action_type == ActionType.EXPLAIN
        assert result.success is True
        assert len(result.result) > 0
    
    @pytest.mark.asyncio
    async def test_execute_refactor(self, handler):
        """Test refactor action"""
        result = await handler.execute_action(
            ActionType.REFACTOR,
            "x = 1\ny = 2",
        )
        
        assert result.action_type == ActionType.REFACTOR
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_execute_test(self, handler):
        """Test test generation"""
        result = await handler.execute_action(
            ActionType.TEST,
            "def add(a, b): return a + b",
        )
        
        assert result.action_type == ActionType.TEST
        assert result.success is True
        assert "pytest" in result.result.lower()
    
    @pytest.mark.asyncio
    async def test_execute_optimize(self, handler):
        """Test optimization"""
        result = await handler.execute_action(
            ActionType.OPTIMIZE,
            "for i in range(10): print(i)",
        )
        
        assert result.action_type == ActionType.OPTIMIZE
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_execute_document(self, handler):
        """Test documentation generation"""
        result = await handler.execute_action(
            ActionType.DOCUMENT,
            "def multiply(a, b): return a * b",
        )
        
        assert result.action_type == ActionType.DOCUMENT
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_execute_debug(self, handler):
        """Test debug analysis"""
        result = await handler.execute_action(
            ActionType.DEBUG,
            "result = 1 / 0",
        )
        
        assert result.action_type == ActionType.DEBUG
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_execute_with_file_path(self, handler):
        """Test action with file path context"""
        result = await handler.execute_action(
            ActionType.EXPLAIN,
            "code",
            file_path="test.py",
            line_start=10,
            line_end=20,
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_execute_with_custom_language(self, handler):
        """Test action with custom language"""
        result = await handler.execute_action(
            ActionType.EXPLAIN,
            "console.log('hello')",
            language="javascript",
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_handler_history(self, handler):
        """Test action history"""
        await handler.execute_action(ActionType.EXPLAIN, "code1")
        await handler.execute_action(ActionType.REFACTOR, "code2")
        
        history = handler.get_history()
        assert len(history) == 2
    
    @pytest.mark.asyncio
    async def test_handler_clear_history(self, handler):
        """Test clearing history"""
        await handler.execute_action(ActionType.EXPLAIN, "code")
        handler.clear_history()
        
        history = handler.get_history()
        assert len(history) == 0


class TestHotkeyMap:
    """Tests for hotkey mappings"""
    
    def test_hotkey_map(self):
        """Test hotkey mappings exist"""
        assert "e" in HOTKEY_MAP
        assert "r" in HOTKEY_MAP
        assert "t" in HOTKEY_MAP
        assert HOTKEY_MAP["e"] == ActionType.EXPLAIN
        assert HOTKEY_MAP["r"] == ActionType.REFACTOR
        assert HOTKEY_MAP["t"] == ActionType.TEST
    
    def test_hotkey_help(self):
        """Test hotkey help text"""
        assert "Explain" in HOTKEY_HELP
        assert "Refactor" in HOTKEY_HELP
        assert "Test" in HOTKEY_HELP or "tests" in HOTKEY_HELP


class TestGlobalFunctions:
    """Tests for global convenience functions"""
    
    @pytest.mark.asyncio
    async def test_explain_code(self):
        """Test explain_code function"""
        # Reset global handler
        import xencode.tui.widgets.code_action_hotkeys as cah
        cah._handler = None
        
        result = await explain_code("print('test')")
        
        assert result.action_type == ActionType.EXPLAIN
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_refactor_code(self):
        """Test refactor_code function"""
        import xencode.tui.widgets.code_action_hotkeys as cah
        cah._handler = None
        
        result = await refactor_code("x = 1")
        
        assert result.action_type == ActionType.REFACTOR
    
    @pytest.mark.asyncio
    async def test_generate_tests(self):
        """Test generate_tests function"""
        import xencode.tui.widgets.code_action_hotkeys as cah
        cah._handler = None
        
        result = await generate_tests("def add(a, b): return a + b")
        
        assert result.action_type == ActionType.TEST
    
    @pytest.mark.asyncio
    async def test_optimize_code(self):
        """Test optimize_code function"""
        import xencode.tui.widgets.code_action_hotkeys as cah
        cah._handler = None
        
        result = await optimize_code("for i in range(10): pass")
        
        assert result.action_type == ActionType.OPTIMIZE
    
    @pytest.mark.asyncio
    async def test_document_code(self):
        """Test document_code function"""
        import xencode.tui.widgets.code_action_hotkeys as cah
        cah._handler = None
        
        result = await document_code("def func(): pass")
        
        assert result.action_type == ActionType.DOCUMENT
    
    @pytest.mark.asyncio
    async def test_debug_code(self):
        """Test debug_code function"""
        import xencode.tui.widgets.code_action_hotkeys as cah
        cah._handler = None
        
        result = await debug_code("x = 1 / 0")
        
        assert result.action_type == ActionType.DEBUG


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
