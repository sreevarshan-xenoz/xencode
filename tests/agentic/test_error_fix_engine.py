#!/usr/bin/env python3
"""
Unit tests for Error Auto-Fix Suggestion Engine
"""

import pytest

from xencode.agentic.error_fix_engine import (
    ErrorFixEngine,
    ErrorCategory,
    FixConfidence,
    ErrorPattern,
    FixSuggestion,
    ErrorAnalysis,
    get_fix_engine,
    analyze_error,
    suggest_fix,
)


class TestErrorCategory:
    """Tests for ErrorCategory enum"""
    
    def test_category_values(self):
        """Test error category enum values"""
        assert ErrorCategory.SYNTAX.value == "syntax"
        assert ErrorCategory.NAME_ERROR.value == "name_error"
        assert ErrorCategory.TYPE_ERROR.value == "type_error"
        assert ErrorCategory.UNKNOWN.value == "unknown"


class TestFixConfidence:
    """Tests for FixConfidence enum"""
    
    def test_confidence_values(self):
        """Test confidence enum values"""
        assert FixConfidence.HIGH.value == "high"
        assert FixConfidence.MEDIUM.value == "medium"
        assert FixConfidence.LOW.value == "low"


class TestFixSuggestion:
    """Tests for FixSuggestion dataclass"""
    
    def test_suggestion_creation(self):
        """Test creating fix suggestion"""
        suggestion = FixSuggestion(
            title="Test Fix",
            description="Test description",
            fix_code="print('fix')",
            explanation="This explains the fix",
            confidence=FixConfidence.HIGH,
        )
        
        assert suggestion.title == "Test Fix"
        assert suggestion.description == "Test description"
        assert suggestion.fix_code == "print('fix')"
        assert suggestion.confidence == FixConfidence.HIGH
    
    def test_suggestion_to_dict(self):
        """Test converting suggestion to dictionary"""
        suggestion = FixSuggestion(
            title="Test",
            description="Desc",
            confidence=FixConfidence.MEDIUM,
            category=ErrorCategory.SYNTAX,
        )
        
        d = suggestion.to_dict()
        
        assert d["title"] == "Test"
        assert d["description"] == "Desc"
        assert d["confidence"] == "medium"
        assert d["category"] == "syntax"


class TestErrorAnalysis:
    """Tests for ErrorAnalysis dataclass"""
    
    def test_analysis_creation(self):
        """Test creating error analysis"""
        analysis = ErrorAnalysis(
            original_error="NameError: name 'x' is not defined",
            error_type="NameError",
            category=ErrorCategory.NAME_ERROR,
            message="name 'x' is not defined",
        )
        
        assert analysis.original_error == "NameError: name 'x' is not defined"
        assert analysis.error_type == "NameError"
        assert analysis.category == ErrorCategory.NAME_ERROR
        assert analysis.has_suggestions is False
    
    def test_analysis_with_suggestions(self):
        """Test analysis with suggestions"""
        suggestion = FixSuggestion(
            title="Define Variable",
            description="Define x before use",
            confidence=FixConfidence.HIGH,
        )
        
        analysis = ErrorAnalysis(
            original_error="NameError: name 'x' is not defined",
            error_type="NameError",
            category=ErrorCategory.NAME_ERROR,
            message="name 'x' is not defined",
            suggestions=[suggestion],
        )
        
        assert analysis.has_suggestions is True
        assert analysis.best_suggestion == suggestion
    
    def test_analysis_best_suggestion_priority(self):
        """Test best suggestion returns highest confidence"""
        suggestions = [
            FixSuggestion(title="Low", description="", confidence=FixConfidence.LOW),
            FixSuggestion(title="High", description="", confidence=FixConfidence.HIGH),
            FixSuggestion(title="Medium", description="", confidence=FixConfidence.MEDIUM),
        ]
        
        analysis = ErrorAnalysis(
            original_error="test",
            error_type="TestError",
            category=ErrorCategory.UNKNOWN,
            message="test",
            suggestions=suggestions,
        )
        
        assert analysis.best_suggestion.title == "High"
    
    def test_analysis_to_dict(self):
        """Test converting analysis to dictionary"""
        analysis = ErrorAnalysis(
            original_error="test error",
            error_type="TestError",
            category=ErrorCategory.SYNTAX,
            message="test message",
        )
        
        d = analysis.to_dict()
        
        assert d["original_error"] == "test error"
        assert d["error_type"] == "TestError"
        assert d["category"] == "syntax"
        assert d["has_suggestions"] is False


class TestErrorFixEngine:
    """Tests for ErrorFixEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create error fix engine"""
        return ErrorFixEngine()
    
    def test_engine_creation(self, engine):
        """Test engine initialization"""
        assert len(engine._compiled_patterns) > 0
    
    def test_analyze_name_error(self, engine):
        """Test analyzing NameError"""
        analysis = engine.analyze_error("NameError: name 'undefined_var' is not defined")
        
        assert analysis.error_type == "NameError"
        assert analysis.category == ErrorCategory.NAME_ERROR
        assert analysis.has_suggestions is True
    
    def test_analyze_type_error(self, engine):
        """Test analyzing TypeError"""
        analysis = engine.analyze_error(
            "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
        )
        
        assert analysis.error_type == "TypeError"
        assert analysis.category == ErrorCategory.TYPE_ERROR
        assert analysis.has_suggestions is True
    
    def test_analyze_syntax_error(self, engine):
        """Test analyzing SyntaxError"""
        analysis = engine.analyze_error("SyntaxError: invalid syntax")
        
        assert analysis.error_type == "SyntaxError"
        assert analysis.category == ErrorCategory.SYNTAX
        assert analysis.has_suggestions is True
    
    def test_analyze_file_not_found(self, engine):
        """Test analyzing FileNotFoundError"""
        analysis = engine.analyze_error(
            "FileNotFoundError: [Errno 2] No such file or directory: 'test.txt'"
        )
        
        assert analysis.category == ErrorCategory.FILE_ERROR
        assert analysis.has_suggestions is True
    
    def test_analyze_zero_division(self, engine):
        """Test analyzing ZeroDivisionError"""
        analysis = engine.analyze_error("ZeroDivisionError: division by zero")
        
        assert analysis.category == ErrorCategory.ZERO_DIVISION
        assert analysis.has_suggestions is True
    
    def test_analyze_import_error(self, engine):
        """Test analyzing ImportError"""
        analysis = engine.analyze_error("ModuleNotFoundError: No module named 'requests'")
        
        assert analysis.category == ErrorCategory.IMPORT_ERROR
        assert analysis.has_suggestions is True
        
        # Should have install suggestion
        install_suggestion = any(
            "install" in s.title.lower() or "pip" in (s.fix_code or "").lower()
            for s in analysis.suggestions
        )
        assert install_suggestion is True
    
    def test_analyze_index_error(self, engine):
        """Test analyzing IndexError"""
        analysis = engine.analyze_error("IndexError: list index out of range")
        
        assert analysis.category == ErrorCategory.INDEX_ERROR
        assert analysis.has_suggestions is True
    
    def test_analyze_key_error(self, engine):
        """Test analyzing KeyError"""
        analysis = engine.analyze_error("KeyError: 'missing_key'")
        
        assert analysis.category == ErrorCategory.KEY_ERROR
        assert analysis.has_suggestions is True
    
    def test_analyze_with_context(self, engine):
        """Test analyzing error with context"""
        context = {
            "file_path": "/path/to/file.py",
            "line_number": 42,
            "code_snippet": "print(undefined_var)",
        }
        
        analysis = engine.analyze_error(
            "NameError: name 'undefined_var' is not defined",
            context=context,
        )
        
        assert analysis.context == context
        
        # Should have context-specific suggestions
        context_suggestions = [
            s for s in analysis.suggestions
            if s.metadata.get("file_path") or s.metadata.get("line_number")
        ]
        assert len(context_suggestions) > 0
    
    def test_analyze_unknown_error(self, engine):
        """Test analyzing unknown error type"""
        analysis = engine.analyze_error("SomeUnknownError: something went wrong")
        
        assert analysis.category == ErrorCategory.UNKNOWN
        assert analysis.error_type == "SomeUnknownError"
    
    def test_get_error_summary(self, engine):
        """Test generating error summary"""
        analysis = engine.analyze_error("NameError: name 'x' is not defined")
        summary = engine.get_error_summary(analysis)
        
        assert "NameError" in summary
        assert "Suggested Fixes" in summary
    
    def test_extract_error_type(self, engine):
        """Test error type extraction"""
        error_type = engine._extract_error_type("ValueError: invalid literal")
        assert error_type == "ValueError"
    
    def test_extract_message(self, engine):
        """Test message extraction"""
        message = engine._extract_message("TypeError: unsupported operand")
        assert message == "unsupported operand"


class TestContextSpecificSuggestions:
    """Tests for context-specific suggestions"""
    
    @pytest.fixture
    def engine(self):
        """Create error fix engine"""
        return ErrorFixEngine()
    
    def test_file_path_context(self, engine):
        """Test file path context suggestions"""
        context = {"file_path": "/test/missing.txt"}
        
        analysis = engine.analyze_error(
            "FileNotFoundError: [Errno 2] No such file or directory",
            context=context,
        )
        
        file_suggestions = [
            s for s in analysis.suggestions
            if "missing.txt" in (s.fix_code or "")
        ]
        assert len(file_suggestions) > 0
    
    def test_variable_name_context(self, engine):
        """Test variable name extraction from NameError"""
        analysis = engine.analyze_error("NameError: name 'my_var' is not defined")
        
        # Should have NAME_ERROR category
        assert analysis.category == ErrorCategory.NAME_ERROR
        
        # Should have suggestions for name errors
        assert analysis.has_suggestions is True
        
        # Check that suggestions mention defining variables
        define_suggestions = [
            s for s in analysis.suggestions
            if "define" in s.title.lower() or "Define" in s.title
        ]
        assert len(define_suggestions) > 0
    
    def test_module_name_context(self, engine):
        """Test module name extraction from ImportError"""
        analysis = engine.analyze_error("ModuleNotFoundError: No module named 'pandas'")
        
        # Should have IMPORT_ERROR category
        assert analysis.category == ErrorCategory.IMPORT_ERROR
        
        # Should have suggestions for import errors
        assert analysis.has_suggestions is True
        
        # Check that suggestions mention installing packages
        install_suggestions = [
            s for s in analysis.suggestions
            if "install" in s.title.lower() or "Install" in s.title
        ]
        assert len(install_suggestions) > 0


class TestGlobalFunctions:
    """Tests for global convenience functions"""
    
    def test_get_fix_engine_singleton(self):
        """Test get_fix_engine returns singleton"""
        # Reset
        import xencode.agentic.error_fix_engine as efe
        efe._engine = None
        
        engine1 = get_fix_engine()
        engine2 = get_fix_engine()
        
        assert engine1 is engine2
    
    def test_analyze_error_function(self):
        """Test analyze_error convenience function"""
        # Reset
        import xencode.agentic.error_fix_engine as efe
        efe._engine = None
        
        analysis = analyze_error("NameError: name 'test' is not defined")
        
        assert analysis.error_type == "NameError"
        assert analysis.category == ErrorCategory.NAME_ERROR
    
    def test_suggest_fix_function(self):
        """Test suggest_fix convenience function"""
        # Reset
        import xencode.agentic.error_fix_engine as efe
        efe._engine = None
        
        suggestion = suggest_fix("ZeroDivisionError: division by zero")
        
        assert suggestion is not None
        assert suggestion.confidence == FixConfidence.HIGH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
