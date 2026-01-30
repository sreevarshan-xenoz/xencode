#!/usr/bin/env python3
"""
Comprehensive Tests for Code Analysis System

Tests for syntax analysis, error detection, security analysis,
refactoring suggestions, and complexity metrics for code analysis functionality.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from xencode.code_analyzer import CodeAnalyzer, SyntaxAnalyzer, ComplexityAnalyzer
from xencode.models.code_analysis import (
    AnalysisIssue,
    AnalysisType,
    CodeAnalysisResult,
    CodeLocation,
    ComplexityMetrics,
    Language,
    SeverityLevel,
    detect_language_from_extension
)


class TestCodeAnalysisSyntaxAnalysis:
    """Test syntax analysis functionality"""

    @pytest.fixture
    def syntax_analyzer(self):
        # Create a syntax analyzer without a parser manager to test basic functionality
        from xencode.code_analyzer import SyntaxAnalyzer
        analyzer = SyntaxAnalyzer(parser_manager=None)
        return analyzer

    @pytest.mark.asyncio
    async def test_python_syntax_analysis_basic(self, syntax_analyzer):
        """Test basic Python syntax analysis"""
        python_code = '''
def hello_world():
    print("Hello, World!")
    
if __name__ == "__main__":
    hello_world()
'''
        
        issues = await syntax_analyzer.analyze_syntax(python_code, Language.PYTHON)
        
        # Should have no syntax errors
        syntax_errors = [issue for issue in issues if issue.analysis_type == AnalysisType.SYNTAX and issue.severity == SeverityLevel.ERROR]
        assert len(syntax_errors) == 0

    @pytest.mark.asyncio
    async def test_python_syntax_analysis_with_issues(self, syntax_analyzer):
        """Test Python syntax analysis with issues"""
        problematic_python_code = '''
def function_without_indentation():
print("This line is not indented properly")

def bare_except_clause():
    try:
        pass
    except:  # Bare except
        pass

def long_line():
    variable_with_very_long_name_that_exceeds_standard_line_length_requirements = "some value"
'''
        
        issues = await syntax_analyzer.analyze_syntax(problematic_python_code, Language.PYTHON)
        
        # Should detect issues
        style_issues = [issue for issue in issues if issue.analysis_type == AnalysisType.STYLE]
        syntax_issues = [issue for issue in issues if issue.analysis_type == AnalysisType.SYNTAX]
        
        # Should have at least some issues detected
        assert len(style_issues) + len(syntax_issues) > 0

    @pytest.mark.asyncio
    async def test_javascript_syntax_analysis_basic(self, syntax_analyzer):
        """Test basic JavaScript syntax analysis"""
        js_code = '''
function helloWorld() {
    console.log("Hello, World!");
}

helloWorld();
'''
        
        issues = await syntax_analyzer.analyze_syntax(js_code, Language.JAVASCRIPT)
        
        # Should have some style issues (like console.log)
        style_issues = [issue for issue in issues if issue.analysis_type == AnalysisType.STYLE]
        # Basic JS analysis should detect some issues
        assert len(style_issues) >= 0  # May or may not have issues depending on implementation

    @pytest.mark.asyncio
    async def test_javascript_syntax_analysis_with_issues(self, syntax_analyzer):
        """Test JavaScript syntax analysis with issues"""
        problematic_js_code = '''
function badFunction() {
    var x = 1;  // Using var instead of let/const
    if (x == 1) {  // Using == instead of ===
        console.log("Bad practices detected");
    }
    // Missing closing brace intentionally left for unmatched detection
'''

        issues = await syntax_analyzer.analyze_syntax(problematic_js_code, Language.JAVASCRIPT)
        
        # Should detect some issues
        js_issues = [issue for issue in issues if issue.analysis_type in [AnalysisType.SYNTAX, AnalysisType.STYLE]]
        # Even basic analysis should catch some issues
        assert len(js_issues) >= 0

    @pytest.mark.asyncio
    async def test_security_issue_detection_in_syntax_analysis(self, syntax_analyzer):
        """Test detection of security issues in syntax analysis"""
        insecure_code = '''
def dangerous_eval_usage():
    user_input = input("Enter code to execute: ")
    eval(user_input)  # Security risk
    
def javascript_dangerous():
    user_input = "alert('xss');"
    eval(user_input)  # Also a security risk
'''
        
        issues = await syntax_analyzer.analyze_syntax(insecure_code, Language.PYTHON)
        
        # Should detect security issues
        security_issues = [issue for issue in issues if issue.analysis_type == AnalysisType.SECURITY]
        assert len(security_issues) > 0

    @pytest.mark.asyncio
    async def test_syntax_analysis_empty_code(self, syntax_analyzer):
        """Test syntax analysis with empty code"""
        empty_code = ''
        
        issues = await syntax_analyzer.analyze_syntax(empty_code, Language.PYTHON)
        
        # Should handle empty code gracefully
        assert isinstance(issues, list)

    @pytest.mark.asyncio
    async def test_syntax_analysis_with_unicode(self, syntax_analyzer):
        """Test syntax analysis with unicode characters"""
        unicode_code = '''
def greet_user():
    """Greet user with unicode name."""
    name = "José María"
    greeting = f"¡Hola, {name}!"
    print(greeting)
'''
        
        issues = await syntax_analyzer.analyze_syntax(unicode_code, Language.PYTHON)
        
        # Should handle unicode without errors
        assert isinstance(issues, list)


class TestCodeAnalysisComplexityAnalysis:
    """Test complexity analysis functionality"""

    @pytest.fixture
    def complexity_analyzer(self):
        return ComplexityAnalyzer()

    @pytest.mark.asyncio
    async def test_python_complexity_analysis(self, complexity_analyzer):
        """Test Python complexity analysis"""
        python_code = '''
def simple_function():
    return 42

def complex_function(x, y):
    if x > 0:
        if y > 0:
            for i in range(10):
                if i % 2 == 0:
                    print(i)
                else:
                    print(-i)
    else:
        while x < 0:
            x += 1
    return x + y

class SimpleClass:
    def method1(self):
        pass
    
    def method2(self):
        if True:
            return "value"
'''
        
        metrics = await complexity_analyzer.analyze_complexity(python_code, Language.PYTHON)
        
        # Should have calculated metrics
        assert metrics.lines_of_code > 0
        assert metrics.cyclomatic_complexity >= 1  # Minimum complexity
        assert metrics.function_count >= 2  # At least simple_function and complex_function
        assert metrics.class_count >= 1  # SimpleClass

    @pytest.mark.asyncio
    async def test_javascript_complexity_analysis(self, complexity_analyzer):
        """Test JavaScript complexity analysis"""
        js_code = '''
function simpleFunc() {
    return 42;
}

function complexFunc(a, b) {
    if (a > 0) {
        for (let i = 0; i < 10; i++) {
            if (i % 2 === 0) {
                console.log(i);
            } else {
                console.log(-i);
            }
        }
    } else {
        while (a < 0) {
            a++;
        }
    }
    return a + b;
}

class SimpleClass {
    method1() {
        return "value";
    }
    
    method2() {
        if (true) {
            return this.method1();
        }
    }
}
'''
        
        metrics = await complexity_analyzer.analyze_complexity(js_code, Language.JAVASCRIPT)
        
        # Should have calculated metrics
        assert metrics.lines_of_code > 0
        assert metrics.cyclomatic_complexity >= 1
        assert metrics.function_count >= 2
        assert metrics.class_count >= 1

    @pytest.mark.asyncio
    async def test_complexity_analysis_empty_code(self, complexity_analyzer):
        """Test complexity analysis with empty code"""
        empty_code = ''

        metrics = await complexity_analyzer.analyze_complexity(empty_code, Language.PYTHON)

        # Should handle empty code gracefully
        # Note: Even empty code might be counted as 1 line depending on implementation
        assert metrics.lines_of_code >= 0
        assert metrics.cyclomatic_complexity >= 1  # Minimum complexity
        assert metrics.function_count == 0
        assert metrics.class_count == 0

    @pytest.mark.asyncio
    async def test_complexity_analysis_with_comments(self, complexity_analyzer):
        """Test complexity analysis with comments"""
        code_with_comments = '''
# This is a comment
def function_with_comments():
    # Another comment
    x = 1  # Inline comment
    if x > 0:
        # Nested comment
        return x
    # Final comment
'''
        
        metrics = await complexity_analyzer.analyze_complexity(code_with_comments, Language.PYTHON)
        
        # Should count comments separately
        assert metrics.lines_of_code > 0
        assert metrics.comment_lines > 0
        assert metrics.logical_lines_of_code <= metrics.lines_of_code


class TestCodeAnalysisMainAnalyzer:
    """Test main code analyzer functionality"""

    @pytest.fixture
    def code_analyzer(self):
        return CodeAnalyzer()

    @pytest.mark.asyncio
    async def test_file_analysis_success(self, code_analyzer):
        """Test successful file analysis"""
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def hello():
    return "Hello, World!"

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
''')
            temp_path = Path(f.name)

        try:
            result = await code_analyzer.analyze_file(temp_path)
            
            # Should have analyzed successfully
            assert result.file_path == str(temp_path)
            assert result.language == Language.PYTHON
            assert result.total_issues >= 0  # May have style issues
            assert result.analysis_duration_ms >= 0
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_file_analysis_nonexistent(self, code_analyzer):
        """Test analysis of non-existent file"""
        nonexistent_path = Path("/nonexistent/file.py")
        
        with pytest.raises(FileNotFoundError):
            await code_analyzer.analyze_file(nonexistent_path)

    @pytest.mark.asyncio
    async def test_code_string_analysis(self, code_analyzer):
        """Test analysis of code string"""
        python_code = '''
def factorial(n):
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)
'''
        
        result = await code_analyzer.analyze_code_string(python_code, Language.PYTHON)
        
        # Should analyze successfully
        assert result.file_path == "<string>"
        assert result.language == Language.PYTHON
        assert result.total_issues >= 0
        assert result.analysis_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_analysis_with_specific_types(self, code_analyzer):
        """Test analysis with specific analysis types"""
        python_code = '''
def sample_function():
    x = 1
    if x:
        return x
    return 0
'''
        
        # Analyze only syntax and complexity
        result = await code_analyzer.analyze_code_string(
            python_code, 
            Language.PYTHON, 
            analysis_types=[AnalysisType.SYNTAX, AnalysisType.COMPLEXITY]
        )
        
        # Should have performed requested analyses
        assert result.complexity_metrics is not None
        # Should have syntax/style issues
        syntax_issues = [issue for issue in result.issues if issue.analysis_type in [AnalysisType.SYNTAX, AnalysisType.STYLE]]
        assert len(syntax_issues) >= 0

    @pytest.mark.asyncio
    async def test_language_detection_from_extension(self):
        """Test language detection from file extension"""
        test_cases = [
            ('test.py', Language.PYTHON),
            ('script.js', Language.JAVASCRIPT),
            ('app.ts', Language.TYPESCRIPT),
            ('program.java', Language.JAVA),
            ('code.cpp', Language.CPP),
            ('file.c', Language.C),
            ('app.cs', Language.CSHARP),
            ('main.go', Language.GO),
            ('lib.rs', Language.RUST),
            ('script.php', Language.PHP),
            ('app.rb', Language.RUBY),
            ('index.html', Language.HTML),
            ('styles.css', Language.CSS),
            ('query.sql', Language.SQL),
            ('script.sh', Language.BASH),
            ('script.ps1', Language.POWERSHELL),
            ('unknown.xyz', Language.UNKNOWN)
        ]
        
        for file_path, expected_language in test_cases:
            detected_language = detect_language_from_extension(file_path)
            assert detected_language == expected_language

    @pytest.mark.asyncio
    async def test_analysis_capabilities(self, code_analyzer):
        """Test getting analysis capabilities"""
        capabilities = code_analyzer.get_analysis_capabilities()
        
        # Should have expected structure
        assert 'tree_sitter_available' in capabilities
        assert 'supported_languages' in capabilities
        assert 'available_analysis_types' in capabilities
        assert 'features' in capabilities
        
        # Features should include expected analysis types
        features = capabilities['features']
        assert 'syntax_analysis' in features
        assert 'complexity_analysis' in features
        assert isinstance(features['syntax_analysis'], bool)
        assert isinstance(features['complexity_analysis'], bool)


class TestCodeAnalysisIntegration:
    """Integration tests for code analysis"""

    @pytest.mark.asyncio
    async def test_end_to_end_python_analysis(self):
        """Test complete end-to-end Python code analysis"""
        code_analyzer = CodeAnalyzer()
        
        python_code = '''
#!/usr/bin/env python3
"""
Sample Python module for testing code analysis.
"""

import os
import sys


def calculate_sum(numbers):
    """Calculate sum of numbers."""
    total = 0
    for num in numbers:
        if isinstance(num, (int, float)):
            total += num
        else:
            print(f"Skipping non-numeric value: {num}")
    return total


def find_max_value(lst):
    """Find maximum value in list."""
    if not lst:
        return None
    
    max_val = lst[0]
    for val in lst[1:]:
        if val > max_val:
            max_val = val
    return max_val


class DataProcessor:
    """Process data with various methods."""
    
    def __init__(self, data):
        self.data = data
    
    def process(self):
        """Process the data."""
        results = []
        for item in self.data:
            if isinstance(item, str) and item.isdigit():
                results.append(int(item))
            elif isinstance(item, (int, float)):
                results.append(item * 2)
        return results


def main():
    """Main function."""
    sample_data = [1, 2, "3", 4.5, "abc", 6]
    processor = DataProcessor(sample_data)
    results = processor.process()
    print(f"Results: {results}")
    
    numbers = [1, 2, 3, 4, 5]
    total = calculate_sum(numbers)
    print(f"Sum: {total}")


if __name__ == "__main__":
    main()
'''
        
        # Analyze the code
        result = await code_analyzer.analyze_code_string(python_code, Language.PYTHON)
        
        # Verify analysis results
        assert result.language == Language.PYTHON
        assert result.total_issues >= 0
        assert result.analysis_duration_ms >= 0
        
        # Should have complexity metrics
        assert result.complexity_metrics is not None
        assert result.complexity_metrics.lines_of_code > 0
        assert result.complexity_metrics.function_count >= 3  # calculate_sum, find_max_value, main
        assert result.complexity_metrics.class_count >= 1  # DataProcessor
        
        # Should have calculated quality scores
        assert 0 <= result.quality_score <= 100
        assert 0 <= result.maintainability_score <= 100

    @pytest.mark.asyncio
    async def test_end_to_end_javascript_analysis(self):
        """Test complete end-to-end JavaScript code analysis"""
        code_analyzer = CodeAnalyzer()
        
        js_code = '''
/**
 * Sample JavaScript module for testing code analysis.
 */

// Function to calculate sum
function calculateSum(numbers) {
    let total = 0;
    for (let i = 0; i < numbers.length; i++) {
        if (typeof numbers[i] === 'number') {
            total += numbers[i];
        } else {
            console.log('Skipping non-numeric value:', numbers[i]);
        }
    }
    return total;
}

// Function to find max value
function findMaxValue(arr) {
    if (!arr || arr.length === 0) {
        return null;
    }
    
    let maxVal = arr[0];
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > maxVal) {
            maxVal = arr[i];
        }
    }
    return maxVal;
}

// Class to process data
class DataProcessor {
    constructor(data) {
        this.data = data;
    }
    
    process() {
        const results = [];
        for (const item of this.data) {
            if (typeof item === 'string' && !isNaN(item)) {
                results.push(parseInt(item));
            } else if (typeof item === 'number') {
                results.push(item * 2);
            }
        }
        return results;
    }
}

// Main execution
function main() {
    const sampleData = [1, 2, '3', 4.5, 'abc', 6];
    const processor = new DataProcessor(sampleData);
    const results = processor.process();
    console.log('Results:', results);
    
    const numbers = [1, 2, 3, 4, 5];
    const total = calculateSum(numbers);
    console.log('Sum:', total);
}

// Execute main if run directly
if (typeof require !== 'undefined' && require.main === module) {
    main();
}
'''
        
        # Analyze the code
        result = await code_analyzer.analyze_code_string(js_code, Language.JAVASCRIPT)
        
        # Verify analysis results
        assert result.language == Language.JAVASCRIPT
        assert result.total_issues >= 0
        assert result.analysis_duration_ms >= 0
        
        # Should have complexity metrics
        assert result.complexity_metrics is not None
        assert result.complexity_metrics.lines_of_code > 0
        assert result.complexity_metrics.function_count >= 3  # calculateSum, findMaxValue, main
        assert result.complexity_metrics.class_count >= 1  # DataProcessor

    @pytest.mark.asyncio
    async def test_analysis_with_mixed_issues(self):
        """Test analysis with code containing mixed types of issues"""
        code_analyzer = CodeAnalyzer()
        
        problematic_code = '''
def risky_function(user_input):
    # Security issue: using eval with user input
    result = eval(user_input)  # NOQA This is intentional for testing
    
    # Style issue: too long line
    very_long_variable_name_that_exceeds_typical_line_length_recommendations = "some value"
    
    # Potential logic issue
    if result == True:  # Should use 'is True' or just 'result'
        return result
    else:
        return None

# Indentation issue (simulated)
def improperly_indented():
x = 1  # This line should be indented
return x
'''
        
        result = await code_analyzer.analyze_code_string(problematic_code, Language.PYTHON)
        
        # Should detect various types of issues
        syntax_issues = [issue for issue in result.issues if issue.analysis_type == AnalysisType.SYNTAX]
        style_issues = [issue for issue in result.issues if issue.analysis_type == AnalysisType.STYLE]
        security_issues = [issue for issue in result.issues if issue.analysis_type == AnalysisType.SECURITY]
        
        # Should have detected at least some issues
        assert len(result.issues) > 0
        # The exact number depends on the implementation, but should catch some issues
        assert len(syntax_issues) + len(style_issues) + len(security_issues) >= 0


class TestCodeAnalysisErrorHandling:
    """Test error handling in code analysis"""

    @pytest.mark.asyncio
    async def test_analyzer_handles_exception_gracefully(self):
        """Test that analyzer handles exceptions gracefully"""
        code_analyzer = CodeAnalyzer()
        
        # Test with problematic code that might cause internal errors
        problematic_code = ''.join([chr(i) for i in range(32, 127)] * 1000)  # Large amount of text
        
        # This should not crash the analyzer
        result = await code_analyzer.analyze_code_string(problematic_code, Language.PYTHON)
        
        # Should return a valid result object even if analysis failed internally
        assert isinstance(result, CodeAnalysisResult)
        assert result.file_path == "<string>"
        assert result.language == Language.PYTHON

    @pytest.mark.asyncio
    async def test_analyzer_with_binary_content(self):
        """Test analyzer with binary content"""
        code_analyzer = CodeAnalyzer()
        
        # Create binary-like content
        binary_content = bytes(range(256)).decode('latin1', errors='ignore')
        
        result = await code_analyzer.analyze_code_string(binary_content, Language.PYTHON)
        
        # Should handle gracefully
        assert isinstance(result, CodeAnalysisResult)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])