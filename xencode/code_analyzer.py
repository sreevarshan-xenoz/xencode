#!/usr/bin/env python3
"""
Advanced Code Analyzer

Provides comprehensive code analysis using tree-sitter for syntax parsing,
with support for multiple programming languages, complexity analysis,
security scanning, and refactoring suggestions.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    tree_sitter = None
    Language = None
    Parser = None

from xencode.models.code_analysis import (
    AnalysisIssue,
    AnalysisType,
    CodeAnalysisResult,
    CodeLocation,
    ComplexityMetrics,
    Language,
    SeverityLevel,
    detect_language_from_extension,
    get_tree_sitter_language_name
)

# Import specialized analyzers
try:
    from xencode.analyzers.security_analyzer import SecurityAnalyzer
    SECURITY_ANALYZER_AVAILABLE = True
except ImportError:
    SECURITY_ANALYZER_AVAILABLE = False

try:
    from xencode.analyzers.error_detector import ErrorDetector
    ERROR_DETECTOR_AVAILABLE = True
except ImportError:
    ERROR_DETECTOR_AVAILABLE = False

try:
    from xencode.analyzers.refactoring_engine import RefactoringEngine
    REFACTORING_ENGINE_AVAILABLE = True
except ImportError:
    REFACTORING_ENGINE_AVAILABLE = False


class TreeSitterParserManager:
    """Manages tree-sitter parsers for different languages"""
    
    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter is required for code analysis. "
                "Install with: pip install tree-sitter"
            )
        
        self.parsers: Dict[Language, Parser] = {}
        self.languages: Dict[Language, 'tree_sitter.Language'] = {}
        self._initialize_available_parsers()
    
    def _initialize_available_parsers(self) -> None:
        """Initialize available tree-sitter parsers"""
        # This would normally load compiled language libraries
        # For now, we'll create a basic structure that can be extended
        
        # Note: In a real implementation, you would need to:
        # 1. Install tree-sitter language packages
        # 2. Compile language libraries
        # 3. Load them here
        
        # Example for Python (if available):
        try:
            # This is a placeholder - actual implementation would load compiled languages
            # python_lang = Language(library_path, 'python')
            # self.languages[Language.PYTHON] = python_lang
            # parser = Parser()
            # parser.set_language(python_lang)
            # self.parsers[Language.PYTHON] = parser
            pass
        except Exception:
            # Language not available
            pass
    
    def get_parser(self, language: Language) -> Optional[Parser]:
        """Get parser for specified language"""
        return self.parsers.get(language)
    
    def is_language_supported(self, language: Language) -> bool:
        """Check if language is supported"""
        return language in self.parsers
    
    def get_supported_languages(self) -> List[Language]:
        """Get list of supported languages"""
        return list(self.parsers.keys())


class SyntaxAnalyzer:
    """Analyzes code syntax using tree-sitter"""
    
    def __init__(self, parser_manager: TreeSitterParserManager):
        self.parser_manager = parser_manager
    
    async def analyze_syntax(self, 
                           code: str, 
                           language: Language,
                           file_path: str = "") -> List[AnalysisIssue]:
        """Analyze code syntax and return issues"""
        issues = []
        
        try:
            parser = self.parser_manager.get_parser(language) if self.parser_manager else None
            if not parser:
                # Fallback to basic syntax analysis
                return await self._basic_syntax_analysis(code, language, file_path)
            
            # Parse code with tree-sitter
            tree = parser.parse(bytes(code, 'utf8'))
            
            # Check for syntax errors
            if tree.root_node.has_error:
                issues.extend(await self._extract_syntax_errors(tree, code, file_path))
            
            # Analyze AST for other issues
            issues.extend(await self._analyze_ast(tree, code, language, file_path))
            
        except Exception as e:
            # Fallback to basic analysis if tree-sitter fails
            issues.append(AnalysisIssue(
                analysis_type=AnalysisType.SYNTAX,
                severity=SeverityLevel.WARNING,
                message=f"Tree-sitter analysis failed: {str(e)}",
                description="Falling back to basic syntax analysis"
            ))
            issues.extend(await self._basic_syntax_analysis(code, language, file_path))
        
        return issues
    
    async def _extract_syntax_errors(self, 
                                   tree: 'tree_sitter.Tree', 
                                   code: str,
                                   file_path: str) -> List[AnalysisIssue]:
        """Extract syntax errors from tree-sitter parse tree"""
        issues = []
        lines = code.split('\n')
        
        def find_error_nodes(node):
            """Recursively find error nodes"""
            if node.type == 'ERROR':
                line_num = node.start_point[0] + 1
                col_num = node.start_point[1] + 1
                
                # Get code snippet around error
                start_line = max(0, line_num - 2)
                end_line = min(len(lines), line_num + 1)
                code_snippet = '\n'.join(lines[start_line:end_line])
                
                issue = AnalysisIssue(
                    analysis_type=AnalysisType.SYNTAX,
                    severity=SeverityLevel.ERROR,
                    message="Syntax error detected",
                    description=f"Parse error at line {line_num}, column {col_num}",
                    location=CodeLocation(
                        line=line_num,
                        column=col_num,
                        end_line=node.end_point[0] + 1,
                        end_column=node.end_point[1] + 1
                    ),
                    code_snippet=code_snippet,
                    affected_code=node.text.decode('utf8') if node.text else ""
                )
                issues.append(issue)
            
            for child in node.children:
                find_error_nodes(child)
        
        find_error_nodes(tree.root_node)
        return issues
    
    async def _analyze_ast(self, 
                         tree: 'tree_sitter.Tree', 
                         code: str,
                         language: Language,
                         file_path: str) -> List[AnalysisIssue]:
        """Analyze AST for various issues"""
        issues = []
        
        # This would contain language-specific AST analysis
        # For now, we'll implement basic checks
        
        if language == Language.PYTHON:
            issues.extend(await self._analyze_python_ast(tree, code, file_path))
        elif language == Language.JAVASCRIPT:
            issues.extend(await self._analyze_javascript_ast(tree, code, file_path))
        
        return issues
    
    async def _analyze_python_ast(self, 
                                tree: 'tree_sitter.Tree', 
                                code: str,
                                file_path: str) -> List[AnalysisIssue]:
        """Python-specific AST analysis"""
        issues = []
        
        # This would contain comprehensive Python AST analysis
        # For now, we'll implement basic checks that work without tree-sitter
        lines = code.split('\n')
        issues.extend(await self._basic_python_checks(lines, file_path))
        
        return issues
    
    async def _analyze_javascript_ast(self, 
                                    tree: 'tree_sitter.Tree', 
                                    code: str,
                                    file_path: str) -> List[AnalysisIssue]:
        """JavaScript-specific AST analysis"""
        issues = []
        
        # This would contain comprehensive JavaScript AST analysis
        # For now, we'll implement basic checks that work without tree-sitter
        lines = code.split('\n')
        issues.extend(await self._basic_javascript_checks(lines, file_path))
        
        return issues
    
    async def _basic_syntax_analysis(self, 
                                   code: str, 
                                   language: Language,
                                   file_path: str) -> List[AnalysisIssue]:
        """Basic syntax analysis without tree-sitter"""
        issues = []
        lines = code.split('\n')
        
        # Basic checks based on language
        if language == Language.PYTHON:
            issues.extend(await self._basic_python_checks(lines, file_path))
        elif language == Language.JAVASCRIPT:
            issues.extend(await self._basic_javascript_checks(lines, file_path))
        
        return issues
    
    async def _basic_python_checks(self, lines: List[str], file_path: str) -> List[AnalysisIssue]:
        """Basic Python syntax checks"""
        issues = []
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for common Python issues
            if line_stripped.endswith(':') and not line_stripped.startswith('#'):
                # Check if next line is properly indented
                if line_num < len(lines):
                    next_line = lines[line_num].rstrip()
                    if next_line and not next_line.startswith(' ') and not next_line.startswith('\t'):
                        issues.append(AnalysisIssue(
                            analysis_type=AnalysisType.SYNTAX,
                            severity=SeverityLevel.ERROR,
                            message="Expected indented block",
                            description=f"Line {line_num + 1} should be indented after colon",
                            location=CodeLocation(line=line_num + 1, column=1),
                            code_snippet=f"{line}\n{next_line}"
                        ))
            
            # Check for mixed tabs and spaces (basic check)
            if '\t' in line and '    ' in line:
                issues.append(AnalysisIssue(
                    analysis_type=AnalysisType.STYLE,
                    severity=SeverityLevel.WARNING,
                    message="Mixed tabs and spaces",
                    description="Inconsistent indentation detected",
                    location=CodeLocation(line=line_num, column=1),
                    code_snippet=line,
                    suggested_fix="Use either tabs or spaces consistently"
                ))
            
            # Check for potential security issues
            if 'eval(' in line and not line_stripped.startswith('#'):
                issues.append(AnalysisIssue(
                    analysis_type=AnalysisType.SECURITY,
                    severity=SeverityLevel.WARNING,
                    message="Use of eval() detected",
                    description="eval() can execute arbitrary code and is a security risk",
                    location=CodeLocation(line=line_num, column=line.find('eval(') + 1),
                    code_snippet=line.strip(),
                    suggested_fix="Consider using ast.literal_eval() for safe evaluation"
                ))
            
            # Check for bare except clauses
            if line_stripped == 'except:' or line_stripped.startswith('except:'):
                issues.append(AnalysisIssue(
                    analysis_type=AnalysisType.STYLE,
                    severity=SeverityLevel.WARNING,
                    message="Bare except clause",
                    description="Catching all exceptions can hide bugs",
                    location=CodeLocation(line=line_num, column=1),
                    code_snippet=line.strip(),
                    suggested_fix="Specify exception types: except Exception:"
                ))
            
            # Check for long lines (PEP 8)
            if len(line) > 88:  # Slightly more lenient than PEP 8's 79
                issues.append(AnalysisIssue(
                    analysis_type=AnalysisType.STYLE,
                    severity=SeverityLevel.INFO,
                    message="Line too long",
                    description=f"Line has {len(line)} characters (recommended: ≤88)",
                    location=CodeLocation(line=line_num, column=89),
                    code_snippet=line.strip()[:50] + "..." if len(line.strip()) > 50 else line.strip(),
                    suggested_fix="Break line into multiple lines"
                ))
        
        return issues
    
    async def _basic_javascript_checks(self, lines: List[str], file_path: str) -> List[AnalysisIssue]:
        """Basic JavaScript syntax checks"""
        issues = []
        
        brace_count = 0
        paren_count = 0
        bracket_count = 0
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Count braces, parentheses, brackets
            brace_count += line.count('{') - line.count('}')
            paren_count += line.count('(') - line.count(')')
            bracket_count += line.count('[') - line.count(']')
            
            # Check for var usage (prefer let/const)
            if 'var ' in line and not line_stripped.startswith('//') and not line_stripped.startswith('/*'):
                issues.append(AnalysisIssue(
                    analysis_type=AnalysisType.STYLE,
                    severity=SeverityLevel.WARNING,
                    message="Use 'let' or 'const' instead of 'var'",
                    description="'var' has function scope, prefer block-scoped 'let' or 'const'",
                    location=CodeLocation(line=line_num, column=line.find('var ') + 1),
                    code_snippet=line.strip(),
                    suggested_fix="Replace 'var' with 'let' or 'const'"
                ))
            
            # Check for == instead of ===
            if '==' in line and '===' not in line and '!=' in line and '!==' not in line:
                if not line_stripped.startswith('//') and not line_stripped.startswith('/*'):
                    issues.append(AnalysisIssue(
                        analysis_type=AnalysisType.STYLE,
                        severity=SeverityLevel.WARNING,
                        message="Use strict equality (===) instead of loose equality (==)",
                        description="Strict equality avoids type coercion issues",
                        location=CodeLocation(line=line_num, column=line.find('==') + 1),
                        code_snippet=line.strip(),
                        suggested_fix="Replace '==' with '===' and '!=' with '!=='"
                    ))
            
            # Check for console.log (should be removed in production)
            if 'console.log(' in line and not line_stripped.startswith('//'):
                issues.append(AnalysisIssue(
                    analysis_type=AnalysisType.STYLE,
                    severity=SeverityLevel.INFO,
                    message="console.log() statement found",
                    description="Remove console.log statements before production",
                    location=CodeLocation(line=line_num, column=line.find('console.log(') + 1),
                    code_snippet=line.strip(),
                    suggested_fix="Remove or replace with proper logging"
                ))
            
            # Check for potential security issues
            if 'eval(' in line and not line_stripped.startswith('//'):
                issues.append(AnalysisIssue(
                    analysis_type=AnalysisType.SECURITY,
                    severity=SeverityLevel.WARNING,
                    message="Use of eval() detected",
                    description="eval() can execute arbitrary code and is a security risk",
                    location=CodeLocation(line=line_num, column=line.find('eval(') + 1),
                    code_snippet=line.strip(),
                    suggested_fix="Avoid eval() or use JSON.parse() for safe parsing"
                ))
            
            # Check for missing semicolons (basic check)
            if (line_stripped.endswith(')') or line_stripped.endswith(']') or 
                line_stripped.endswith('}')) and not line_stripped.endswith(';'):
                # Skip if it's a control structure or function declaration
                if not any(keyword in line_stripped for keyword in ['if', 'for', 'while', 'function', 'class', 'else']):
                    issues.append(AnalysisIssue(
                        analysis_type=AnalysisType.STYLE,
                        severity=SeverityLevel.INFO,
                        message="Missing semicolon",
                        description="Consider adding semicolon for clarity",
                        location=CodeLocation(line=line_num, column=len(line.rstrip())),
                        code_snippet=line.strip(),
                        suggested_fix="Add semicolon at end of statement"
                    ))
        
        # Check for unmatched braces/parentheses
        if brace_count != 0:
            issues.append(AnalysisIssue(
                analysis_type=AnalysisType.SYNTAX,
                severity=SeverityLevel.ERROR,
                message="Unmatched braces",
                description=f"Missing {abs(brace_count)} {'closing' if brace_count > 0 else 'opening'} brace(s)"
            ))
        
        if paren_count != 0:
            issues.append(AnalysisIssue(
                analysis_type=AnalysisType.SYNTAX,
                severity=SeverityLevel.ERROR,
                message="Unmatched parentheses",
                description=f"Missing {abs(paren_count)} {'closing' if paren_count > 0 else 'opening'} parenthesis/parentheses"
            ))
        
        return issues


class ComplexityAnalyzer:
    """Analyzes code complexity metrics"""
    
    def __init__(self):
        pass
    
    async def analyze_complexity(self, 
                               code: str, 
                               language: Language,
                               file_path: str = "") -> ComplexityMetrics:
        """Analyze code complexity and return metrics"""
        
        lines = code.split('\n')
        metrics = ComplexityMetrics()
        
        # Basic line counting
        metrics.lines_of_code = len(lines)
        metrics.blank_lines = sum(1 for line in lines if not line.strip())
        metrics.comment_lines = await self._count_comment_lines(lines, language)
        metrics.logical_lines_of_code = metrics.lines_of_code - metrics.blank_lines - metrics.comment_lines
        
        # Language-specific complexity analysis
        if language == Language.PYTHON:
            await self._analyze_python_complexity(code, lines, metrics)
        elif language == Language.JAVASCRIPT:
            await self._analyze_javascript_complexity(code, lines, metrics)
        else:
            await self._analyze_generic_complexity(code, lines, metrics)
        
        return metrics
    
    async def _count_comment_lines(self, lines: List[str], language: Language) -> int:
        """Count comment lines based on language"""
        comment_count = 0
        
        if language == Language.PYTHON:
            in_multiline = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_multiline = not in_multiline
                    comment_count += 1
                elif in_multiline or stripped.startswith('#'):
                    comment_count += 1
        
        elif language in [Language.JAVASCRIPT, Language.JAVA, Language.CPP, Language.C]:
            in_multiline = False
            for line in lines:
                stripped = line.strip()
                if '/*' in stripped:
                    in_multiline = True
                    comment_count += 1
                elif '*/' in stripped:
                    in_multiline = False
                elif in_multiline or stripped.startswith('//'):
                    comment_count += 1
        
        return comment_count
    
    async def _analyze_python_complexity(self, 
                                       code: str, 
                                       lines: List[str], 
                                       metrics: ComplexityMetrics) -> None:
        """Analyze Python-specific complexity"""
        
        # Count functions and classes
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def '):
                metrics.function_count += 1
            elif stripped.startswith('class '):
                metrics.class_count += 1
        
        # Basic cyclomatic complexity (count decision points)
        decision_keywords = ['if', 'elif', 'for', 'while', 'except', 'and', 'or']
        for line in lines:
            for keyword in decision_keywords:
                metrics.cyclomatic_complexity += line.count(f' {keyword} ')
                metrics.cyclomatic_complexity += line.count(f'{keyword} ')
        
        # Add 1 for the main path
        metrics.cyclomatic_complexity += 1
        
        # Calculate nesting depth
        max_depth = 0
        current_depth = 0
        total_depth = 0
        depth_count = 0
        
        for line in lines:
            # Simple indentation-based depth calculation
            if line.strip():
                indent_level = (len(line) - len(line.lstrip())) // 4  # Assuming 4-space indents
                current_depth = indent_level
                max_depth = max(max_depth, current_depth)
                total_depth += current_depth
                depth_count += 1
        
        metrics.max_nesting_depth = max_depth
        metrics.avg_nesting_depth = total_depth / depth_count if depth_count > 0 else 0
        
        # Calculate average function complexity
        if metrics.function_count > 0:
            metrics.avg_function_complexity = metrics.cyclomatic_complexity / metrics.function_count
    
    async def _analyze_javascript_complexity(self, 
                                           code: str, 
                                           lines: List[str], 
                                           metrics: ComplexityMetrics) -> None:
        """Analyze JavaScript-specific complexity"""
        
        # Count functions
        for line in lines:
            if 'function' in line or '=>' in line:
                metrics.function_count += 1
        
        # Count classes (ES6)
        for line in lines:
            if line.strip().startswith('class '):
                metrics.class_count += 1
        
        # Basic cyclomatic complexity
        decision_keywords = ['if', 'else', 'for', 'while', 'switch', 'case', '&&', '||', '?']
        for line in lines:
            for keyword in decision_keywords:
                metrics.cyclomatic_complexity += line.count(keyword)
        
        metrics.cyclomatic_complexity += 1  # Main path
        
        # Calculate nesting depth based on braces
        max_depth = 0
        current_depth = 0
        total_depth = 0
        depth_count = 0
        
        for line in lines:
            current_depth += line.count('{') - line.count('}')
            max_depth = max(max_depth, current_depth)
            if line.strip():
                total_depth += current_depth
                depth_count += 1
        
        metrics.max_nesting_depth = max_depth
        metrics.avg_nesting_depth = total_depth / depth_count if depth_count > 0 else 0
        
        if metrics.function_count > 0:
            metrics.avg_function_complexity = metrics.cyclomatic_complexity / metrics.function_count
    
    async def _analyze_generic_complexity(self, 
                                        code: str, 
                                        lines: List[str], 
                                        metrics: ComplexityMetrics) -> None:
        """Generic complexity analysis for unsupported languages"""
        
        # Basic metrics only
        metrics.cyclomatic_complexity = 1  # Minimum complexity
        
        # Count potential decision points
        decision_indicators = ['if', 'for', 'while', '?', '&&', '||']
        for line in lines:
            for indicator in decision_indicators:
                metrics.cyclomatic_complexity += line.lower().count(indicator)


class CodeAnalyzer:
    """Main code analyzer with comprehensive analysis capabilities"""
    
    def __init__(self):
        try:
            self.parser_manager = TreeSitterParserManager()
            self.tree_sitter_available = True
        except ImportError:
            self.parser_manager = None
            self.tree_sitter_available = False
        
        # Always create syntax analyzer, even without tree-sitter
        self.syntax_analyzer = SyntaxAnalyzer(self.parser_manager)
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # Initialize specialized analyzers
        self.security_analyzer = SecurityAnalyzer() if SECURITY_ANALYZER_AVAILABLE else None
        self.error_detector = ErrorDetector() if ERROR_DETECTOR_AVAILABLE else None
        self.refactoring_engine = RefactoringEngine() if REFACTORING_ENGINE_AVAILABLE else None
    
    async def analyze_file(self, 
                          file_path: Union[str, Path],
                          analysis_types: Optional[List[AnalysisType]] = None) -> CodeAnalysisResult:
        """Analyze a code file and return comprehensive results"""
        
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Default to all analysis types
        if analysis_types is None:
            analysis_types = [AnalysisType.SYNTAX, AnalysisType.COMPLEXITY, AnalysisType.STYLE]
        
        start_time = time.time()
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except Exception as e:
            result = CodeAnalysisResult(
                file_path=str(file_path),
                language=Language.UNKNOWN
            )
            result.add_issue(AnalysisIssue(
                analysis_type=AnalysisType.SYNTAX,
                severity=SeverityLevel.ERROR,
                message=f"Failed to read file: {str(e)}"
            ))
            return result
        
        # Detect language
        language = detect_language_from_extension(file_path)
        
        # Create result object
        result = CodeAnalysisResult(
            file_path=str(file_path),
            language=language
        )
        
        try:
            # Perform requested analyses
            if (AnalysisType.SYNTAX in analysis_types or AnalysisType.STYLE in analysis_types) and self.syntax_analyzer:
                syntax_issues = await self.syntax_analyzer.analyze_syntax(code, language, str(file_path))
                for issue in syntax_issues:
                    result.add_issue(issue)
            
            if AnalysisType.COMPLEXITY in analysis_types:
                complexity_metrics = await self.complexity_analyzer.analyze_complexity(code, language, str(file_path))
                result.complexity_metrics = complexity_metrics
            
            # Security analysis
            if AnalysisType.SECURITY in analysis_types and self.security_analyzer:
                security_issues, security_objects = await self.security_analyzer.analyze_security(code, language, str(file_path))
                for issue in security_issues:
                    result.add_issue(issue)
                result.security_issues.extend(security_objects)
            
            # Error detection
            if self.error_detector and (AnalysisType.SYNTAX in analysis_types or AnalysisType.STYLE in analysis_types):
                error_issues = await self.error_detector.detect_errors(code, language, str(file_path))
                for issue in error_issues:
                    result.add_issue(issue)
            
            # Refactoring suggestions
            if AnalysisType.REFACTORING in analysis_types and self.refactoring_engine:
                refactoring_issues, refactoring_suggestions = await self.refactoring_engine.generate_refactoring_suggestions(code, language, str(file_path))
                for issue in refactoring_issues:
                    result.add_issue(issue)
                result.refactoring_suggestions.extend(refactoring_suggestions)
            
        except Exception as e:
            result.add_issue(AnalysisIssue(
                analysis_type=AnalysisType.SYNTAX,
                severity=SeverityLevel.ERROR,
                message=f"Analysis failed: {str(e)}"
            ))
        
        # Set analysis duration
        result.analysis_duration_ms = int((time.time() - start_time) * 1000)
        
        return result
    
    async def analyze_code_string(self, 
                                 code: str,
                                 language: Language,
                                 analysis_types: Optional[List[AnalysisType]] = None) -> CodeAnalysisResult:
        """Analyze code string directly"""
        
        if analysis_types is None:
            analysis_types = [AnalysisType.SYNTAX, AnalysisType.COMPLEXITY, AnalysisType.STYLE]
        
        start_time = time.time()
        
        result = CodeAnalysisResult(
            file_path="<string>",
            language=language
        )
        
        try:
            # Perform analyses
            if (AnalysisType.SYNTAX in analysis_types or AnalysisType.STYLE in analysis_types) and self.syntax_analyzer:
                syntax_issues = await self.syntax_analyzer.analyze_syntax(code, language, "<string>")
                for issue in syntax_issues:
                    result.add_issue(issue)
            
            if AnalysisType.COMPLEXITY in analysis_types:
                complexity_metrics = await self.complexity_analyzer.analyze_complexity(code, language)
                result.complexity_metrics = complexity_metrics
            
            # Security analysis
            if AnalysisType.SECURITY in analysis_types and self.security_analyzer:
                security_issues, security_objects = await self.security_analyzer.analyze_security(code, language, "<string>")
                for issue in security_issues:
                    result.add_issue(issue)
                result.security_issues.extend(security_objects)
            
            # Error detection
            if self.error_detector and (AnalysisType.SYNTAX in analysis_types or AnalysisType.STYLE in analysis_types):
                error_issues = await self.error_detector.detect_errors(code, language, "<string>")
                for issue in error_issues:
                    result.add_issue(issue)
            
            # Refactoring suggestions
            if AnalysisType.REFACTORING in analysis_types and self.refactoring_engine:
                refactoring_issues, refactoring_suggestions = await self.refactoring_engine.generate_refactoring_suggestions(code, language, "<string>")
                for issue in refactoring_issues:
                    result.add_issue(issue)
                result.refactoring_suggestions.extend(refactoring_suggestions)
            
        except Exception as e:
            result.add_issue(AnalysisIssue(
                analysis_type=AnalysisType.SYNTAX,
                severity=SeverityLevel.ERROR,
                message=f"Analysis failed: {str(e)}"
            ))
        
        result.analysis_duration_ms = int((time.time() - start_time) * 1000)
        return result
    
    def is_language_supported(self, language: Language) -> bool:
        """Check if language is supported for advanced analysis"""
        if not self.parser_manager:
            return language in [Language.PYTHON, Language.JAVASCRIPT]  # Basic support
        
        return self.parser_manager.is_language_supported(language)
    
    def get_supported_languages(self) -> List[Language]:
        """Get list of supported languages"""
        if not self.parser_manager:
            return [Language.PYTHON, Language.JAVASCRIPT]  # Basic support
        
        return self.parser_manager.get_supported_languages()
    
    def get_analysis_capabilities(self) -> Dict[str, Any]:
        """Get information about analysis capabilities"""
        return {
            'tree_sitter_available': self.tree_sitter_available,
            'supported_languages': [lang.value for lang in self.get_supported_languages()],
            'available_analysis_types': [analysis_type.value for analysis_type in AnalysisType],
            'features': {
                'syntax_analysis': True,
                'complexity_analysis': True,
                'security_analysis': SECURITY_ANALYZER_AVAILABLE,
                'error_detection': ERROR_DETECTOR_AVAILABLE,
                'performance_analysis': False,  # To be implemented
                'refactoring_suggestions': REFACTORING_ENGINE_AVAILABLE
            }
        }


# Global code analyzer instance
code_analyzer = CodeAnalyzer()
