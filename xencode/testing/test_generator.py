#!/usr/bin/env python3
"""
Test Generation Engine

Automated test generation system that analyzes code changes and generates
relevant tests for unittest, pytest, and doctest formats.

Features:
- Code analysis and test case generation based on function signatures
- Edge case identification and test generation
- Mock/stub generation for dependencies
- Support for multiple test frameworks (unittest, pytest, doctest)
- Integration with agentic workflow from Phase 1
"""

import ast
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from typing import get_args, get_origin, get_type_hints
except ImportError:
    pass


class TestFramework(Enum):
    """Supported test frameworks"""
    UNITTEST = "unittest"
    PYTEST = "pytest"
    DOCTEST = "doctest"
    MIXED = "mixed"


class TestType(Enum):
    """Types of tests that can be generated"""
    UNIT = "unit"
    INTEGRATION = "integration"
    EDGE_CASE = "edge_case"
    ERROR_HANDLING = "error_handling"
    BOUNDARY = "boundary"
    PROPERTY_BASED = "property_based"


class EdgeCaseType(Enum):
    """Types of edge cases to generate tests for"""
    EMPTY_INPUT = "empty_input"
    NONE_INPUT = "none_input"
    MAX_VALUE = "max_value"
    MIN_VALUE = "min_value"
    NEGATIVE_VALUE = "negative_value"
    ZERO_VALUE = "zero_value"
    LARGE_INPUT = "large_input"
    SPECIAL_CHARACTERS = "special_characters"
    UNICODE = "unicode"
    DUPLICATE_INPUT = "duplicate_input"
    SORTED_INPUT = "sorted_input"
    REVERSE_SORTED_INPUT = "reverse_sorted_input"


@dataclass
class FunctionSignature:
    """Represents a function signature extracted from code"""
    name: str
    args: List[Tuple[str, Optional[str]]]  # (arg_name, arg_type)
    return_type: Optional[str]
    docstring: Optional[str]
    decorators: List[str]
    is_async: bool
    is_method: bool
    class_name: Optional[str]
    line_number: int
    end_line_number: int
    source_code: str


@dataclass
class TestCase:
    """Represents a generated test case"""
    test_id: str
    name: str
    description: str
    test_type: TestType
    framework: TestFramework
    function_name: str
    input_data: Any
    expected_output: Any
    expected_exception: Optional[type] = None
    mocks: List[str] = field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""
    assertions: List[str] = field(default_factory=list)
    edge_case_type: Optional[EdgeCaseType] = None
    priority: int = 1  # 1 = highest priority
    tags: List[str] = field(default_factory=list)


@dataclass
class GeneratedTestFile:
    """Represents a generated test file"""
    file_id: str
    file_path: str
    framework: TestFramework
    test_cases: List[TestCase]
    imports: List[str]
    fixtures: List[str]
    generated_at: datetime
    source_file: str
    coverage_estimate: float = 0.0


@dataclass
class TestGenerationConfig:
    """Configuration for test generation"""
    frameworks: List[TestFramework] = field(default_factory=lambda: [TestFramework.PYTEST])
    test_types: List[TestType] = field(default_factory=lambda: [
        TestType.UNIT, TestType.EDGE_CASE, TestType.ERROR_HANDLING
    ])
    generate_mocks: bool = True
    generate_edge_cases: bool = True
    generate_property_tests: bool = False
    min_coverage_target: float = 80.0
    max_tests_per_function: int = 10
    include_docstring_tests: bool = True
    output_dir: str = "tests/generated"
    prefix: str = "test_"
    suffix: str = ".py"


class CodeAnalyzer:
    """Analyzes Python source code to extract function signatures and dependencies"""

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.tree = ast.parse(source_code)
        self.functions: List[FunctionSignature] = []
        self.classes: Dict[str, List[FunctionSignature]] = {}
        self.imports: List[str] = []
        self.dependencies: Set[str] = set()

    def analyze(self) -> 'CodeAnalyzer':
        """Perform code analysis"""
        self._extract_imports()
        self._extract_functions()
        self._extract_dependencies()
        return self

    def _extract_imports(self) -> None:
        """Extract import statements from the code"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    self.imports.append(f"from {module} import {alias.name}")

    def _extract_functions(self) -> None:
        """Extract function signatures from the code"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                signature = self._extract_function_signature(node)
                if signature:
                    self.functions.append(signature)

        # Extract class methods
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        signature = self._extract_function_signature(item, class_name=node.name)
                        if signature:
                            methods.append(signature)
                self.classes[node.name] = methods

    def _extract_function_signature(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        class_name: Optional[str] = None
    ) -> Optional[FunctionSignature]:
        """Extract function signature from AST node"""
        try:
            # Get function name
            name = node.name

            # Get arguments
            args = []
            for arg in node.args.args:
                arg_type = None
                if arg.annotation:
                    arg_type = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                args.append((arg.arg, arg_type))

            # Get return type
            return_type = None
            if node.returns:
                return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

            # Get docstring
            docstring = ast.get_docstring(node)

            # Get decorators
            decorators = []
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    decorators.append(decorator.id)
                elif isinstance(decorator, ast.Attribute):
                    decorators.append(ast.unparse(decorator) if hasattr(ast, 'unparse') else str(decorator))
                elif isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name):
                        decorators.append(decorator.func.id)

            # Check if async
            is_async = isinstance(node, ast.AsyncFunctionDef)

            # Check if method (has 'self' or 'cls' as first arg)
            is_method = class_name is not None or (args and args[0][0] in ('self', 'cls'))

            # Get source code for the function
            source_lines = self.source_code.split('\n')
            func_source = '\n'.join(source_lines[node.lineno - 1:node.end_lineno])

            return FunctionSignature(
                name=name,
                args=args,
                return_type=return_type,
                docstring=docstring,
                decorators=decorators,
                is_async=is_async,
                is_method=is_method,
                class_name=class_name,
                line_number=node.lineno,
                end_line_number=node.end_lineno,
                source_code=func_source
            )
        except Exception:
            return None

    def _extract_dependencies(self) -> None:
        """Extract external dependencies from the code"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.dependencies.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        self.dependencies.add(node.func.value.id)

    def get_functions(self) -> List[FunctionSignature]:
        """Get all extracted function signatures"""
        return self.functions

    def get_classes(self) -> Dict[str, List[FunctionSignature]]:
        """Get all classes with their methods"""
        return self.classes

    def get_imports(self) -> List[str]:
        """Get all import statements"""
        return self.imports

    def get_dependencies(self) -> Set[str]:
        """Get external dependencies"""
        return self.dependencies


class MockGenerator:
    """Generates mock objects for dependencies"""

    def __init__(self):
        self.mocks: Dict[str, str] = {}

    def generate_mock(self, dependency_name: str, return_value: Any = None) -> str:
        """Generate mock code for a dependency"""
        mock_name = f"mock_{dependency_name.lower()}"

        if return_value is None:
            return_value = self._get_default_return_value(dependency_name)

        mock_code = f"""
{mock_name} = MagicMock()
{mock_name}.return_value = {repr(return_value)}
"""
        self.mocks[dependency_name] = mock_code
        return mock_code

    def generate_patch_decorator(self, dependency_name: str, module_path: str) -> str:
        """Generate pytest patch decorator for a dependency"""
        return f'@patch("{module_path}.{dependency_name}")'

    def generate_context_manager_mock(self, name: str) -> str:
        """Generate a mock for a context manager"""
        return f"""
@contextmanager
def mock_{name.lower()}():
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    yield mock
"""

    def _get_default_return_value(self, name: str) -> Any:
        """Get a sensible default return value based on name"""
        name_lower = name.lower()

        if 'count' in name_lower or 'size' in name_lower or 'length' in name_lower:
            return 0
        elif 'list' in name_lower or 'items' in name_lower or 'array' in name_lower:
            return []
        elif 'dict' in name_lower or 'map' in name_lower or 'config' in name_lower:
            return {}
        elif 'str' in name_lower or 'name' in name_lower or 'message' in name_lower:
            return ""
        elif 'bool' in name_lower or 'is_' in name_lower or 'has_' in name_lower:
            return True
        elif 'float' in name_lower or 'rate' in name_lower or 'price' in name_lower:
            return 0.0
        else:
            return None

    def get_all_mocks(self) -> Dict[str, str]:
        """Get all generated mocks"""
        return self.mocks


class EdgeCaseGenerator:
    """Generates edge case test inputs"""

    def __init__(self):
        self.edge_cases: Dict[EdgeCaseType, Any] = {
            EdgeCaseType.EMPTY_INPUT: "",
            EdgeCaseType.NONE_INPUT: None,
            EdgeCaseType.MAX_VALUE: sys.maxsize,
            EdgeCaseType.MIN_VALUE: -sys.maxsize - 1,
            EdgeCaseType.NEGATIVE_VALUE: -1,
            EdgeCaseType.ZERO_VALUE: 0,
            EdgeCaseType.LARGE_INPUT: "a" * 10000,
            EdgeCaseType.SPECIAL_CHARACTERS: "!@#$%^&*()_+-=[]{}|;':\",./<>?",
            EdgeCaseType.UNICODE: "你好世界🌍",
            EdgeCaseType.DUPLICATE_INPUT: [1, 1, 2, 2, 3, 3],
            EdgeCaseType.SORTED_INPUT: [1, 2, 3, 4, 5],
            EdgeCaseType.REVERSE_SORTED_INPUT: [5, 4, 3, 2, 1],
        }

    def get_edge_cases_for_type(self, arg_type: Optional[str]) -> List[Tuple[EdgeCaseType, Any]]:
        """Get relevant edge cases for a given type"""
        if arg_type is None:
            return list(self.edge_cases.items())[:5]  # Return first 5 for unknown types

        arg_type_lower = arg_type.lower()
        relevant_cases = []

        # String types
        if 'str' in arg_type_lower or 'string' in arg_type_lower:
            relevant_cases.extend([
                (EdgeCaseType.EMPTY_INPUT, ""),
                (EdgeCaseType.NONE_INPUT, None),
                (EdgeCaseType.SPECIAL_CHARACTERS, self.edge_cases[EdgeCaseType.SPECIAL_CHARACTERS]),
                (EdgeCaseType.UNICODE, self.edge_cases[EdgeCaseType.UNICODE]),
                (EdgeCaseType.LARGE_INPUT, self.edge_cases[EdgeCaseType.LARGE_INPUT]),
            ])

        # Numeric types
        elif any(t in arg_type_lower for t in ['int', 'float', 'number']):
            relevant_cases.extend([
                (EdgeCaseType.ZERO_VALUE, 0),
                (EdgeCaseType.NEGATIVE_VALUE, -1),
                (EdgeCaseType.MAX_VALUE, sys.maxsize),
                (EdgeCaseType.MIN_VALUE, -sys.maxsize - 1),
            ])

        # List/sequence types
        elif any(t in arg_type_lower for t in ['list', 'array', 'sequence', 'tuple']):
            relevant_cases.extend([
                (EdgeCaseType.EMPTY_INPUT, []),
                (EdgeCaseType.NONE_INPUT, None),
                (EdgeCaseType.DUPLICATE_INPUT, self.edge_cases[EdgeCaseType.DUPLICATE_INPUT]),
                (EdgeCaseType.SORTED_INPUT, self.edge_cases[EdgeCaseType.SORTED_INPUT]),
                (EdgeCaseType.REVERSE_SORTED_INPUT, self.edge_cases[EdgeCaseType.REVERSE_SORTED_INPUT]),
            ])

        # Dict/mapping types
        elif any(t in arg_type_lower for t in ['dict', 'map', 'mapping']):
            relevant_cases.extend([
                (EdgeCaseType.EMPTY_INPUT, {}),
                (EdgeCaseType.NONE_INPUT, None),
            ])

        # Boolean types
        elif 'bool' in arg_type_lower:
            relevant_cases.extend([
                (EdgeCaseType.ZERO_VALUE, False),
                (EdgeCaseType.MAX_VALUE, True),
            ])

        # Default cases for unknown types
        if not relevant_cases:
            relevant_cases.extend([
                (EdgeCaseType.NONE_INPUT, None),
                (EdgeCaseType.EMPTY_INPUT, ""),
            ])

        return relevant_cases

    def generate_boundary_values(self, min_val: float = 0, max_val: float = 100) -> List[float]:
        """Generate boundary test values"""
        return [
            min_val - 1,  # Just below minimum
            min_val,  # At minimum
            min_val + 0.001,  # Just above minimum
            (min_val + max_val) / 2,  # Middle
            max_val - 0.001,  # Just below maximum
            max_val,  # At maximum
            max_val + 1,  # Just above maximum
        ]


class TestTemplateGenerator(ABC):
    """Abstract base class for test template generators"""

    @abstractmethod
    def generate_test_file(self, test_cases: List[TestCase], config: TestGenerationConfig) -> str:
        """Generate test file content"""
        pass

    @abstractmethod
    def generate_test_case(self, test_case: TestCase) -> str:
        """Generate individual test case code"""
        pass


class PytestTemplateGenerator(TestTemplateGenerator):
    """Generates pytest-style test files"""

    def generate_test_file(self, test_cases: List[TestCase], config: TestGenerationConfig) -> str:
        """Generate pytest-style test file"""
        imports = self._generate_imports(test_cases)
        fixtures = self._generate_fixtures(test_cases)
        test_functions = [self.generate_test_case(tc) for tc in test_cases]

        content = f'''#!/usr/bin/env python3
"""
Auto-generated test file
Generated at: {datetime.now().isoformat()}
Framework: pytest
"""

{imports}

{fixtures}

{chr(10).join(test_functions)}
'''
        return content

    def _generate_imports(self, test_cases: List[TestCase]) -> str:
        """Generate import statements"""
        imports = [
            "import pytest",
            "from unittest.mock import MagicMock, patch, Mock",
            "from contextlib import contextmanager",
        ]

        # Add imports based on test case requirements
        for tc in test_cases:
            for mock in tc.mocks:
                if mock not in str(imports):
                    imports.append(f"from unittest.mock import {mock}")

        # Remove duplicates while preserving order
        seen = set()
        unique_imports = []
        for imp in imports:
            if imp not in seen:
                seen.add(imp)
                unique_imports.append(imp)

        return "\n".join(unique_imports)

    def _generate_fixtures(self, test_cases: List[TestCase]) -> str:
        """Generate pytest fixtures"""
        fixtures = []

        # Check if any test case needs fixtures
        for tc in test_cases:
            if tc.setup_code:
                fixture_name = f"fixture_{tc.function_name}"
                fixture = f'''
@pytest.fixture
def {fixture_name}():
    """Fixture for {tc.function_name}"""
    {tc.setup_code}
    yield
    {tc.teardown_code}
'''
                fixtures.append(fixture)

        return "\n".join(fixtures)

    def generate_test_case(self, test_case: TestCase) -> str:
        """Generate pytest-style test function"""
        decorators = []

        # Add parametrize decorator if input_data is a list
        if isinstance(test_case.input_data, list) and len(test_case.input_data) > 1:
            decorators.append(f'@pytest.mark.parametrize("input_data,expected", {test_case.input_data})')

        # Add mark decorators based on test type
        if test_case.test_type == TestType.EDGE_CASE:
            decorators.append('@pytest.mark.edge_case')
        elif test_case.test_type == TestType.ERROR_HANDLING:
            decorators.append('@pytest.mark.error_handling')
        elif test_case.test_type == TestType.INTEGRATION:
            decorators.append('@pytest.mark.integration')

        # Add async marker if needed
        if test_case.framework == TestFramework.PYTEST and any('async' in m for m in test_case.mocks):
            decorators.append('@pytest.mark.asyncio')

        decorator_str = "\n".join(decorators)

        # Generate test function
        async_prefix = "async " if any('async' in str(tc.mocks) for tc in [test_case]) else ""

        # Generate assertions
        assertions = self._generate_assertions(test_case)

        # Generate mock setup
        mock_setup = self._generate_mock_setup(test_case)

        test_func = f'''
{decorator_str}
def {async_prefix}test_{test_case.name}(self):
    """
    {test_case.description}

    Test Type: {test_case.test_type.value}
    Function: {test_case.function_name}
    Priority: {test_case.priority}
    Tags: {", ".join(test_case.tags)}
    """
    {mock_setup}
    {test_case.setup_code}

    {assertions}

    {test_case.teardown_code}
'''
        return test_func

    def _generate_assertions(self, test_case: TestCase) -> str:
        """Generate assertion code"""
        assertions = []

        if test_case.expected_exception:
            assertions.append(f'with pytest.raises({test_case.expected_exception.__name__}):')
            assertions.append('    result = function_call()')
        else:
            assertions.append('result = function_call()')
            if test_case.expected_output is not None:
                assertions.append(f'assert result == {repr(test_case.expected_output)}')

        # Add custom assertions
        for assertion in test_case.assertions:
            assertions.append(assertion)

        return "\n    ".join(assertions)

    def _generate_mock_setup(self, test_case: TestCase) -> str:
        """Generate mock setup code"""
        mock_setup = []

        for mock_name in test_case.mocks:
            mock_setup.append(f'{mock_name} = MagicMock()')

        return "\n    ".join(mock_setup) if mock_setup else "# No mocks required"


class UnittestTemplateGenerator(TestTemplateGenerator):
    """Generates unittest-style test files"""

    def generate_test_file(self, test_cases: List[TestCase], config: TestGenerationConfig) -> str:
        """Generate unittest-style test file"""
        imports = self._generate_imports()
        test_class = self._generate_test_class(test_cases)

        content = f'''#!/usr/bin/env python3
"""
Auto-generated test file
Generated at: {datetime.now().isoformat()}
Framework: unittest
"""

{imports}


{test_class}


if __name__ == "__main__":
    unittest.main()
'''
        return content

    def _generate_imports(self) -> str:
        """Generate import statements"""
        return """import unittest
from unittest.mock import MagicMock, patch, Mock
from contextlib import contextmanager
"""

    def _generate_test_class(self, test_cases: List[TestCase]) -> str:
        """Generate test class"""
        if not test_cases:
            return ""

        # Get class name from first test case
        base_name = test_cases[0].function_name.replace('_', ' ').title().replace(' ', '')
        class_name = f"Test{base_name}"

        methods = [self.generate_test_case(tc) for tc in test_cases]

        test_class = f'''
class {class_name}(unittest.TestCase):
    """Auto-generated test class for {test_cases[0].function_name}"""

    def setUp(self):
        """Set up test fixtures"""
        pass

    def tearDown(self):
        """Tear down test fixtures"""
        pass

{chr(10).join(methods)}
'''
        return test_class

    def generate_test_case(self, test_case: TestCase) -> str:
        """Generate unittest-style test method"""
        # Generate assertions
        assertions = self._generate_assertions(test_case)

        # Generate mock setup
        mock_setup = self._generate_mock_setup(test_case)

        test_method = f'''
    def test_{test_case.name}(self):
        """
        {test_case.description}

        Test Type: {test_case.test_type.value}
        Function: {test_case.function_name}
        Priority: {test_case.priority}
        """
        {mock_setup}
        {test_case.setup_code}

        {assertions}

        {test_case.teardown_code}
'''
        return test_method

    def _generate_assertions(self, test_case: TestCase) -> str:
        """Generate assertion code"""
        assertions = []

        if test_case.expected_exception:
            assertions.append(f'self.assertRaises({test_case.expected_exception.__name__}, function_call)')
        else:
            assertions.append('result = function_call()')
            if test_case.expected_output is not None:
                assertions.append(f'self.assertEqual(result, {repr(test_case.expected_output)})')

        # Add custom assertions
        for assertion in test_case.assertions:
            assertions.append(assertion.replace('assert ', 'self.'))

        return "\n        ".join(assertions)

    def _generate_mock_setup(self, test_case: TestCase) -> str:
        """Generate mock setup code"""
        mock_setup = []

        for mock_name in test_case.mocks:
            mock_setup.append(f'{mock_name} = MagicMock()')

        return "\n        ".join(mock_setup) if mock_setup else "# No mocks required"


class DoctestTemplateGenerator(TestTemplateGenerator):
    """Generates doctest-style tests"""

    def generate_test_file(self, test_cases: List[TestCase], config: TestGenerationConfig) -> str:
        """Generate doctest-style test file"""
        docstring_tests = []

        for tc in test_cases:
            docstring_tests.append(self.generate_test_case(tc))

        content = f'''#!/usr/bin/env python3
"""
Auto-generated test file with doctests
Generated at: {datetime.now().isoformat()}
Framework: doctest

{chr(10).join(docstring_tests)}
"""


def run_tests():
    """Run all doctests"""
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    run_tests()
'''
        return content

    def generate_test_case(self, test_case: TestCase) -> str:
        """Generate doctest-style test"""
        # Format expected output
        expected = repr(test_case.expected_output) if test_case.expected_output else "None"

        doctest = f'''
>>> # Test: {test_case.name}
>>> # Description: {test_case.description}
>>> # Type: {test_case.test_type.value}
>>> result = function_call({test_case.input_data!r})
>>> result == {expected}
True
'''
        return doctest


class TestGenerator:
    """Main test generation engine"""

    def __init__(self, config: Optional[TestGenerationConfig] = None):
        self.config = config or TestGenerationConfig()
        self.mock_generator = MockGenerator()
        self.edge_case_generator = EdgeCaseGenerator()
        self.template_generators: Dict[TestFramework, TestTemplateGenerator] = {
            TestFramework.PYTEST: PytestTemplateGenerator(),
            TestFramework.UNITTEST: UnittestTemplateGenerator(),
            TestFramework.DOCTEST: DoctestTemplateGenerator(),
        }
        self.generated_files: List[GeneratedTestFile] = []

    def analyze_code(self, source_code: str) -> CodeAnalyzer:
        """Analyze source code"""
        analyzer = CodeAnalyzer(source_code)
        return analyzer.analyze()

    def generate_tests(
        self,
        source_code: str,
        source_file: str = "unknown.py",
        functions: Optional[List[str]] = None
    ) -> List[GeneratedTestFile]:
        """Generate tests for the given source code"""
        analyzer = self.analyze_code(source_code)
        test_files = []

        # Get functions to generate tests for
        all_functions = analyzer.get_functions()
        if functions:
            all_functions = [f for f in all_functions if f.name in functions]

        # Generate tests for each function
        for func in all_functions:
            test_cases = self._generate_test_cases_for_function(func, analyzer)

            # Group test cases by framework
            for framework in self.config.frameworks:
                framework_tests = [tc for tc in test_cases if tc.framework == framework]

                if framework_tests:
                    test_file = self._create_test_file(
                        framework_tests,
                        source_file,
                        func.name,
                        framework
                    )
                    test_files.append(test_file)
                    self.generated_files.append(test_file)

        return test_files

    def _generate_test_cases_for_function(
        self,
        func: FunctionSignature,
        analyzer: CodeAnalyzer
    ) -> List[TestCase]:
        """Generate test cases for a single function"""
        test_cases = []

        # Generate basic unit tests
        if TestType.UNIT in self.config.test_types:
            unit_tests = self._generate_unit_tests(func)
            test_cases.extend(unit_tests)

        # Generate edge case tests
        if self.config.generate_edge_cases and TestType.EDGE_CASE in self.config.test_types:
            edge_tests = self._generate_edge_case_tests(func)
            test_cases.extend(edge_tests)

        # Generate error handling tests
        if TestType.ERROR_HANDLING in self.config.test_types:
            error_tests = self._generate_error_handling_tests(func)
            test_cases.extend(error_tests)

        # Generate boundary tests
        if TestType.BOUNDARY in self.config.test_types:
            boundary_tests = self._generate_boundary_tests(func)
            test_cases.extend(boundary_tests)

        # Limit tests per function
        if len(test_cases) > self.config.max_tests_per_function:
            # Prioritize by priority value (lower is higher priority)
            test_cases.sort(key=lambda tc: tc.priority)
            test_cases = test_cases[:self.config.max_tests_per_function]

        return test_cases

    def _generate_unit_tests(self, func: FunctionSignature) -> List[TestCase]:
        """Generate basic unit tests"""
        test_cases = []

        # Generate happy path test
        input_data, expected_output = self._generate_typical_input(func)

        test_case = TestCase(
            test_id=str(uuid.uuid4()),
            name=f"{func.name}_happy_path",
            description=f"Test {func.name} with typical input",
            test_type=TestType.UNIT,
            framework=self.config.frameworks[0],
            function_name=func.name,
            input_data=input_data,
            expected_output=expected_output,
            mocks=list(self.mock_generator.get_all_mocks().keys()),
            priority=1,
            tags=["unit", "happy_path", func.name]
        )
        test_cases.append(test_case)

        return test_cases

    def _generate_edge_case_tests(self, func: FunctionSignature) -> List[TestCase]:
        """Generate edge case tests"""
        test_cases = []

        for arg_name, arg_type in func.args:
            if arg_name in ('self', 'cls'):
                continue

            edge_cases = self.edge_case_generator.get_edge_cases_for_type(arg_type)

            for edge_type, edge_value in edge_cases[:3]:  # Limit to 3 edge cases per arg
                test_case = TestCase(
                    test_id=str(uuid.uuid4()),
                    name=f"{func.name}_edge_{edge_type.value}",
                    description=f"Test {func.name} with {edge_type.value} for {arg_name}",
                    test_type=TestType.EDGE_CASE,
                    framework=self.config.frameworks[0],
                    function_name=func.name,
                    input_data={arg_name: edge_value},
                    expected_output=None,
                    edge_case_type=edge_type,
                    priority=2,
                    tags=["edge_case", edge_type.value, func.name]
                )
                test_cases.append(test_case)

        return test_cases

    def _generate_error_handling_tests(self, func: FunctionSignature) -> List[TestCase]:
        """Generate error handling tests"""
        test_cases = []

        # Test with None when not expected
        for arg_name, arg_type in func.args:
            if arg_name in ('self', 'cls'):
                continue

            # Check if arg accepts None
            if arg_type and 'Optional' not in arg_type and '?' not in arg_type:
                test_case = TestCase(
                    test_id=str(uuid.uuid4()),
                    name=f"{func.name}_error_none_{arg_name}",
                    description=f"Test {func.name} raises error when {arg_name} is None",
                    test_type=TestType.ERROR_HANDLING,
                    framework=self.config.frameworks[0],
                    function_name=func.name,
                    input_data={arg_name: None},
                    expected_output=None,
                    expected_exception=TypeError,
                    priority=2,
                    tags=["error_handling", "type_error", func.name]
                )
                test_cases.append(test_case)

        return test_cases

    def _generate_boundary_tests(self, func: FunctionSignature) -> List[TestCase]:
        """Generate boundary value tests"""
        test_cases = []

        for arg_name, arg_type in func.args:
            if arg_name in ('self', 'cls'):
                continue

            if arg_type and any(t in arg_type.lower() for t in ['int', 'float', 'number']):
                boundary_values = self.edge_case_generator.generate_boundary_values()

                for i, value in enumerate(boundary_values[:3]):
                    test_case = TestCase(
                        test_id=str(uuid.uuid4()),
                        name=f"{func.name}_boundary_{i}",
                        description=f"Test {func.name} with boundary value {value} for {arg_name}",
                        test_type=TestType.BOUNDARY,
                        framework=self.config.frameworks[0],
                        function_name=func.name,
                        input_data={arg_name: value},
                        expected_output=None,
                        priority=2,
                        tags=["boundary", func.name]
                    )
                    test_cases.append(test_case)

        return test_cases

    def _generate_typical_input(self, func: FunctionSignature) -> Tuple[Dict, Any]:
        """Generate typical input and expected output for a function"""
        input_data = {}

        for arg_name, arg_type in func.args:
            if arg_name in ('self', 'cls'):
                continue

            # Generate sensible default based on type
            if arg_type:
                arg_type_lower = arg_type.lower()
                if 'str' in arg_type_lower:
                    input_data[arg_name] = "test_value"
                elif 'int' in arg_type_lower:
                    input_data[arg_name] = 42
                elif 'float' in arg_type_lower:
                    input_data[arg_name] = 3.14
                elif 'bool' in arg_type_lower:
                    input_data[arg_name] = True
                elif 'list' in arg_type_lower:
                    input_data[arg_name] = [1, 2, 3]
                elif 'dict' in arg_type_lower:
                    input_data[arg_name] = {"key": "value"}
                else:
                    input_data[arg_name] = None
            else:
                input_data[arg_name] = None

        # Generate expected output based on return type
        expected_output = None
        if func.return_type:
            return_type_lower = func.return_type.lower()
            if 'str' in return_type_lower:
                expected_output = "result"
            elif 'int' in return_type_lower:
                expected_output = 0
            elif 'float' in return_type_lower:
                expected_output = 0.0
            elif 'bool' in return_type_lower:
                expected_output = True
            elif 'list' in return_type_lower:
                expected_output = []
            elif 'dict' in return_type_lower:
                expected_output = {}
            elif 'none' in return_type_lower or 'void' in return_type_lower:
                expected_output = None

        return input_data, expected_output

    def _create_test_file(
        self,
        test_cases: List[TestCase],
        source_file: str,
        function_name: str,
        framework: TestFramework
    ) -> GeneratedTestFile:
        """Create a generated test file"""
        template_generator = self.template_generators.get(framework)

        if not template_generator:
            raise ValueError(f"Unsupported framework: {framework}")

        # Generate file content
        template_generator.generate_test_file(test_cases, self.config)

        # Generate file path
        source_name = Path(source_file).stem
        file_path = f"{self.config.output_dir}/{self.config.prefix}{source_name}_{function_name}{self.config.suffix}"

        # Estimate coverage
        coverage_estimate = min(95.0, len(test_cases) * 10.0)

        return GeneratedTestFile(
            file_id=str(uuid.uuid4()),
            file_path=file_path,
            framework=framework,
            test_cases=test_cases,
            imports=self._get_required_imports(test_cases),
            fixtures=self._get_required_fixtures(test_cases),
            generated_at=datetime.now(),
            source_file=source_file,
            coverage_estimate=coverage_estimate
        )

    def _get_required_imports(self, test_cases: List[TestCase]) -> List[str]:
        """Get required imports for test cases"""
        imports = set()

        for tc in test_cases:
            for mock in tc.mocks:
                imports.add(f"from unittest.mock import {mock}")

        return list(imports)

    def _get_required_fixtures(self, test_cases: List[TestCase]) -> List[str]:
        """Get required fixtures for test cases"""
        fixtures = []

        for tc in test_cases:
            if tc.setup_code or tc.teardown_code:
                fixtures.append(f"fixture_{tc.function_name}")

        return fixtures

    def get_generated_files(self) -> List[GeneratedTestFile]:
        """Get all generated test files"""
        return self.generated_files

    def save_test_files(self, output_dir: Optional[str] = None) -> List[str]:
        """Save generated test files to disk"""
        output_dir = output_dir or self.config.output_dir
        saved_paths = []

        for test_file in self.generated_files:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Generate file path
            file_path = Path(output_dir) / Path(test_file.file_path).name

            # Get template generator
            template_generator = self.template_generators.get(test_file.framework)
            if template_generator:
                content = template_generator.generate_test_file(test_file.test_cases, self.config)

                # Write file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                saved_paths.append(str(file_path))

        return saved_paths


def create_test_generator(config: Optional[TestGenerationConfig] = None) -> TestGenerator:
    """Factory function to create a test generator"""
    return TestGenerator(config)


def generate_tests_for_code(
    source_code: str,
    source_file: str = "unknown.py",
    frameworks: Optional[List[TestFramework]] = None,
    output_dir: str = "tests/generated"
) -> List[GeneratedTestFile]:
    """Convenience function to generate tests for code"""
    config = TestGenerationConfig(
        frameworks=frameworks or [TestFramework.PYTEST],
        output_dir=output_dir
    )
    generator = TestGenerator(config)
    return generator.generate_tests(source_code, source_file)


# Integration with agentic workflow from Phase 1
class AgenticTestGenerator:
    """Test generator integrated with agentic workflow"""

    def __init__(self, test_generator: Optional[TestGenerator] = None):
        self.test_generator = test_generator or TestGenerator()
        self.workflow_context: Dict[str, Any] = {}

    def analyze_and_generate(
        self,
        code_changes: Dict[str, str],
        workflow_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze code changes and generate tests as part of agentic workflow"""
        self.workflow_context = workflow_context or {}

        results = {
            "workflow_id": self.workflow_context.get("workflow_id", str(uuid.uuid4())),
            "timestamp": datetime.now().isoformat(),
            "files_analyzed": [],
            "tests_generated": [],
            "coverage_estimate": 0.0,
            "status": "success"
        }

        total_coverage = 0.0

        for file_path, code in code_changes.items():
            try:
                # Analyze code
                analyzer = self.test_generator.analyze_code(code)
                functions = analyzer.get_functions()

                results["files_analyzed"].append({
                    "file_path": file_path,
                    "functions_found": len(functions),
                    "dependencies": list(analyzer.get_dependencies())
                })

                # Generate tests
                test_files = self.test_generator.generate_tests(code, file_path)

                for test_file in test_files:
                    results["tests_generated"].append({
                        "file_path": test_file.file_path,
                        "framework": test_file.framework.value,
                        "test_count": len(test_file.test_cases),
                        "coverage_estimate": test_file.coverage_estimate
                    })
                    total_coverage += test_file.coverage_estimate

            except Exception as e:
                results["files_analyzed"].append({
                    "file_path": file_path,
                    "error": str(e),
                    "status": "failed"
                })
                results["status"] = "partial_failure"

        # Calculate average coverage
        if results["tests_generated"]:
            results["coverage_estimate"] = total_coverage / len(results["tests_generated"])

        return results

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "workflow_id": self.workflow_context.get("workflow_id"),
            "generated_files_count": len(self.test_generator.get_generated_files()),
            "status": "active" if self.workflow_context else "idle"
        }


if __name__ == "__main__":
    # Example usage
    sample_code = '''
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def divide(a: float, b: float) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


class Calculator:
    """Simple calculator class"""

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    def subtract(self, a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b
'''

    # Generate tests
    generator = TestGenerator()
    test_files = generator.generate_tests(sample_code, "calculator.py")

    print(f"Generated {len(test_files)} test file(s)")
    for tf in test_files:
        print(f"  - {tf.file_path}: {len(tf.test_cases)} tests")
