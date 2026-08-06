#!/usr/bin/env python3
"""
Testing API Router

FastAPI router for test generation and execution endpoints.
Provides REST API access to the automated test generation system.

Endpoints:
- POST /tests/generate - Generate tests for code
- POST /tests/run - Execute tests
- GET /tests/results/{id} - Get test results
- GET /tests/coverage - Get coverage report
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from xencode.testing.failure_analyzer import (
    AnalysisResult,
    FailureAnalyzer,
    FixSuggestion,
    create_auto_retry_engine,
    create_failure_analyzer,
)
from xencode.testing.test_generator import (
    AgenticTestGenerator,
    GeneratedTestFile,
    TestFramework,
    TestGenerationConfig,
    TestGenerator,
    TestType,
    create_test_generator,
)
from xencode.testing.test_runner import (
    ExecutionConfig,
    ExecutionMode,
    TestExecutionResult,
    TestRunner,
    create_test_runner,
)

router = APIRouter()

# In-memory storage for results (use Redis/DB in production)
_generated_tests: Dict[str, GeneratedTestFile] = {}
_execution_results: Dict[str, TestExecutionResult] = {}
_analysis_results: Dict[str, AnalysisResult] = {}


# Pydantic models for API requests/responses
class TestGenerationRequest(BaseModel):
    """Request for test generation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str = Field(..., description="Source code to generate tests for")
    source_file: str = Field("unknown.py", description="Source file name")
    frameworks: List[str] = Field(
        default=["pytest"],
        description="Test frameworks to generate (pytest, unittest, doctest)"
    )
    test_types: List[str] = Field(
        default=["unit", "edge_case"],
        description="Types of tests to generate"
    )
    generate_mocks: bool = Field(True, description="Whether to generate mocks")
    generate_edge_cases: bool = Field(True, description="Whether to generate edge case tests")
    output_dir: str = Field("tests/generated", description="Output directory for tests")
    max_tests_per_function: int = Field(10, description="Maximum tests per function")
    functions: Optional[List[str]] = Field(None, description="Specific functions to test")


class TestGenerationResponse(BaseModel):
    """Response for test generation"""
    request_id: str
    status: str
    generated_files: List[Dict[str, Any]]
    total_tests: int
    coverage_estimate: float
    timestamp: datetime


class TestExecutionRequest(BaseModel):
    """Request for test execution"""
    test_files: List[str] = Field(..., description="List of test files to execute")
    parallel: bool = Field(True, description="Run tests in parallel")
    max_workers: int = Field(4, description="Maximum parallel workers")
    timeout_seconds: float = Field(300.0, description="Test timeout in seconds")
    collect_coverage: bool = Field(True, description="Collect coverage data")
    retry_failed: bool = Field(True, description="Retry failed tests")
    max_retries: int = Field(2, description="Maximum retry attempts")
    source_files: Optional[List[str]] = Field(None, description="Source files for coverage")


class TestExecutionResponse(BaseModel):
    """Response for test execution"""
    execution_id: str
    status: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_ms: float
    coverage_percent: Optional[float]
    timestamp: datetime


class TestResultsResponse(BaseModel):
    """Response for test results"""
    execution_id: str
    status: str
    summary: Dict[str, Any]
    test_results: List[Dict[str, Any]]
    coverage_report: Optional[Dict[str, Any]]
    timestamp: datetime


class CoverageReportResponse(BaseModel):
    """Response for coverage report"""
    total_lines: int
    covered_lines: int
    percent_covered: float
    missing_lines_count: int
    files: Dict[str, Dict[str, Any]]
    timestamp: datetime


class FailureAnalysisRequest(BaseModel):
    """Request for failure analysis"""
    execution_id: str = Field(..., description="Execution ID to analyze")
    test_codes: Optional[Dict[str, str]] = Field(None, description="Test code for suggestions")


class FailureAnalysisResponse(BaseModel):
    """Response for failure analysis"""
    analysis_id: str
    total_failures: int
    failure_types: Dict[str, int]
    suggestions: List[Dict[str, Any]]
    patterns_detected: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timestamp: datetime


class AutoRetryRequest(BaseModel):
    """Request for auto-retry with fixes"""
    execution_id: str = Field(..., description="Execution ID to retry")
    max_retries: int = Field(3, description="Maximum retry attempts")


class AutoRetryResponse(BaseModel):
    """Response for auto-retry"""
    success: bool
    retries_attempted: int
    fixes_applied: int
    final_result: Dict[str, Any]
    history: List[Dict[str, Any]]


# Dependency injection
def get_test_generator() -> TestGenerator:
    """Get test generator instance"""
    return create_test_generator()


def get_test_runner() -> TestRunner:
    """Get test runner instance"""
    return create_test_runner()


def get_failure_analyzer() -> FailureAnalyzer:
    """Get failure analyzer instance"""
    return create_failure_analyzer()


# API Endpoints
@router.post("/generate", response_model=TestGenerationResponse)
async def generate_tests(
    request: TestGenerationRequest,
    background_tasks: BackgroundTasks,
    generator: TestGenerator = Depends(get_test_generator)
):
    """
    Generate tests for source code

    - **code**: Source code to analyze and generate tests for
    - **source_file**: Name of the source file
    - **frameworks**: Test frameworks to generate (pytest, unittest, doctest)
    - **test_types**: Types of tests to generate (unit, integration, edge_case, etc.)
    - **generate_mocks**: Whether to generate mock objects
    - **generate_edge_cases**: Whether to generate edge case tests
    - **output_dir**: Directory to save generated tests
    - **max_tests_per_function**: Maximum number of tests per function
    - **functions**: Optional list of specific functions to test

    Returns generated test files with coverage estimates.
    """
    try:
        request_id = str(uuid.uuid4())

        # Convert string enums to TestFramework and TestType
        framework_map = {
            "pytest": TestFramework.PYTEST,
            "unittest": TestFramework.UNITTEST,
            "doctest": TestFramework.DOCTEST,
            "mixed": TestFramework.MIXED
        }
        test_type_map = {
            "unit": TestType.UNIT,
            "integration": TestType.INTEGRATION,
            "edge_case": TestType.EDGE_CASE,
            "error_handling": TestType.ERROR_HANDLING,
            "boundary": TestType.BOUNDARY
        }

        frameworks = [framework_map.get(fw, TestFramework.PYTEST) for fw in request.frameworks]
        test_types = [test_type_map.get(tt, TestType.UNIT) for tt in request.test_types]

        # Create configuration
        config = TestGenerationConfig(
            frameworks=frameworks,
            test_types=test_types,
            generate_mocks=request.generate_mocks,
            generate_edge_cases=request.generate_edge_cases,
            output_dir=request.output_dir,
            max_tests_per_function=request.max_tests_per_function
        )

        # Create generator with config
        generator = TestGenerator(config)

        # Generate tests
        test_files = generator.generate_tests(
            source_code=request.code,
            source_file=request.source_file,
            functions=request.functions
        )

        # Store generated files
        for tf in test_files:
            _generated_tests[tf.file_id] = tf

        # Save test files to disk
        saved_paths = generator.save_test_files()

        # Calculate totals
        total_tests = sum(len(tf.test_cases) for tf in test_files)
        avg_coverage = (
            sum(tf.coverage_estimate for tf in test_files) / len(test_files)
            if test_files else 0.0
        )

        return TestGenerationResponse(
            request_id=request_id,
            status="success",
            generated_files=[
                {
                    "file_id": tf.file_id,
                    "file_path": tf.file_path,
                    "framework": tf.framework.value,
                    "test_count": len(tf.test_cases),
                    "coverage_estimate": tf.coverage_estimate,
                    "saved_path": path
                }
                for tf, path in zip(test_files, saved_paths)
            ],
            total_tests=total_tests,
            coverage_estimate=avg_coverage,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate tests: {str(e)}"
        )  from e


@router.post("/run", response_model=TestExecutionResponse)
async def run_tests_endpoint(
    request: TestExecutionRequest,
    background_tasks: BackgroundTasks,
    runner: TestRunner = Depends(get_test_runner)
):
    """
    Execute tests

    - **test_files**: List of test file paths to execute
    - **parallel**: Run tests in parallel
    - **max_workers**: Maximum number of parallel workers
    - **timeout_seconds**: Test execution timeout
    - **collect_coverage**: Collect code coverage data
    - **retry_failed**: Retry failed tests automatically
    - **max_retries**: Maximum retry attempts per test
    - **source_files**: Source files for coverage calculation

    Returns execution results with pass/fail counts and coverage.
    """
    try:
        # Create execution config
        config = ExecutionConfig(
            mode=ExecutionMode.PARALLEL if request.parallel else ExecutionMode.SEQUENTIAL,
            max_workers=request.max_workers,
            timeout_seconds=request.timeout_seconds,
            collect_coverage=request.collect_coverage,
            retry_failed=request.retry_failed,
            max_retries=request.max_retries,
            coverage_source=request.source_files
        )

        # Create runner with config
        runner = TestRunner(config)

        # Execute tests
        execution_result = runner.execute(request.test_files)

        # Store result
        _execution_results[execution_result.execution_id] = execution_result

        # Get coverage percent
        coverage_percent = None
        if execution_result.coverage_report:
            coverage_percent = execution_result.coverage_report.get("percent_covered")

        return TestExecutionResponse(
            execution_id=execution_result.execution_id,
            status="completed",
            total_tests=execution_result.total_tests,
            passed=execution_result.passed,
            failed=execution_result.failed,
            skipped=execution_result.skipped,
            errors=execution_result.errors,
            duration_ms=execution_result.duration_ms,
            coverage_percent=coverage_percent,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute tests: {str(e)}"
        )  from e


@router.get("/results/{execution_id}", response_model=TestResultsResponse)
async def get_test_results(execution_id: str):
    """
    Get detailed test results

    - **execution_id**: ID of the test execution

    Returns detailed results for each test including errors and coverage.
    """
    if execution_id not in _execution_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution result not found: {execution_id}"
        )

    execution_result = _execution_results[execution_id]

    # Convert test results to dict
    test_results = []
    for tr in execution_result.test_results:
        test_results.append({
            "test_id": tr.test_id,
            "test_name": tr.test_name,
            "test_file": tr.test_file,
            "status": tr.status.value,
            "duration_ms": tr.duration_ms,
            "error_message": tr.error_message,
            "output": tr.output,
            "retry_count": tr.retry_count,
            "timestamp": tr.timestamp.isoformat()
        })

    return TestResultsResponse(
        execution_id=execution_id,
        status="completed",
        summary={
            "total": execution_result.total_tests,
            "passed": execution_result.passed,
            "failed": execution_result.failed,
            "skipped": execution_result.skipped,
            "errors": execution_result.errors,
            "success_rate": (
                execution_result.passed / execution_result.total_tests * 100
                if execution_result.total_tests > 0 else 0
            )
        },
        test_results=test_results,
        coverage_report=execution_result.coverage_report,
        timestamp=execution_result.end_time or execution_result.start_time
    )


@router.get("/coverage", response_model=CoverageReportResponse)
async def get_coverage_report(
    execution_id: Optional[str] = None
):
    """
    Get coverage report

    - **execution_id**: Optional execution ID to get coverage for

    Returns code coverage statistics.
    """
    if execution_id:
        if execution_id not in _execution_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution result not found: {execution_id}"
            )
        execution_result = _execution_results[execution_id]
        coverage_report = execution_result.coverage_report
    else:
        # Return latest coverage if no execution_id
        if _execution_results:
            latest = list(_execution_results.values())[-1]
            coverage_report = latest.coverage_report
        else:
            coverage_report = None

    if not coverage_report:
        return CoverageReportResponse(
            total_lines=0,
            covered_lines=0,
            percent_covered=0.0,
            missing_lines_count=0,
            files={},
            timestamp=datetime.now()
        )

    return CoverageReportResponse(
        total_lines=coverage_report.get("total_lines", 0),
        covered_lines=coverage_report.get("covered_lines", 0),
        percent_covered=coverage_report.get("percent_covered", 0.0),
        missing_lines_count=len(coverage_report.get("missing_lines", [])),
        files=coverage_report.get("files", {}),
        timestamp=datetime.now()
    )


@router.post("/analyze", response_model=FailureAnalysisResponse)
async def analyze_failures(
    request: FailureAnalysisRequest,
    analyzer: FailureAnalyzer = Depends(get_failure_analyzer)
):
    """
    Analyze test failures

    - **execution_id**: ID of the test execution to analyze
    - **test_codes**: Optional test code for fix suggestions

    Returns failure analysis with fix suggestions.
    """
    if request.execution_id not in _execution_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution result not found: {request.execution_id}"
        )

    execution_result = _execution_results[request.execution_id]

    # Analyze failures
    analysis_result = analyzer.analyze(execution_result, request.test_codes)

    # Store analysis
    _analysis_results[analysis_result.analysis_id] = analysis_result

    # Convert suggestions to dict
    suggestions = []
    for s in analysis_result.suggestions:
        suggestions.append({
            "suggestion_id": s.suggestion_id,
            "strategy": s.strategy.value,
            "description": s.description,
            "confidence": s.confidence,
            "auto_applicable": s.auto_applicable,
            "requires_review": s.requires_review,
            "explanation": s.explanation
        })

    # Convert patterns to dict
    patterns = []
    for p in analysis_result.patterns_detected:
        patterns.append({
            "pattern_id": p.pattern_id,
            "pattern_name": p.pattern_name,
            "description": p.description,
            "occurrence_count": p.occurrence_count
        })

    return FailureAnalysisResponse(
        analysis_id=analysis_result.analysis_id,
        total_failures=len(analysis_result.failures),
        failure_types=analysis_result.summary.get("failure_types", {}),
        suggestions=suggestions,
        patterns_detected=patterns,
        summary=analysis_result.summary,
        timestamp=datetime.now()
    )


@router.post("/retry", response_model=AutoRetryResponse)
async def auto_retry_tests(
    request: AutoRetryRequest,
    background_tasks: BackgroundTasks
):
    """
    Auto-retry failed tests with fixes

    - **execution_id**: ID of the test execution to retry
    - **max_retries**: Maximum retry attempts

    Returns retry results with fixes applied.
    """
    if request.execution_id not in _execution_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution result not found: {request.execution_id}"
        )

    execution_result = _execution_results[request.execution_id]

    # Create auto-retry engine
    retry_engine = create_auto_retry_engine(max_retries=request.max_retries)

    # Define fix application function (placeholder - would integrate with code modification)
    def apply_fix_fn(suggestion: FixSuggestion) -> bool:
        # In production, this would apply the fix to the code
        # For now, just return True for auto-applicable suggestions
        return suggestion.auto_applicable

    # Define test runner function
    def run_tests_fn() -> TestExecutionResult:
        # Re-run the same tests
        runner = TestRunner(ExecutionConfig(
            max_workers=2,
            retry_failed=False
        ))
        test_files = [tr.test_file for tr in execution_result.test_results]
        return runner.execute(test_files)

    # Execute retry
    retry_result = retry_engine.retry_with_fixes(
        execution_result=execution_result,
        apply_fix_fn=apply_fix_fn,
        run_tests_fn=run_tests_fn
    )

    return AutoRetryResponse(
        success=retry_result["success"],
        retries_attempted=retry_result["retries_attempted"],
        fixes_applied=retry_result["fixes_applied"],
        final_result=retry_result["final_result"],
        history=retry_result["history"]
    )


@router.get("/generated")
async def list_generated_tests():
    """
    List all generated test files

    Returns metadata about all generated test files.
    """
    tests = []
    for _file_id, tf in _generated_tests.items():
        tests.append({
            "file_id": tf.file_id,
            "file_path": tf.file_path,
            "framework": tf.framework.value,
            "test_count": len(tf.test_cases),
            "source_file": tf.source_file,
            "generated_at": tf.generated_at.isoformat(),
            "coverage_estimate": tf.coverage_estimate
        })

    return {
        "total": len(tests),
        "tests": tests
    }


@router.get("/generated/{file_id}")
async def get_generated_test(file_id: str):
    """
    Get details of a generated test file

    - **file_id**: ID of the generated test file
    """
    if file_id not in _generated_tests:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Generated test file not found: {file_id}"
        )

    tf = _generated_tests[file_id]

    # Convert test cases to dict
    test_cases = []
    for tc in tf.test_cases:
        test_cases.append({
            "test_id": tc.test_id,
            "name": tc.name,
            "description": tc.description,
            "test_type": tc.test_type.value,
            "framework": tc.framework.value,
            "function_name": tc.function_name,
            "priority": tc.priority,
            "tags": tc.tags
        })

    return {
        "file_id": tf.file_id,
        "file_path": tf.file_path,
        "framework": tf.framework.value,
        "test_cases": test_cases,
        "imports": tf.imports,
        "fixtures": tf.fixtures,
        "generated_at": tf.generated_at.isoformat(),
        "source_file": tf.source_file,
        "coverage_estimate": tf.coverage_estimate
    }


@router.get("/status")
async def get_testing_status():
    """
    Get testing system status

    Returns current state of the testing system.
    """
    return {
        "status": "healthy",
        "generated_tests_count": len(_generated_tests),
        "execution_results_count": len(_execution_results),
        "analysis_results_count": len(_analysis_results),
        "timestamp": datetime.now().isoformat()
    }


@router.delete("/results/{execution_id}")
async def delete_test_results(execution_id: str):
    """
    Delete test results

    - **execution_id**: ID of the test execution to delete
    """
    if execution_id not in _execution_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution result not found: {execution_id}"
        )

    del _execution_results[execution_id]

    return {
        "success": True,
        "message": f"Deleted results for execution: {execution_id}"
    }


@router.delete("/generated/{file_id}")
async def delete_generated_test(file_id: str):
    """
    Delete generated test file record

    - **file_id**: ID of the generated test file
    """
    if file_id not in _generated_tests:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Generated test file not found: {file_id}"
        )

    del _generated_tests[file_id]

    return {
        "success": True,
        "message": f"Deleted generated test: {file_id}"
    }


# Agentic workflow integration endpoint
@router.post("/agentic/generate")
async def agentic_test_generation(
    code_changes: Dict[str, str] = Body(..., description="Dictionary of file paths to code"),
    workflow_context: Optional[Dict[str, Any]] = Body(None, description="Workflow context")
):
    """
    Generate tests as part of agentic workflow

    Integrates with Phase 1 agentic workflow system.

    - **code_changes**: Dictionary mapping file paths to code content
    - **workflow_context**: Optional workflow context from agentic system

    Returns generation results integrated with workflow.
    """
    try:
        # Create agentic test generator
        agentic_generator = AgenticTestGenerator()

        # Generate tests for all changed files
        result = agentic_generator.analyze_and_generate(
            code_changes=code_changes,
            workflow_context=workflow_context or {}
        )

        # Store generated files
        for tf in agentic_generator.test_generator.get_generated_files():
            _generated_tests[tf.file_id] = tf

        return {
            "success": result["status"] == "success",
            "workflow_id": result["workflow_id"],
            "files_analyzed": result["files_analyzed"],
            "tests_generated": result["tests_generated"],
            "coverage_estimate": result["coverage_estimate"],
            "timestamp": result["timestamp"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agentic test generation failed: {str(e)}"
        )  from e


router.tags = ["Testing"]
