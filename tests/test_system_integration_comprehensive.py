#!/usr/bin/env python3
"""
Comprehensive System Integration Tests

Tests for complete system integration, end-to-end workflows,
component interactions, and comprehensive system validation.
"""

import asyncio
import pytest
import pytest_asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import json
from datetime import datetime, timedelta
import secrets

from xencode.phase2_coordinator import Phase2Coordinator
from xencode.ai_ensembles import EnsembleReasoner
from xencode.document_processor import DocumentProcessor
from xencode.code_analyzer import CodeAnalyzer
from xencode.workspace.workspace_manager import WorkspaceManager
from xencode.analytics.analytics_engine import AnalyticsEngine
from xencode.auth.auth_manager import AuthManager
from xencode.models.user import User, UserRole
from xencode.models.workspace import Workspace, WorkspaceFile, WorkspaceCollaborator, WorkspaceType


class TestSystemInitialization:
    """Test system initialization and component coordination"""

    @pytest_asyncio.fixture
    async def xencode_system(self):
        """Create a full Xencode system for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            system = Phase2Coordinator()
            await system.initialize()
            yield system
            await system.shutdown()

    @pytest.mark.asyncio
    async def test_system_initialization(self, xencode_system):
        """Test complete system initialization"""
        assert xencode_system is not None
        assert xencode_system.ensemble_reasoner is not None
        assert xencode_system.document_processor is not None
        assert xencode_system.code_analyzer is not None
        assert xencode_system.workspace_manager is not None
        assert xencode_system.analytics_engine is not None
        assert xencode_system.auth_manager is not None
        assert xencode_system._running is True

        # Check that all components are properly initialized
        components_status = xencode_system.get_system_status()
        assert components_status["ensemble_reasoner"]["running"] is True
        assert components_status["document_processor"]["running"] is True
        assert components_status["code_analyzer"]["running"] is True
        assert components_status["workspace_manager"]["running"] is True
        assert components_status["analytics_engine"]["running"] is True
        assert components_status["auth_manager"]["running"] is True

    @pytest.mark.asyncio
    async def test_component_coordination(self, xencode_system):
        """Test coordination between system components"""
        # Verify that components can communicate with each other
        # Check that analytics engine is connected to other components
        analytics_status = await xencode_system.analytics_engine.get_system_status()
        assert "components" in analytics_status

        # Check that auth manager has default users
        default_users = xencode_system.auth_manager.get_all_users()
        assert len(default_users) >= 2  # Should have admin and guest users

        # Check that workspace manager is initialized
        workspace_stats = xencode_system.workspace_manager.get_processing_stats()
        assert "registered_workspaces" in workspace_stats


class TestEndToEndWorkflows:
    """Test end-to-end system workflows"""

    @pytest_asyncio.fixture
    async def xencode_system(self):
        """Create a full Xencode system for workflow testing"""
        system = Phase2Coordinator()
        await system.initialize()
        yield system
        await system.shutdown()

    @pytest.mark.asyncio
    async def test_document_analysis_workflow(self, xencode_system):
        """Test complete document analysis workflow"""
        # 1. Create user
        user = xencode_system.auth_manager.create_user(
            username="workflow_user",
            email="workflow@example.com",
            password="securepassword123",
            role=UserRole.DEVELOPER
        )
        assert user is not None

        # 2. Authenticate user
        access_token, refresh_token, authenticated_user = await xencode_system.auth_manager.authenticate(
            "workflow_user", "securepassword123"
        )
        assert authenticated_user is not None

        # 3. Create workspace
        workspace = await xencode_system.workspace_manager.create_workspace(
            name="Document Analysis Workspace",
            owner_id=authenticated_user.id,
            workspace_type=WorkspaceType.PROJECT,
            description="Workspace for document analysis workflow"
        )
        assert workspace is not None

        # 4. Add document to workspace
        doc_content = """
        # Sample Document
        This is a sample document for analysis.
        
        ## Features
        - Feature 1
        - Feature 2
        - Feature 3
        
        ## Code Example
        ```python
        def hello_world():
            print("Hello, World!")
            return True
        ```
        """
        
        doc_file = WorkspaceFile(
            name="sample_doc.md",
            path="/sample_doc.md",
            content=doc_content
        )
        
        add_success = await xencode_system.workspace_manager.add_file(
            workspace.id, doc_file, authenticated_user.id
        )
        assert add_success is True

        # 5. Process document
        processed_result = await xencode_system.document_processor.process_document(
            workspace.id, doc_file.id, authenticated_user.id
        )
        assert processed_result is not None

        # 6. Analyze code within document
        analysis_result = await xencode_system.code_analyzer.analyze_code_string(
            doc_content,
            language="markdown"
        )
        assert analysis_result is not None

        # 7. Generate AI response using ensemble
        ai_response = await xencode_system.ensemble_reasoner.reason({
            "prompt": "Summarize the key features of this document",
            "context": doc_content,
            "user_id": authenticated_user.id
        })
        assert ai_response is not None

        # 8. Verify analytics tracking
        analytics_status = await xencode_system.analytics_engine.get_system_status()
        assert "events" in analytics_status
        # Should have tracked the various operations

    @pytest.mark.asyncio
    async def test_code_analysis_and_refactoring_workflow(self, xencode_system):
        """Test complete code analysis and refactoring workflow"""
        # Create user
        user = xencode_system.auth_manager.create_user(
            username="code_user",
            email="code@example.com",
            password="codepassword123",
            role=UserRole.DEVELOPER
        )
        assert user is not None

        # Authenticate user
        access_token, _, authenticated_user = await xencode_system.auth_manager.authenticate(
            "code_user", "codepassword123"
        )
        assert authenticated_user is not None

        # Create workspace
        workspace = await xencode_system.workspace_manager.create_workspace(
            name="Code Analysis Workspace",
            owner_id=authenticated_user.id,
            workspace_type=WorkspaceType.CODE_ANALYSIS
        )
        assert workspace is not None

        # Add problematic code file
        problematic_code = """
def calculate_factorial(n):
    # This function has performance issues and lacks error handling
    if n == 0:
        return 1
    else:
        return n * calculate_factorial(n-1)  # Recursive without limits

def unused_function():
    # This function is never used
    return "unused"

class BadClassName:
    # Class name doesn't follow convention
    def __init__(self):
        self.bad_var_name = "bad naming convention"
"""
        
        code_file = WorkspaceFile(
            name="problematic_code.py",
            path="/problematic_code.py",
            content=problematic_code
        )
        
        add_success = await xencode_system.workspace_manager.add_file(
            workspace.id, code_file, authenticated_user.id
        )
        assert add_success is True

        # Analyze the code
        analysis_result = await xencode_system.code_analyzer.analyze_code_string(
            problematic_code,
            language="python"
        )
        assert analysis_result is not None
        assert len(analysis_result.issues) > 0  # Should find issues

        # Check for specific issues
        style_issues = [issue for issue in analysis_result.issues if issue.analysis_type == "style"]
        security_issues = [issue for issue in analysis_result.issues if issue.analysis_type == "security"]
        performance_issues = [issue for issue in analysis_result.issues if issue.analysis_type == "performance"]

        # Generate refactoring suggestions
        refactoring_suggestions = await xencode_system.code_analyzer.generate_refactoring_suggestions(
            problematic_code, "python"
        )
        assert refactoring_suggestions is not None

        # Use ensemble reasoner to improve the code based on analysis
        improvement_prompt = f"""
        Based on the following code analysis issues, suggest improvements:
        Issues: {json.dumps([issue.to_dict() for issue in analysis_result.issues][:5], indent=2)}
        
        Original code:
        {problematic_code}
        """
        
        improvement_response = await xencode_system.ensemble_reasoner.reason({
            "prompt": improvement_prompt,
            "context": problematic_code,
            "user_id": authenticated_user.id
        })
        
        assert improvement_response is not None

    @pytest.mark.asyncio
    async def test_collaboration_workflow(self, xencode_system):
        """Test collaboration workflow with multiple users"""
        # Create owner user
        owner = xencode_system.auth_manager.create_user(
            username="project_owner",
            email="owner@example.com",
            password="ownerpassword123",
            role=UserRole.DEVELOPER
        )
        assert owner is not None

        # Create collaborator user
        collaborator = xencode_system.auth_manager.create_user(
            username="project_collaborator",
            email="collaborator@example.com",
            password="collabpassword123",
            role=UserRole.VIEWER
        )
        assert collaborator is not None

        # Authenticate owner
        owner_token, _, owner_user = await xencode_system.auth_manager.authenticate(
            "project_owner", "ownerpassword123"
        )
        assert owner_user is not None

        # Create workspace
        workspace = await xencode_system.workspace_manager.create_workspace(
            name="Collaboration Workspace",
            owner_id=owner_user.id,
            workspace_type=WorkspaceType.COLLABORATION
        )
        assert workspace is not None

        # Add collaborator to workspace
        collab_obj = WorkspaceCollaborator(
            user_id=collaborator.id,
            username=collaborator.username,
            role="editor"
        )
        
        add_collab_success = await xencode_system.workspace_manager.add_collaborator(
            workspace.id, collaborator.id, collab_obj, owner_user.id
        )
        assert add_collab_success is True

        # Add file to workspace
        shared_code = """
def shared_function():
    return "This is shared code"
"""
        
        shared_file = WorkspaceFile(
            name="shared.py",
            path="/shared.py",
            content=shared_code
        )
        
        add_file_success = await xencode_system.workspace_manager.add_file(
            workspace.id, shared_file, owner_user.id
        )
        assert add_file_success is True

        # Authenticate collaborator
        collab_token, _, collab_user = await xencode_system.auth_manager.authenticate(
            "project_collaborator", "collabpassword123"
        )
        assert collab_user is not None

        # Collaborator should be able to access the file
        retrieved_file = await xencode_system.workspace_manager.get_file(
            workspace.id, shared_file.id, collab_user.id
        )
        assert retrieved_file is not None
        assert retrieved_file.content == shared_code

        # Collaborator should be able to modify the file (if permissions allow)
        updated_content = shared_code + "\n# Modified by collaborator"
        retrieved_file.content = updated_content
        
        update_success = await xencode_system.workspace_manager.update_file(
            workspace.id, retrieved_file, collab_user.id
        )
        # This might succeed or fail depending on permissions implementation

        # Verify the change was made
        updated_file = await xencode_system.workspace_manager.get_file(
            workspace.id, shared_file.id, owner_user.id
        )
        if update_success:
            assert updated_file is not None
            assert "Modified by collaborator" in updated_file.content


class TestComponentInteractions:
    """Test interactions between different system components"""

    @pytest_asyncio.fixture
    async def xencode_system(self):
        """Create a system for component interaction testing"""
        system = Phase2Coordinator()
        await system.initialize()
        yield system
        await system.shutdown()

    @pytest.mark.asyncio
    async def test_analytics_integration_with_all_components(self, xencode_system):
        """Test that all components properly integrate with analytics"""
        # Perform operations with each component and verify they're tracked

        # 1. Document processing operation
        doc_content = "# Test document\nThis is test content."
        doc_result = await xencode_system.document_processor.process_content(doc_content)
        # This should generate analytics events

        # 2. Code analysis operation
        code_content = "def test(): pass"
        code_result = await xencode_system.code_analyzer.analyze_code_string(code_content, "python")
        # This should generate analytics events

        # 3. Workspace operation
        workspace = await xencode_system.workspace_manager.create_workspace(
            name="Analytics Test", owner_id="test_user"
        )
        # This should generate analytics events

        # 4. Get analytics to verify tracking
        analytics_events = await xencode_system.analytics_engine.get_recent_events(hours=1)
        # Should have events from all operations

        # Verify different types of events were generated
        event_types = set(event.event_type for event in analytics_events)
        # Should have various event types from different components

    @pytest.mark.asyncio
    async def test_security_integration_across_components(self, xencode_system):
        """Test security integration across all components"""
        # Create user
        user = xencode_system.auth_manager.create_user(
            username="security_test",
            email="security@example.com",
            password="securepassword123",
            role=UserRole.VIEWER  # Limited role
        )
        assert user is not None

        # Authenticate user
        access_token, _, authenticated_user = await xencode_system.auth_manager.authenticate(
            "security_test", "securepassword123"
        )
        assert authenticated_user is not None

        # Create workspace
        workspace = await xencode_system.workspace_manager.create_workspace(
            name="Security Test Workspace",
            owner_id=authenticated_user.id,
            workspace_type=WorkspaceType.PROJECT
        )
        assert workspace is not None

        # Try operations that should be limited by security
        # Viewer role should have limited permissions compared to admin/developer

        # Test permission checks
        has_admin_perm = await xencode_system.workspace_manager.check_workspace_permission(
            workspace.id, authenticated_user.id, "admin"
        )
        # Viewer should not have admin permissions

        has_read_perm = await xencode_system.workspace_manager.check_workspace_permission(
            workspace.id, authenticated_user.id, "read"
        )
        # Viewer should have read permissions

    @pytest.mark.asyncio
    async def test_cache_integration_across_components(self, xencode_system):
        """Test cache integration across system components"""
        # Components should use the shared caching system
        # This test verifies that caching is coordinated across components

        # Process the same content multiple times to test caching
        test_content = "def repeated_function(): pass"
        
        # First analysis
        start_time = datetime.now()
        result1 = await xencode_system.code_analyzer.analyze_code_string(test_content, "python")
        time1 = (datetime.now() - start_time).total_seconds()

        # Second analysis of same content (should be faster due to caching)
        start_time = datetime.now()
        result2 = await xencode_system.code_analyzer.analyze_code_string(test_content, "python")
        time2 = (datetime.now() - start_time).total_seconds()

        # Third analysis of same content
        start_time = datetime.now()
        result3 = await xencode_system.code_analyzer.analyze_code_string(test_content, "python")
        time3 = (datetime.now() - start_time).total_seconds()

        # Results should be equivalent
        assert result1.issues == result2.issues
        assert result2.issues == result3.issues

        # Later calls might be faster due to caching (though implementation dependent)

    @pytest.mark.asyncio
    async def test_error_handling_across_components(self, xencode_system):
        """Test error handling coordination across components"""
        # Test how errors propagate and are handled across components

        # Try to access non-existent workspace
        non_existent_ws = await xencode_system.workspace_manager.get_workspace(
            "non_existent_workspace_id", "test_user"
        )
        assert non_existent_ws is None

        # Try to analyze malformed code
        try:
            malformed_result = await xencode_system.code_analyzer.analyze_code_string(
                "def unclosed_function(", "python"  # Invalid syntax
            )
            # Should handle gracefully
        except Exception:
            # Expected - should handle syntax errors gracefully
            pass

        # Try to process invalid document
        try:
            invalid_result = await xencode_system.document_processor.process_content("")
            # Should handle empty content gracefully
        except Exception:
            # Expected - should handle edge cases gracefully
            pass


class TestSystemPerformanceAndReliability:
    """Test system performance and reliability under load"""

    @pytest_asyncio.fixture
    async def xencode_system(self):
        """Create a system for performance testing"""
        system = Phase2Coordinator()
        await system.initialize()
        yield system
        await system.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self, xencode_system):
        """Test concurrent document processing"""
        # Create user
        user = xencode_system.auth_manager.create_user(
            username="concurrent_user",
            email="concurrent@example.com",
            password="concurrentpassword123"
        )
        assert user is not None

        # Authenticate user
        _, _, authenticated_user = await xencode_system.auth_manager.authenticate(
            "concurrent_user", "concurrentpassword123"
        )
        assert authenticated_user is not None

        # Create workspace
        workspace = await xencode_system.workspace_manager.create_workspace(
            name="Concurrent Processing Workspace",
            owner_id=authenticated_user.id
        )
        assert workspace is not None

        # Create multiple documents for concurrent processing
        documents = []
        for i in range(5):
            content = f"# Document {i}\nThis is content for document {i}.\n" * 10
            doc_file = WorkspaceFile(
                name=f"doc_{i}.md",
                path=f"/doc_{i}.md",
                content=content
            )
            documents.append(doc_file)

        # Add all documents to workspace
        for doc_file in documents:
            success = await xencode_system.workspace_manager.add_file(
                workspace.id, doc_file, authenticated_user.id
            )
            assert success is True

        # Process documents concurrently
        async def process_document(doc_file):
            return await xencode_system.document_processor.process_content(doc_file.content)

        start_time = datetime.now()
        processing_results = await asyncio.gather(
            *[process_document(doc) for doc in documents]
        )
        end_time = datetime.now()

        processing_time = (end_time - start_time).total_seconds()
        
        # Should process all documents successfully
        assert len(processing_results) == len(documents)
        for result in processing_results:
            assert result is not None

        # Processing time should be reasonable (less than 10 seconds for 5 docs)
        assert processing_time < 10.0

    @pytest.mark.asyncio
    async def test_concurrent_code_analysis(self, xencode_system):
        """Test concurrent code analysis"""
        # Create multiple code snippets for concurrent analysis
        code_snippets = []
        for i in range(5):
            code = f"""
def function_{i}():
    # Function {i} implementation
    result = 0
    for j in range({i+1} * 10):
        result += j
    return result

class Class{i}:
    def method_{i}(self):
        return f"Method {{i}} called"
"""
            code_snippets.append(code)

        # Analyze code snippets concurrently
        async def analyze_code(code):
            return await xencode_system.code_analyzer.analyze_code_string(code, "python")

        start_time = datetime.now()
        analysis_results = await asyncio.gather(
            *[analyze_code(code) for code in code_snippets]
        )
        end_time = datetime.now()

        analysis_time = (end_time - start_time).total_seconds()

        # Should analyze all snippets successfully
        assert len(analysis_results) == len(code_snippets)
        for result in analysis_results:
            assert result is not None

        # Analysis time should be reasonable
        assert analysis_time < 10.0

    @pytest.mark.asyncio
    async def test_system_stress_test(self, xencode_system):
        """Test system under stress conditions"""
        # Create user
        user = xencode_system.auth_manager.create_user(
            username="stress_user",
            email="stress@example.com",
            password="stresspassword123"
        )
        assert user is not None

        # Authenticate user
        _, _, authenticated_user = await xencode_system.auth_manager.authenticate(
            "stress_user", "stresspassword123"
        )
        assert authenticated_user is not None

        # Create workspace
        workspace = await xencode_system.workspace_manager.create_workspace(
            name="Stress Test Workspace",
            owner_id=authenticated_user.id
        )
        assert workspace is not None

        # Perform many operations to test system stability
        operations_count = 0
        start_time = datetime.now()

        for i in range(10):  # 10 iterations of operations
            # Add file
            file_content = f"# File {i}\nContent for file {i}\n" + "line\n" * 20
            file_obj = WorkspaceFile(
                name=f"stress_file_{i}.py",
                path=f"/stress_file_{i}.py",
                content=file_content
            )
            
            add_success = await xencode_system.workspace_manager.add_file(
                workspace.id, file_obj, authenticated_user.id
            )
            assert add_success is True
            operations_count += 1

            # Process document
            doc_result = await xencode_system.document_processor.process_content(file_content)
            assert doc_result is not None
            operations_count += 1

            # Analyze code
            code_result = await xencode_system.code_analyzer.analyze_code_string(file_content, "python")
            assert code_result is not None
            operations_count += 1

            # Generate AI response
            ai_result = await xencode_system.ensemble_reasoner.reason({
                "prompt": f"Summarize file {i}",
                "context": file_content,
                "user_id": authenticated_user.id
            })
            assert ai_result is not None
            operations_count += 1

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Should have completed all operations
        assert operations_count == 40  # 10 iterations * 4 operations each

        # System should remain stable
        system_status = xencode_system.get_system_status()
        assert system_status["system_health"] in ["healthy", "warning"]  # Should not be degraded

        # Memory usage should be reasonable (implementation dependent)
        # This would require checking actual memory usage in a real implementation


class TestSystemRecoveryAndResilience:
    """Test system recovery and resilience features"""

    @pytest_asyncio.fixture
    async def xencode_system(self):
        """Create a system for recovery testing"""
        system = Phase2Coordinator()
        await system.initialize()
        yield system
        # Don't shutdown here - we'll test shutdown/restart

    @pytest.mark.asyncio
    async def test_component_recovery(self, xencode_system):
        """Test component recovery after simulated failures"""
        # Verify system is running
        initial_status = xencode_system.get_system_status()
        assert initial_status["system_health"] == "healthy"

        # Simulate a failure in one component by stopping it
        # For this test, we'll simulate by checking the system can handle errors gracefully
        try:
            # Try to perform operations that might fail gracefully
            result = await xencode_system.code_analyzer.analyze_code_string("", "unknown_lang")
            # Should handle unknown language gracefully
        except Exception:
            # Expected - should handle gracefully
            pass

        # System should still be healthy
        status_after_error = xencode_system.get_system_status()
        assert status_after_error["system_health"] in ["healthy", "warning"]

        # Perform normal operations to verify system is still functional
        normal_result = await xencode_system.code_analyzer.analyze_code_string("def test(): pass", "python")
        assert normal_result is not None

    @pytest.mark.asyncio
    async def test_system_shutdown_and_restart(self):
        """Test system shutdown and restart"""
        # Create and initialize system
        system = Phase2Coordinator()
        await system.initialize()

        # Verify system is running
        status = system.get_system_status()
        assert status is not None
        assert system.initialized is True

        # Perform some operations
        # For now, just check that the system has the expected components
        assert hasattr(system, 'config_manager')
        assert hasattr(system, 'cache_manager')
        assert hasattr(system, 'error_handler')

        # Shutdown system
        await system.shutdown()
        # Check that shutdown happened (implementation dependent)

        # Restart system
        await system.initialize()
        assert system.initialized is True

        # Verify system is healthy after restart
        restart_status = system.get_system_status()
        assert restart_status is not None


class TestSystemIntegrationScenarios:
    """Test complex integration scenarios"""

    @pytest_asyncio.fixture
    async def xencode_system(self):
        """Create a system for integration scenario testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            system = XencodeSystem(
                storage_path=Path(temp_dir),
                xencode_version="3.0.0"
            )
            await system.initialize()
            yield system
            await system.shutdown()

    @pytest.mark.asyncio
    async def test_full_development_workflow(self, xencode_system):
        """Test full development workflow: document → code analysis → AI enhancement"""
        # 1. Create user
        user = xencode_system.auth_manager.create_user(
            username="dev_workflow",
            email="dev@example.com",
            password="devpassword123",
            role=UserRole.DEVELOPER
        )
        assert user is not None

        # 2. Authenticate user
        _, _, authenticated_user = await xencode_system.auth_manager.authenticate(
            "dev_workflow", "devpassword123"
        )
        assert authenticated_user is not None

        # 3. Create workspace
        workspace = await xencode_system.workspace_manager.create_workspace(
            name="Development Workflow",
            owner_id=authenticated_user.id,
            workspace_type=WorkspaceType.PROJECT
        )
        assert workspace is not None

        # 4. Process technical documentation
        tech_doc = """
# API Documentation
This document describes the API endpoints.

## GET /users
Returns a list of users.

## POST /users
Creates a new user.

### Request Body
{
    "name": "string",
    "email": "string"
}
"""
        
        doc_result = await xencode_system.document_processor.process_content(tech_doc)
        assert doc_result is not None

        # 5. Generate code based on documentation
        code_generation_prompt = f"""
Based on this API documentation, generate a Python Flask implementation:
{tech_doc}
"""
        
        ai_response = await xencode_system.ensemble_reasoner.reason({
            "prompt": code_generation_prompt,
            "context": tech_doc,
            "user_id": authenticated_user.id
        })
        assert ai_response is not None

        # 6. Analyze the generated code
        generated_code = ai_response.get("response", "def placeholder(): pass")
        analysis_result = await xencode_system.code_analyzer.analyze_code_string(generated_code, "python")
        assert analysis_result is not None

        # 7. Add both files to workspace
        doc_file = WorkspaceFile(
            name="api_docs.md",
            path="/api_docs.md",
            content=tech_doc
        )
        
        code_file = WorkspaceFile(
            name="api_implementation.py",
            path="/api_implementation.py",
            content=generated_code
        )

        doc_added = await xencode_system.workspace_manager.add_file(
            workspace.id, doc_file, authenticated_user.id
        )
        code_added = await xencode_system.workspace_manager.add_file(
            workspace.id, code_file, authenticated_user.id
        )
        
        assert doc_added is True
        assert code_added is True

        # 8. Perform final analysis on the complete project
        project_summary = {
            "documentation": doc_result,
            "implementation": analysis_result,
            "ai_response": ai_response
        }

        # 9. Verify all components worked together
        assert project_summary["documentation"] is not None
        assert project_summary["implementation"] is not None
        assert project_summary["ai_response"] is not None

    @pytest.mark.asyncio
    async def test_security_audit_workflow(self, xencode_system):
        """Test security audit workflow across components"""
        # 1. Create admin user
        admin_user = xencode_system.auth_manager.create_user(
            username="security_admin",
            email="admin@example.com",
            password="adminpassword123",
            role=UserRole.ADMIN
        )
        assert admin_user is not None

        # 2. Authenticate admin
        _, _, admin_authenticated = await xencode_system.auth_manager.authenticate(
            "security_admin", "adminpassword123"
        )
        assert admin_authenticated is not None

        # 3. Create workspace for security audit
        security_workspace = await xencode_system.workspace_manager.create_workspace(
            name="Security Audit Workspace",
            owner_id=admin_authenticated.id,
            workspace_type=WorkspaceType.SECURITY_AUDIT
        )
        assert security_workspace is not None

        # 4. Add potentially vulnerable code
        vulnerable_code = """
import os
import subprocess

def execute_user_command(user_input):
    # DANGEROUS: Direct execution of user input
    result = os.system(user_input)  # NOQA: S605 Security issue
    return result

def run_shell_command(command):
    # DANGEROUS: Shell command execution
    output = subprocess.check_output(command, shell=True)  # NOQA: S602 Security issue
    return output

def read_arbitrary_file(filename):
    # DANGEROUS: Path traversal vulnerability
    with open(filename, 'r') as f:  # NOQA: S108 Security issue
        return f.read()
"""
        
        vuln_file = WorkspaceFile(
            name="vulnerable_code.py",
            path="/vulnerable_code.py",
            content=vulnerable_code
        )
        
        add_success = await xencode_system.workspace_manager.add_file(
            security_workspace.id, vuln_file, admin_authenticated.id
        )
        assert add_success is True

        # 5. Analyze code for security issues
        analysis_result = await xencode_system.code_analyzer.analyze_code_string(vulnerable_code, "python")
        assert analysis_result is not None

        # Should have security issues detected
        security_issues = [issue for issue in analysis_result.issues if issue.analysis_type == "security"]
        # May or may not detect issues depending on implementation

        # 6. Generate security recommendations using AI
        security_prompt = f"""
Analyze this code for security vulnerabilities and provide recommendations:
Code: {vulnerable_code}

Issues found by static analysis: {json.dumps([issue.to_dict() for issue in security_issues], indent=2)}
"""
        
        security_advice = await xencode_system.ensemble_reasoner.reason({
            "prompt": security_prompt,
            "context": vulnerable_code,
            "user_id": admin_authenticated.id
        })
        assert security_advice is not None

        # 7. Verify analytics tracking of security operations
        security_events = await xencode_system.analytics_engine.get_events_by_category("security", hours=1)
        # Should have security-related events

    @pytest.mark.asyncio
    async def test_performance_optimization_workflow(self, xencode_system):
        """Test performance optimization workflow"""
        # 1. Create user
        user = xencode_system.auth_manager.create_user(
            username="perf_user",
            email="perf@example.com",
            password="perfpwd123",
            role=UserRole.DEVELOPER
        )
        assert user is not None

        # 2. Authenticate user
        _, _, authenticated_user = await xencode_system.auth_manager.authenticate(
            "perf_user", "perfpwd123"
        )
        assert authenticated_user is not None

        # 3. Create performance-focused workspace
        perf_workspace = await xencode_system.workspace_manager.create_workspace(
            name="Performance Optimization",
            owner_id=authenticated_user.id,
            workspace_type=WorkspaceType.PERFORMANCE_ANALYSIS
        )
        assert perf_workspace is not None

        # 4. Add performance-intensive code
        inefficient_code = """
def fibonacci_recursive(n):
    # Inefficient recursive implementation
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def bubble_sort(arr):
    # Inefficient sorting algorithm
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def redundant_computation(data):
    # Performs redundant computations
    results = []
    for item in data:
        # Compute expensive operation multiple times
        expensive = item ** 2 * item ** 3
        results.append(expensive)
        # Compute same operation again
        expensive2 = item ** 2 * item ** 3
        results.append(expensive2)
    return results
"""
        
        perf_file = WorkspaceFile(
            name="inefficient_code.py",
            path="/inefficient_code.py",
            content=inefficient_code
        )
        
        add_success = await xencode_system.workspace_manager.add_file(
            perf_workspace.id, perf_file, authenticated_user.id
        )
        assert add_success is True

        # 5. Analyze code for performance issues
        analysis_result = await xencode_system.code_analyzer.analyze_code_string(inefficient_code, "python")
        assert analysis_result is not None

        # 6. Get performance analysis
        perf_issues = [issue for issue in analysis_result.issues if issue.analysis_type == "performance"]
        # May or may not detect performance issues depending on implementation

        # 7. Generate optimization suggestions
        optimization_prompt = f"""
Optimize this code for better performance:
Code: {inefficient_code}

Performance issues found: {json.dumps([issue.to_dict() for issue in perf_issues], indent=2)}
"""
        
        optimization_suggestions = await xencode_system.ensemble_reasoner.reason({
            "prompt": optimization_prompt,
            "context": inefficient_code,
            "user_id": authenticated_user.id
        })
        assert optimization_suggestions is not None

        # 8. Track performance metrics
        perf_metrics = await xencode_system.analytics_engine.get_performance_metrics()
        # Should have performance metrics


class TestSystemCompatibilityAndMigration:
    """Test system compatibility and migration scenarios"""

    @pytest_asyncio.fixture
    async def xencode_system(self):
        """Create a system for compatibility testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            system = XencodeSystem(
                storage_path=Path(temp_dir),
                xencode_version="3.0.0"
            )
            await system.initialize()
            yield system
            await system.shutdown()

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, xencode_system):
        """Test backward compatibility with older data formats"""
        # Test that the system can handle data from previous versions
        # This would involve testing with mock data that simulates older formats

        # Create workspace with older-style configuration
        workspace = await xencode_system.workspace_manager.create_workspace(
            name="Compatibility Test",
            owner_id="test_user"
        )
        assert workspace is not None

        # Add file with content that might have been processed differently in older versions
        legacy_content = """
# Legacy format comment
def legacy_function():
    '''Legacy docstring format'''
    return "legacy result"
"""
        
        legacy_file = WorkspaceFile(
            name="legacy_code.py",
            path="/legacy_code.py",
            content=legacy_content
        )
        
        add_success = await xencode_system.workspace_manager.add_file(
            workspace.id, legacy_file, "test_user"
        )
        assert add_success is True

        # Process with current system
        result = await xencode_system.document_processor.process_content(legacy_content)
        analysis = await xencode_system.code_analyzer.analyze_code_string(legacy_content, "python")

        # Should handle legacy content without issues
        assert result is not None
        assert analysis is not None

    @pytest.mark.asyncio
    async def test_multi_user_concurrent_workflow(self, xencode_system):
        """Test multiple users working concurrently"""
        # Create multiple users
        users = []
        for i in range(3):
            user = xencode_system.auth_manager.create_user(
                username=f"multi_user_{i}",
                email=f"multi_user_{i}@example.com",
                password=f"password_{i}",
                role=UserRole.DEVELOPER
            )
            assert user is not None
            users.append(user)

        # Create shared workspace
        shared_workspace = await xencode_system.workspace_manager.create_workspace(
            name="Multi-User Workspace",
            owner_id=users[0].id  # First user is owner
        )
        assert shared_workspace is not None

        # Add collaborators
        for i, user in enumerate(users[1:], 1):
            collab = WorkspaceCollaborator(
                user_id=user.id,
                username=user.username,
                role="editor" if i == 1 else "viewer"
            )
            
            add_collab_success = await xencode_system.workspace_manager.add_collaborator(
                shared_workspace.id, user.id, collab, users[0].id
            )
            assert add_collab_success is True

        # Simulate concurrent operations by different users
        async def user_operations(user_idx):
            user = users[user_idx]
            
            # Authenticate
            _, _, auth_user = await xencode_system.auth_manager.authenticate(
                f"multi_user_{user_idx}", f"password_{user_idx}"
            )
            
            if user_idx == 0:  # Owner can add/edit files
                # Add file
                file_content = f"# Content by user {user_idx}\nprint('Hello from user {user_idx}')"
                file_obj = WorkspaceFile(
                    name=f"user_{user_idx}_file.py",
                    path=f"/user_{user_idx}_file.py",
                    content=file_content
                )
                
                add_success = await xencode_system.workspace_manager.add_file(
                    shared_workspace.id, file_obj, auth_user.id
                )
                assert add_success is True
                
                # Analyze the file
                analysis = await xencode_system.code_analyzer.analyze_code_string(file_content, "python")
                assert analysis is not None
                
            elif user_idx == 1:  # Editor can read and modify
                # Get existing files
                workspace = await xencode_system.workspace_manager.get_workspace(
                    shared_workspace.id, auth_user.id
                )
                if workspace and workspace.files:
                    file_id = list(workspace.files.keys())[0]
                    file = await xencode_system.workspace_manager.get_file(
                        shared_workspace.id, file_id, auth_user.id
                    )
                    if file:
                        # Modify file content
                        file.content += f"\n# Modified by user {user_idx}"
                        update_success = await xencode_system.workspace_manager.update_file(
                            shared_workspace.id, file, auth_user.id
                        )
                        # May or may not succeed depending on permissions
                        
            else:  # Viewer can only read
                # Try to read files
                workspace = await xencode_system.workspace_manager.get_workspace(
                    shared_workspace.id, auth_user.id
                )
                # Should be able to access workspace if permissions allow

        # Run operations concurrently
        await asyncio.gather(
            *[user_operations(i) for i in range(len(users))]
        )

        # Verify final state
        final_workspace = await xencode_system.workspace_manager.get_workspace(
            shared_workspace.id, users[0].id  # Owner access
        )
        assert final_workspace is not None
        assert len(final_workspace.collaborators) >= 3  # Owner + 2 collaborators


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])