#!/usr/bin/env python3
"""
Tests for File Operations Plugin

TDD tests for the file system operations plugin including ls_dir, read_file,
search capabilities, and secure mutations with RBAC.
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# These imports will fail initially - that's the point of TDD!
try:
    from xencode.plugins.file_operations import FileOperationsPlugin, PluginContext
    PLUGIN_AVAILABLE = True
except ImportError:
    PLUGIN_AVAILABLE = False
    # Mock classes for initial failing tests
    class FileOperationsPlugin:
        def __init__(self, context): pass
        async def ls_dir(self, path): raise NotImplementedError("ls_dir not implemented")
        async def read_file(self, path, binary=False): raise NotImplementedError("read_file not implemented")
    
    class PluginContext:
        def __init__(self): 
            self.workspace_id = "test_workspace"
            self.user_id = "test_user"
            self.permissions = ["file:read", "file:write", "file:search"]


class TestFileOperationsPluginBase:
    """Test base file operations: ls_dir and read_file"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)
            
            # Create test structure
            (workspace_path / "test_file.txt").write_text("Hello, World!")
            (workspace_path / "test_binary.bin").write_bytes(b"\x00\x01\x02\x03")
            (workspace_path / "subdir").mkdir()
            (workspace_path / "subdir" / "nested.py").write_text("print('nested')")
            
            yield workspace_path
    
    @pytest.fixture
    def plugin_context(self, temp_workspace):
        """Create plugin context for testing"""
        context = PluginContext()
        context.workspace_root = temp_workspace
        context.permissions = ["file:read", "file:write", "file:search", "file:view", "file:lint", "file:delete"]
        return context
    
    @pytest.fixture
    def file_plugin(self, plugin_context):
        """Create file operations plugin instance"""
        return FileOperationsPlugin(plugin_context)
    
    @pytest.mark.asyncio
    async def test_ls_dir_not_implemented_initially(self, file_plugin, temp_workspace):
        """Test that ls_dir raises NotImplementedError initially (failing test)"""
        with pytest.raises(NotImplementedError, match="ls_dir not implemented"):
            await file_plugin.ls_dir(temp_workspace)
    
    @pytest.mark.asyncio
    async def test_read_file_not_implemented_initially(self, file_plugin, temp_workspace):
        """Test that read_file raises NotImplementedError initially (failing test)"""
        test_file = temp_workspace / "test_file.txt"
        with pytest.raises(NotImplementedError, match="read_file not implemented"):
            await file_plugin.read_file(test_file)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_ls_dir_returns_correct_structure(self, file_plugin, temp_workspace):
        """Test ls_dir returns correct directory structure"""
        result = await file_plugin.ls_dir(temp_workspace)
        
        assert isinstance(result, dict)
        assert "files" in result
        assert "dirs" in result
        
        # Check files
        assert "test_file.txt" in result["files"]
        assert "test_binary.bin" in result["files"]
        
        # Check directories
        assert "subdir" in result["dirs"]
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_read_file_text_content(self, file_plugin, temp_workspace):
        """Test reading text file content"""
        test_file = temp_workspace / "test_file.txt"
        content = await file_plugin.read_file(test_file)
        
        assert content == "Hello, World!"
        assert isinstance(content, str)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_read_file_binary_content(self, file_plugin, temp_workspace):
        """Test reading binary file content"""
        test_file = temp_workspace / "test_binary.bin"
        content = await file_plugin.read_file(test_file, binary=True)
        
        assert content == b"\x00\x01\x02\x03"
        assert isinstance(content, bytes)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_ls_dir_nonexistent_path(self, file_plugin, temp_workspace):
        """Test ls_dir with nonexistent path raises appropriate error"""
        nonexistent = temp_workspace / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            await file_plugin.ls_dir(nonexistent)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_read_file_nonexistent(self, file_plugin, temp_workspace):
        """Test read_file with nonexistent file raises appropriate error"""
        nonexistent = temp_workspace / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            await file_plugin.read_file(nonexistent)


class TestFileOperationsSearch:
    """Test search operations: pathname search, content search, in-file search"""
    
    @pytest.fixture
    def temp_workspace_with_code(self):
        """Create workspace with code files for search testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)
            
            # Create Python files
            (workspace_path / "main.py").write_text("""
import os
import sys

def main():
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    main()
""")
            
            (workspace_path / "utils.py").write_text("""
import json
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    with open("config.json") as f:
        return json.load(f)
""")
            
            # Create subdirectory with more files
            (workspace_path / "tests").mkdir()
            (workspace_path / "tests" / "test_main.py").write_text("""
import pytest
from main import main

def test_main():
    assert main() == 0
""")
            
            yield workspace_path
    
    @pytest.fixture
    def search_plugin(self, temp_workspace_with_code):
        """Create plugin for search testing"""
        context = PluginContext()
        context.workspace_root = temp_workspace_with_code
        return FileOperationsPlugin(context)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_search_pathnames_only_python_files(self, search_plugin, temp_workspace_with_code):
        """Test searching for Python files by pathname pattern"""
        results = await search_plugin.search_pathnames_only("*.py")
        
        assert len(results) >= 3
        py_files = [str(p) for p in results]
        assert any("main.py" in f for f in py_files)
        assert any("utils.py" in f for f in py_files)
        assert any("test_main.py" in f for f in py_files)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_search_for_files_with_import(self, search_plugin, temp_workspace_with_code):
        """Test searching for files containing 'import' keyword"""
        results = list(await search_plugin.search_for_files("import", recursive=True))
        
        assert len(results) >= 2
        # Should find files with import statements
        file_paths = [str(result[0]) for result in results]
        assert any("main.py" in f for f in file_paths)
        assert any("utils.py" in f for f in file_paths)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_search_in_file_specific_content(self, search_plugin, temp_workspace_with_code):
        """Test searching within a specific file for content"""
        main_py = temp_workspace_with_code / "main.py"
        results = await search_plugin.search_in_file(main_py, "def main")
        
        assert len(results) >= 1
        line_num, snippet = results[0]
        assert isinstance(line_num, int)
        assert "def main" in snippet


class TestFileOperationsTree:
    """Test directory tree visualization"""
    
    @pytest.fixture
    def tree_plugin(self, temp_workspace_with_code):
        """Create plugin for tree testing"""
        context = PluginContext()
        context.workspace_root = temp_workspace_with_code
        return FileOperationsPlugin(context)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_get_dir_tree_basic(self, tree_plugin, temp_workspace_with_code):
        """Test basic directory tree generation"""
        tree_str = await tree_plugin.get_dir_tree(temp_workspace_with_code, depth=2)
        
        assert isinstance(tree_str, str)
        assert "├──" in tree_str or "└──" in tree_str
        assert "main.py" in tree_str
        assert "utils.py" in tree_str
        assert "tests" in tree_str


class TestFileOperationsLint:
    """Test lint error reading"""
    
    @pytest.fixture
    def lint_workspace(self):
        """Create workspace with Python file containing lint errors"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)
            
            # Create Python file with intentional lint errors
            (workspace_path / "bad_code.py").write_text("""
import os,sys
def bad_function( ):
    x=1+2
    print( "hello world" )
    return x
""")
            
            yield workspace_path
    
    @pytest.fixture
    def lint_plugin(self, lint_workspace):
        """Create plugin for lint testing"""
        context = PluginContext()
        context.workspace_root = lint_workspace
        return FileOperationsPlugin(context)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_read_lint_errors(self, lint_plugin, lint_workspace):
        """Test reading lint errors from Python file"""
        bad_file = lint_workspace / "bad_code.py"
        errors = await lint_plugin.read_lint_errors(bad_file)
        
        assert isinstance(errors, list)
        if errors:  # Only check if linter is available
            error = errors[0]
            assert "line" in error
            assert "msg" in error
            assert "severity" in error


class TestFileOperationsMutations:
    """Test file/folder creation, deletion, and editing (with RBAC)"""
    
    @pytest.fixture
    def mutation_workspace(self):
        """Create workspace for mutation testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)
            
            # Create initial file for editing
            (workspace_path / "edit_me.txt").write_text("old content\nkeep this line\nold content")
            
            yield workspace_path
    
    @pytest.fixture
    def mutation_plugin(self, mutation_workspace):
        """Create plugin for mutation testing"""
        context = PluginContext()
        context.workspace_root = mutation_workspace
        context.permissions = ["file:read", "file:write", "file:delete"]
        return FileOperationsPlugin(context)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_create_file_with_content(self, mutation_plugin, mutation_workspace):
        """Test creating a new file with content"""
        new_file = mutation_workspace / "new_file.txt"
        
        await mutation_plugin.create_file_or_folder(new_file, content="Hello, new file!")
        
        assert new_file.exists()
        assert new_file.read_text() == "Hello, new file!"
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_create_folder(self, mutation_plugin, mutation_workspace):
        """Test creating a new folder"""
        new_dir = mutation_workspace / "new_directory"
        
        await mutation_plugin.create_file_or_folder(new_dir / "", is_directory=True)
        
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_delete_file(self, mutation_plugin, mutation_workspace):
        """Test deleting a file"""
        file_to_delete = mutation_workspace / "edit_me.txt"
        assert file_to_delete.exists()
        
        await mutation_plugin.delete_file_or_folder(file_to_delete)
        
        assert not file_to_delete.exists()
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_edit_file_content(self, mutation_plugin, mutation_workspace):
        """Test editing file content with search and replace"""
        file_to_edit = mutation_workspace / "edit_me.txt"
        
        await mutation_plugin.edit_file(file_to_edit, search="old content", replace="new content")
        
        content = file_to_edit.read_text()
        assert "new content" in content
        assert "old content" not in content
        assert "keep this line" in content  # Should preserve other content
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_rbac_permission_denied(self, mutation_workspace):
        """Test RBAC permission denial for mutations"""
        # Create plugin with limited permissions
        context = PluginContext()
        context.workspace_root = mutation_workspace
        context.permissions = ["file:read"]  # No write permission
        
        limited_plugin = FileOperationsPlugin(context)
        
        with pytest.raises(PermissionError, match="file:write"):
            await limited_plugin.create_file_or_folder(
                mutation_workspace / "forbidden.txt", 
                content="Should not work"
            )


class TestFileOperationsIntegration:
    """Integration tests for file operations plugin"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_plugin_registration_and_execution(self):
        """Test that plugin can be registered and executed through plugin manager"""
        # This test will be implemented when we wire to coordinator
        pass
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PLUGIN_AVAILABLE, reason="Plugin not implemented yet")
    async def test_api_endpoint_integration(self):
        """Test file operations through API endpoints"""
        # This test will be implemented when we add API endpoints
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])