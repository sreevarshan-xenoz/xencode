import pytest
import asyncio
import os
import shutil
from pathlib import Path
from git import Repo
from xencode.tui.widgets.git_status import GitStatusManager

@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repo"""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    repo = Repo.init(repo_dir)
    
    # Configure git user for commit
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()
    
    return repo_dir, repo

@pytest.mark.asyncio
async def test_git_commit_flow(temp_git_repo):
    """Test get_diff and commit flow"""
    repo_dir, repo = temp_git_repo
    manager = GitStatusManager(repo_dir)
    
    # 1. Create a file and stage it
    test_file = repo_dir / "test.txt"
    test_file.write_text("Hello World")
    
    repo.index.add(["test.txt"])
    
    # 2. Verify diff (staged)
    # Note: git diff --staged might be empty for new file if not committed yet? 
    # Actually for new file it shows in diff --staged
    
    diff = await manager.get_diff(staged=True)
    print(f"Diff: {diff}")
    assert "test.txt" in diff or "Hello World" in diff or "new file" in diff
    
    # 3. Commit
    success = await manager.commit("Initial commit")
    assert success
    
    # 4. Verify commit
    assert repo.head.commit.message.strip() == "Initial commit"
    assert not repo.is_dirty()
    
    # 5. Modify file and stage
    test_file.write_text("Hello World Updated")
    repo.index.add(["test.txt"])
    
    # 6. Verify diff again
    diff = await manager.get_diff(staged=True)
    assert "Updated" in diff
    
    # 7. Commit again
    success = await manager.commit("Update file")
    assert success
    assert repo.head.commit.message.strip() == "Update file"
