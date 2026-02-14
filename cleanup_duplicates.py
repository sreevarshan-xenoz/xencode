#!/usr/bin/env python3
"""
Script to remove duplicate and unwanted files from the workspace.
"""

import os
import sys

# Files to remove - organized by category
DUPLICATE_FILES = [
    # Agent memory databases (keep only what's needed)
    "agent_agent_2_memory.db",
    "agent_basic_code_memory.db",
    "agent_basic_execution_memory.db",
    "agent_basic_general_memory.db",
    "agent_basic_planning_memory.db",
    "agent_basic_research_memory.db",
    "agent_code_memory.db",
    "agent_execution_memory.db",
    "agent_general_memory.db",
    "agent_planning_memory.db",
    "agent_research_memory.db",
    "agent_test_agent_1_memory.db",
    
    # Security databases (duplicates)
    "security_access_control.db",
    "security_audit.db",
    "security_identity.db",
    "security_privacy.db",
    
    # Other duplicate databases
    "advanced_analytics.db",
    "checkpoints.db",
    "chroma.sqlite3",
    "domain_knowledge.db",
    "metrics.db",
    "resource_management.db",
    "shared_knowledge.db",
    "supervision.db",
    "workflows.db",
    "workspaces.db",
    
    # Duplicate test files
    "test_basic_functionality_fixed.py",
    "comprehensive_test.py",
    "full_integration_test.py",
    "test_bytebot_diagnosis.py",
    "test_tui_launch.py",
    
    # Backup files
    "xencode_core.py.bak",
    
    # Coverage files
    ".coverage",
    "coverage.xml",
    
    # Log files
    "audit.log",
    "test_audit.log",
    
    # Temporary/update files
    "updated_tasks.md",
]

def main():
    files_to_delete = []
    
    for filename in DUPLICATE_FILES:
        if os.path.exists(filename):
            files_to_delete.append(filename)
    
    if not files_to_delete:
        print("No duplicate files found.")
        return 0
    
    deleted_count = 0
    errors = []
    
    for filename in files_to_delete:
        try:
            os.remove(filename)
            print(f"Deleted: {filename}")
            deleted_count += 1
        except Exception as e:
            errors.append(f"Error deleting {filename}: {e}")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  {error}")
    
    print(f"\nSuccessfully deleted {deleted_count} files.")
    return 0 if not errors else 1

if __name__ == "__main__":
    sys.exit(main())
