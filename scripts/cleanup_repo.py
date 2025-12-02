#!/usr/bin/env python3
"""
Repository cleanup script - removes old documentation and duplicate test files.
"""

import os
import subprocess

# Files to delete
FILES_TO_DELETE = [
    # Old summary/report MD files (keeping only PLAN.md, README.md, CLI_GUIDE.md, QUICK_START.md)
    "AI_ML_LEVIATHAN_SUMMARY.md",
    "CRITICAL_FIXES.md",
    "CRUSH.md",
    "DEMO_IMMERSIVE.md",
    "ENHANCEMENT_SYSTEMS_README.md",
    "FINAL_SUMMARY.md",
    "FIXES_COMPLETED.md",
    "IMMERSIVE_MODE.md",
    "IMPLEMENTATION_SUMMARY.md",
    "PHASE1_COMPLETION_REPORT.md",
    "PHASE1_PROGRESS.md",
    "PHASE2_COMPLETION_REPORT.md",
    "QUICK_FIX_GUIDE.md",
    "README_FIXES.md",
    "SMART_MODEL_SELECTION.md",
    "WARP_TERMINAL_PROGRESS_REPORT.md",
    "WEEK3_AI_COMPLETION_REPORT.md",
    "XENCODE_ANALYSIS_AND_IMPROVEMENTS.md",
    "XENCODE_WARP_FINAL_SUMMARY.md",
    
    # Old/duplicate test files in root
    "test_advanced_analytics_dashboard.py",
    "test_analytics_infrastructure.py",
    "test_analytics_simple.py",
    "test_cli_integration.py",
    "test_enhancement_systems.py",
    "test_performance_dashboard.py",
    "test_phase2_comprehensive.py",
    "test_plugin_system.py",
    "test_plugin_system_standalone.py",
    "test_resource_management_simple.py",
    "test_workspace.py",
    "simple_test.py",
    
    # Duplicate test files
    "debug_chat.log",
    "debug_traceback.py",
    
    # Old benchmark/demo files
    "benchmark_leviathan.py",
    "quick_leviathan_demo.py",
    "example_enhanced_xencode.py",
    
    # Test fixtures
    "test_fixes.sh",
    "verify_fixes.py",
]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(base_dir)
    
    os.chdir(repo_root)
    
    deleted_count = 0
    not_found_count = 0
    
    print("=" * 80)
    print("REPOSITORY CLEANUP")
    print("=" * 80)
    print()
    
    for file_path in FILES_TO_DELETE:
        full_path = os.path.join(repo_root, file_path)
        
        if os.path.exists(full_path):
            print(f"Deleting: {file_path}")
            
            # Remove from git if tracked
            try:
                subprocess.run(
                    ["git", "rm", "-f", file_path],
                    capture_output=True,
                    check=False
                )
            except Exception:
                # If git rm fails, just delete the file
                os.remove(full_path)
            
            deleted_count += 1
        else:
            not_found_count += 1
    
    print()
    print("=" * 80)
    print(f"✅ Deleted: {deleted_count} files")
    print(f"⚠️  Not found: {not_found_count} files")
    print("=" * 80)
    print()
    print("Run 'git status' to see changes")
    print("Run 'git commit -m \"Clean up old documentation and duplicate test files\"' to commit")

if __name__ == "__main__":
    main()
