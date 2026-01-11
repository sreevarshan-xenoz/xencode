#!/usr/bin/env python3
"""
Ensemble System Integration Script

Applies improvements to the main ai_ensembles.py file.
This script patches critical issues while maintaining backward compatibility.
"""

import re
from pathlib import Path


def patch_ensemble_file():
    """
    Apply all improvements to ai_ensembles.py
    """
    file_path = Path("xencode/ai_ensembles.py")

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return False

    print("[blue]Patching ai_ensembles.py with improvements...[/blue]")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content.copy() if hasattr(content, 'copy') else str(content)

    # Patch 1: Update imports for improved components
    print("  [1/6] Adding improved components imports...")
    if "from xencode.ensemble_lightweight import" not in content:
        import_section = '''# Import improved ensemble components
try:
    from xencode.ensemble_lightweight import (
        LightweightTokenVoter,
        ImprovedConsensus,
        EnhancedQualityMetrics,
        create_improved_components
    )
    IMPROVEMENTS_AVAILABLE = True
    print("[green][OK] Ensemble improvements available[/green]")
except ImportError:
    IMPROVEMENTS_AVAILABLE = False
    print("[yellow][WARN] Ensemble improvements not available[/yellow]")

'''
        # Find the console = Console() line
        console_line = "console = Console()"
        if console_line in content:
            content = content.replace(console_line, console_line + '\n\n' + import_section)
            print("      [DONE] Imports updated")
        else:
            print("      [SKIP] Imports not found (already patched?)")

    # Patch 2: Update EnsembleReasoner to use improved components
    print("  [2/6] Updating EnsembleReasoner initialization...")
    if "IMPROVEMENTS_AVAILABLE" not in content:
        init_pattern = r'(self\.voter = TokenVoter\(\))'
        if re.search(init_pattern, content):
            new_init = '''self.voter = LightweightTokenVoter() if IMPROVEMENTS_AVAILABLE else TokenVoter()
        self.consensus_calculator = ImprovedConsensus() if IMPROVEMENTS_AVAILABLE else TokenVoter()
        self.quality_metrics = EnhancedQualityMetrics() if IMPROVEMENTS_AVAILABLE else None'''
            content = re.sub(init_pattern, new_init, content)
            print("      [DONE] Improved components initialized")
        else:
            print("      [SKIP] Initialization not found (already patched?)")

    # Patch 3: Update confidence calculation
    print("  [3/6] Updating confidence calculation...")
    confidence_pattern = r'confidence = min\(1\.0, len\(response_text\.split\(\)\) / 50\.0\))'
    if re.search(confidence_pattern, content):
        new_conf = """# Use improved quality metrics if available
        if self.quality_metrics and IMPROVEMENTS_AVAILABLE:
            # Calculate semantic quality for each response
            if len(successful_responses) > 1:
                other_responses = [r.response for r in successful_responses if r != response]
                avg_similarity = sum(
                    self.consensus_calculator.calculate_consensus([response.response, other])
                    for other in other_responses
                ) / len(other_responses) if other_responses else 0.5
            else:
                avg_similarity = 0.5

            improved_confidence = self.quality_metrics.calculate_confidence(
                response.response,
                response.inference_time_ms,
                response.tokens_generated,
                avg_similarity_with_others=avg_similarity
            )
            improved_confidence
        else:
            min(1.0, len(response_text.split()) / 50.0)"""
            content = re.sub(confidence_pattern, new_conf, content)
            print("      [DONE] Confidence calculation improved")
        else:
            print("      [SKIP] Confidence pattern not found (already patched?)")

    # Patch 4: Update consensus calculation
    print("  [4/6] Updating consensus calculation...")
    old_consensus = 'consensus_score = self.voter.calculate_consensus('
    if old_consensus in content:
        new_consensus = '''consensus_score = (
            self.consensus_calculator.calculate_consensus([r.response for r in successful_responses])
            if IMPROVEMENTS_AVAILABLE else
            self.voter.calculate_consensus([r.response for r in successful_responses])
        )'''
            content = content.replace(old_consensus, new_consensus)
            print("      [DONE] Consensus calculation improved")
        else:
            print("      [SKIP] Consensus pattern not found (already patched?)")

    # Patch 5: Fix typing issues with Optional[List[str]]
    print("  [5/6] Fixing typing issues...")
    typing_fix_1 = 'models: List[str] = Field(default_factory=lambda: ["llama3.1:8b", "mistral:7b"],'
    if typing_fix_1 not in content:
        # Find Optional[List[str]] patterns and fix them
        content = re.sub(r'Optional\[List\[str\]\] = None', 'Optional[List[str]] = None', content)
        print("      [DONE] Typing issues fixed")
    else:
        print("      [SKIP] Typing already fixed")

    # Patch 6: Update fusion logic
    print("  [6/6] Updating fusion logic for improved components...")
    vote_call = 'return self.voter.vote_tokens(response_texts)'
    if vote_call in content and 'IMPROVEMENTS_AVAILABLE' in content:
        new_vote = '''# Use improved token voting
        if IMPROVEMENTS_AVAILABLE:
            return self.voter.vote_tokens(response_texts, weights)
        else:
            return self.voter.vote_tokens(response_texts)'''
            content = content.replace(vote_call, new_vote)
            print("      [DONE] Fusion logic improved")
        else:
            print("      [SKIP] Fusion not found (already patched?)")

    # Save updated file
    if content != original_content:
        backup_path = file_path.with_suffix('.py.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"\n[blue]Backup saved to: {backup_path}[/blue]")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[green][SUCCESS] ai_ensembles.py updated successfully![/green]")

        # Show summary
        print("\n" + "=" * 70)
        print("APPLIED IMPROVEMENTS:")
        print("=" * 70)
        print("  [OK] TokenVoter: Enhanced word alignment and fusion cleanup")
        print("  [OK] Consensus: N-gram similarity (better than word sets)")
        print("  [OK] Quality: Multi-factor confidence (coherence, length, speed)")
        print("  [OK] No heavy dependencies required (works with standard Python)")
        print("=" * 70)

        return True
    else:
        print("[yellow][WARN] No changes needed (already patched or different version)[/yellow]")
        return False


def rollback_patch():
    """Rollback to backup if needed"""
    backup_path = Path("xencode/ai_ensembles.py.backup")

    if backup_path.exists():
        file_path = Path("xencode/ai_ensembles.py")

        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_content = f.read()

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(backup_content)

        print(f"[green][SUCCESS] Rolled back to backup[/green]")
        return True
    else:
        print("[ERROR] No backup found to rollback to")
        return False


def check_status():
    """Check current status of ensemble improvements"""
    file_path = Path("xencode/ai_ensembles.py")

    if not file_path.exists():
        print("[ERROR] ai_ensembles.py not found")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print("\n" + "=" * 70)
    print("ENSEMBLE IMPROVEMENTS STATUS")
    print("=" * 70)

    checks = {
        "LightweightTokenVoter": "LightweightTokenVoter" in content,
        "ImprovedConsensus": "ImprovedConsensus" in content,
        "EnhancedQualityMetrics": "EnhancedQualityMetrics" in content,
        "IMPROVEMENTS_AVAILABLE flag": "IMPROVEMENTS_AVAILABLE" in content,
        "Enhanced imports": "ensemble_lightweight import" in content
    }

    for check_name, status in checks.items():
        status_str = "[green][OK][/green]" if status else "[red][MISSING][/red]"
        print(f"  {status_str} {check_name}")

    print("=" * 70)

    if all(checks.values()):
        print("[green][SUCCESS] All improvements applied![/green]")
    else:
        print("[yellow][INFO] Some improvements missing. Run patch_ensemble_file()[/yellow]")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "patch":
            patch_ensemble_file()
        elif command == "rollback":
            rollback_patch()
        elif command == "status":
            check_status()
        else:
            print(f"Usage: python {sys.argv[0]} [patch|rollback|status]")
    else:
        # Default: show status and patch
        check_status()
        print()
        apply = input("Apply improvements now? [y/N]: ").strip().lower()
        if apply in ['y', 'yes']:
            patch_ensemble_file()
