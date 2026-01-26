"""
SafetyGate - Blocks dangerous operations based on risk and execution mode

This component acts as a veto mechanism that can prevent dangerous operations
from executing based on risk assessment and execution mode.
"""

import re
from typing import Dict, Any, List
from enum import Enum
from datetime import datetime


class ExecutionMode(Enum):
    """Execution modes for ByteBot operations"""
    ASSIST = "assist"      # Suggest only, no execution
    EXECUTE = "execute"    # Auto-run safe steps, confirm risky ones
    AUTONOMOUS = "autonomous"  # Run everything except vetoed operations


class SafetyGate:
    """
    Safety gate that can block dangerous operations based on risk and mode
    """
    
    def __init__(self):
        # Dangerous command patterns that should always be blocked
        self.dangerous_patterns = [
            # System destruction
            r"rm\s+-rf\s+/",
            r"format\s+",
            r"del\s+/f\s+c:\\",
            r"mkfs\.",
            r"dd\s+if=",
            r"shred\s+",
            r"cat\s+/dev/zero\s+>\s+/dev/sda",
            r"echo\s+>\s*/proc/sysrq-trigger",
            r"sync\s+&&\s+echo\s+1\s+>\s+/proc/sys/kernel/sysrq\s+&&\s+echo\s+b\s+>\s+/proc/sysrq-trigger",
            
            # Privilege escalation with dangerous commands
            r"sudo\s+(rm|format|mkfs|dd|shred|del)",
            r"su\s+",
            r"runas\s+",
            
            # Critical system files
            r"/etc/",
            r"/boot/",
            r"/sys/",
            r"/proc/",
            r"C:\\Windows\\",
            r"C:\\ProgramData\\",
            
            # Potentially destructive network commands
            r"curl\s+.*\|.*sh",
            r"wget\s+.*\|.*sh",
        ]
        
        # Patterns that require confirmation even in execute mode
        self.medium_risk_patterns = [
            r"rm\s+-rf\s+",
            r"chmod\s+-R\s+777",
            r"chown\s+-R",
            r"mv\s+/.*\s+/",
            r"kill\s+-9\s+",
            r"pkill\s+-f",
            r"taskkill\s+/f",
            r"truncate\s+--size\s+0",
        ]
        
        # Execution mode thresholds
        self.mode_thresholds = {
            ExecutionMode.ASSIST: 0.0,      # Nothing executes in assist mode
            ExecutionMode.EXECUTE: 0.7,     # Block high-risk operations (>0.7)
            ExecutionMode.AUTONOMOUS: 1.0   # Allow all except absolutely dangerous
        }
    
    def should_block(self, step: Dict[str, Any], risk_score: float, mode: ExecutionMode) -> bool:
        """
        Determine if a step should be blocked based on safety rules
        
        Args:
            step: The step to evaluate (with command, description, etc.)
            risk_score: Risk score calculated by RiskScorer
            mode: Current execution mode
            
        Returns:
            True if the step should be blocked, False otherwise
        """
        command = step.get("command", "").lower()
        
        # Always block dangerous commands regardless of risk score or mode
        if self._is_absolutely_dangerous(command):
            return True
            
        # In assist mode, nothing executes
        if mode == ExecutionMode.ASSIST:
            return True
            
        # Check against mode-specific threshold
        threshold = self.mode_thresholds[mode]
        if risk_score > threshold:
            return True
            
        # For EXECUTE mode, also block medium-risk operations that require confirmation
        if mode == ExecutionMode.EXECUTE and self._is_medium_risk(command):
            return True  # In execute mode, medium-risk operations require explicit confirmation
            
        return False
    
    def should_request_confirmation(self, step: Dict[str, Any], risk_score: float, mode: ExecutionMode) -> bool:
        """
        Determine if a step should request user confirmation
        
        Args:
            step: The step to evaluate
            risk_score: Risk score calculated by RiskScorer
            mode: Current execution mode
            
        Returns:
            True if confirmation should be requested, False otherwise
        """
        command = step.get("command", "").lower()
        
        # Never request confirmation for absolutely dangerous commands (they're blocked outright)
        if self._is_absolutely_dangerous(command):
            return False
            
        # In autonomous mode, don't request confirmation (but still respect absolute blocks)
        if mode == ExecutionMode.AUTONOMOUS:
            return False
            
        # In assist mode, nothing executes anyway
        if mode == ExecutionMode.ASSIST:
            return False
            
        # Request confirmation for medium-risk operations in execute mode
        if mode == ExecutionMode.EXECUTE and self._is_medium_risk(command):
            return True
            
        # Request confirmation for high-risk operations in execute mode
        if mode == ExecutionMode.EXECUTE and 0.5 <= risk_score <= 0.7:
            return True
            
        return False
    
    def _is_absolutely_dangerous(self, command: str) -> bool:
        """Check if a command is absolutely dangerous and should always be blocked"""
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return True
        return False
    
    def _is_medium_risk(self, command: str) -> bool:
        """Check if a command is medium risk and may require confirmation"""
        for pattern in self.medium_risk_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return True
        return False
    
    def get_block_reason(self, step: Dict[str, Any], risk_score: float, mode: ExecutionMode) -> str:
        """
        Get the reason why a step was blocked
        
        Args:
            step: The step that was evaluated
            risk_score: Risk score calculated by RiskScorer
            mode: Current execution mode
            
        Returns:
            String explaining why the step was blocked
        """
        command = step.get("command", "").lower()
        
        # Check for absolutely dangerous commands
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    matched_part = match.group(0)
                    return f"Absolutely dangerous command pattern detected: {matched_part}"
        
        # Check for execution mode
        if mode == ExecutionMode.ASSIST:
            return "Execution mode is 'assist' - no commands are executed, only suggestions provided"
        
        # Check for risk threshold
        threshold = self.mode_thresholds[mode]
        if risk_score > threshold:
            return f"Risk score ({risk_score:.2f}) exceeds mode threshold ({threshold}) for {mode.value} mode"
        
        # Check for medium-risk operations in execute mode
        if mode == ExecutionMode.EXECUTE:
            for pattern in self.medium_risk_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    match = re.search(pattern, command, re.IGNORECASE)
                    if match:
                        matched_part = match.group(0)
                        return f"Medium-risk command pattern detected requiring confirmation: {matched_part}"
        
        return "Unknown reason for blocking"
    
    def get_suggested_alternative(self, step: Dict[str, Any]) -> str:
        """
        Provide a safer alternative to a blocked command
        
        Args:
            step: The step that was blocked
            
        Returns:
            String with a safer alternative command or suggestion
        """
        command = step.get("command", "").lower()
        original_command = step.get("command", "")
        
        # Suggest safer alternatives for common dangerous patterns
        if "rm -rf /" in command:
            return "Instead of deleting root '/', try specifying a specific directory like 'rm -rf temp/'"
        elif "chmod -R 777" in command:
            return f"Instead of 'chmod -R 777' (which gives everyone full access), try 'chmod -R 644' or 'chmod -R 755' for safer permissions"
        elif "kill -9 1" in command:
            return "Never kill PID 1 (init process) as it can crash the system. Use 'kill -9 <specific_process_id>' instead"
        elif re.search(r"rm\s+-rf\s+\w+", command):
            # Suggest using trash command instead of rm if available
            cmd_parts = original_command.split()
            if len(cmd_parts) >= 3:
                target = " ".join(cmd_parts[2:])  # Get the target after 'rm -rf'
                return f"Consider using a safer alternative like 'trash {target}' instead of 'rm -rf {target}', or verify the target is correct"
        
        return "No specific safer alternative available. Please review the command carefully before proceeding."
    
    def validate_context_for_execution(self, context: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate that the current context is appropriate for the execution
        
        Args:
            context: Current context information
            step: The step to be executed
            
        Returns:
            Dictionary with validation results (validity and warnings)
        """
        results = {
            "valid": True,
            "warnings": [],
            "critical_issues": []
        }
        
        command = step.get("command", "").lower()
        
        # Check if we're in a critical directory
        current_dir = context.get("current_directory", "").lower()
        if any(critical_dir in current_dir for critical_dir in ["/", "/etc", "/boot", "/sys", "/proc", "c:\\windows"]):
            if any(dangerous_cmd in command for dangerous_cmd in ["rm", "del", "format", "chmod", "chown"]):
                results["critical_issues"].append(
                    f"Dangerous command '{command.split()[0] if command.split() else 'unknown'}' "
                    f"in critical system directory: {context.get('current_directory', 'unknown')}"
                )
                results["valid"] = False
        
        # Check git status - warn if in a dirty repo
        git_status = context.get("git_status", {})
        if git_status.get("is_git_repo", False) and git_status.get("has_changes", False):
            if any(file_op in command for file_op in ["rm", "mv", "cp", "chmod", "chown"]):
                results["warnings"].append(
                    "Detected uncommitted changes in git repository. "
                    "Consider committing or stashing changes before executing file operations."
                )
        
        # Check system resources - warn if system is under stress
        system_resources = context.get("system_resources", {})
        cpu_percent = system_resources.get("cpu_percent", 0)
        memory_percent = system_resources.get("memory_percent", 0)
        
        if cpu_percent > 90 or memory_percent > 90:
            if any(resource_intensive in command for resource_intensive in ["build", "compile", "test", "run"]):
                results["warnings"].append(
                    f"System resources are highly utilized (CPU: {cpu_percent}%, Memory: {memory_percent}%). "
                    "Executing resource-intensive commands may cause system instability."
                )
        
        return results
    
    def get_execution_policy_summary(self, mode: ExecutionMode) -> str:
        """
        Get a summary of the execution policy for a given mode
        
        Args:
            mode: The execution mode to summarize
            
        Returns:
            String describing the execution policy
        """
        if mode == ExecutionMode.ASSIST:
            return (
                "ASSIST MODE: No commands are executed. ByteBot only provides suggestions "
                "and recommendations. Ideal for reviewing and planning operations."
            )
        elif mode == ExecutionMode.EXECUTE:
            return (
                "EXECUTE MODE: Safe commands execute automatically. Medium-risk operations "
                "require confirmation. High-risk operations (>0.7) are blocked. "
                "Ideal for routine development tasks."
            )
        elif mode == ExecutionMode.AUTONOMOUS:
            return (
                "AUTONOMOUS MODE: All non-dangerous commands execute automatically. "
                "Absolutely dangerous commands (like 'rm -rf /') are still blocked. "
                "Use with caution for automated workflows."
            )
        else:
            return "UNKNOWN MODE: Execution policy not defined for this mode."


# Example usage and testing
if __name__ == "__main__":
    safety_gate = SafetyGate()
    
    # Test steps
    test_steps = [
        {"id": "1", "command": "ls -la", "description": "List directory"},
        {"id": "2", "command": "rm -rf /", "description": "Delete everything"},
        {"id": "3", "command": "git status", "description": "Check git status"},
        {"id": "4", "command": "chmod -R 777 /etc", "description": "Make system files world writable"},
        {"id": "5", "command": "kill -9 1234", "description": "Kill process"},
    ]
    
    print("Safety Gate Evaluation Results:")
    print("=" * 50)
    
    for step in test_steps:
        for mode in ExecutionMode:
            should_block = safety_gate.should_block(step, 0.8, mode)  # Using 0.8 as example risk
            needs_confirmation = safety_gate.should_request_confirmation(step, 0.6, mode)  # Using 0.6 as example risk
            
            print(f"\nStep: {step['command']}")
            print(f"Mode: {mode.value}")
            print(f"Should Block: {should_block}")
            print(f"Needs Confirmation: {needs_confirmation}")
            
            if should_block:
                reason = safety_gate.get_block_reason(step, 0.8, mode)
                print(f"Block Reason: {reason}")
            
            print("-" * 30)