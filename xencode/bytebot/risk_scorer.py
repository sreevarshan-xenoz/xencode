"""
Risk Scorer - Evaluates command risk based on multiple factors

This component evaluates the risk level of commands based on context,
command patterns, system state, and other factors.
"""

import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import os
import subprocess


@dataclass
class RiskAssessment:
    """Complete risk assessment result"""
    score: float  # 0.0 to 1.0
    breakdown: Dict[str, float]  # Individual risk factor scores
    reason: str  # Explanation of the risk assessment
    timestamp: datetime
    is_dangerous: bool  # Whether the command is considered dangerous


class RiskScorer:
    """
    Risk scoring system that evaluates command risk based on context
    """
    
    def __init__(self):
        # Risk factors and their maximum possible scores
        self.risk_factors = {
            # Very high risk operations
            "system_destruction": 0.30,  # rm -rf /, format, etc.
            "privilege_escalation": 0.25,  # sudo, su, runas
            "critical_system_files": 0.20,  # Modifying system files
            
            # High risk operations  
            "system_wide_scope": 0.15,  # Operations affecting entire system
            "irreversible_operation": 0.15,  # Operations that can't be undone
            
            # Medium risk operations
            "file_modification": 0.10,  # Creating, modifying, deleting files
            "network_activity": 0.08,  # Network connections, downloads
            "process_manipulation": 0.07,  # Starting, stopping, killing processes
            
            # Lower risk operations
            "resource_consumption": 0.05,  # High CPU/memory usage
            "environment_change": 0.03,  # Changing environment variables
        }
        
        # Dangerous command patterns that should be heavily penalized
        self.dangerous_patterns = [
            # System destruction
            (r"rm\s+-rf\s+/", "Deleting root directory"),
            (r"format\s+", "Disk formatting"),
            (r"del\s+/f\s+c:\\", "Force deleting system drive"),
            (r"mkfs\.", "File system creation (destructive)"),
            (r"dd\s+if=", "Direct disk access (potentially destructive)"),
            (r"shred\s+", "Secure file deletion"),
            (r"cat\s+/dev/zero\s+>\s+/dev/sda", "Writing zeros to disk"),
            (r"echo\s+>?\s*/proc/sysrq-trigger", "SysRq operations"),
            (r"sync\s+&&\s+echo\s+1\s+>\s+/proc/sys/kernel/sysrq\s+&&\s+echo\s+b\s+>\s+/proc/sysrq-trigger", "Forced reboot"),
            
            # Privilege escalation
            (r"sudo\s+(rm|format|mkfs|dd|shred|del)", "Dangerous operation with elevated privileges"),
            (r"su\s+", "Switching user (potential privilege escalation)"),
            (r"runas\s+", "Running as different user"),
            
            # Critical system files
            (r"/etc/", "Modifying system configuration"),
            (r"/boot/", "Modifying boot files"),
            (r"/sys/", "Modifying system interface"),
            (r"/proc/", "Modifying process information"),
            (r"C:\\Windows\\", "Modifying Windows system files"),
            (r"C:\\ProgramData\\", "Modifying program data"),
        ]
        
        # High risk patterns
        self.high_risk_patterns = [
            # System wide operations
            (r"rm\s+-rf\s+", "Recursive deletion"),
            (r"chmod\s+-R\s+777", "Making all files world writable"),
            (r"chown\s+-R", "Changing ownership recursively"),
            (r"mv\s+/.*\s+/", "Moving files to root"),
            (r"cp\s+/dev/zero", "Copying from zero device"),
            
            # Irreversible operations
            (r"kill\s+-9\s+", "Force killing processes"),
            (r"pkill\s+-f", "Force killing by pattern"),
            (r"taskkill\s+/f", "Force killing tasks"),
            (r"truncate\s+--size\s+0", "Truncating files"),
            
            # Network activity
            (r"curl\s+.*\|.*sh", "Piping curl to shell (potential security risk)"),
            (r"wget\s+.*\|.*sh", "Piping wget to shell (potential security risk)"),
            (r"powershell\s+.*IEX", "Executing PowerShell expressions"),
        ]
        
        # Medium risk patterns
        self.medium_risk_patterns = [
            # File modifications
            (r"rm\s+", "File deletion"),
            (r"del\s+", "File deletion (Windows)"),
            (r"rmdir\s+", "Directory deletion"),
            (r"rd\s+", "Directory deletion (Windows)"),
            
            # Process manipulation
            (r"kill\s+", "Terminating processes"),
            (r"pkill\s+", "Terminating processes by pattern"),
            (r"taskkill\s+", "Terminating tasks (Windows)"),
            
            # Environment changes
            (r"export\s+PATH=", "Modifying PATH environment variable"),
            (r"set\s+PATH=", "Modifying PATH environment variable (Windows)"),
        ]
    
    def score_command(self, command: str, context: Dict[str, Any] = None) -> RiskAssessment:
        """
        Score a command for risk based on multiple factors
        
        Args:
            command: The command to evaluate
            context: Context information that might affect risk assessment
            
        Returns:
            RiskAssessment with score and breakdown
        """
        if not command:
            return RiskAssessment(
                score=0.0,
                breakdown={},
                reason="Empty command has no risk",
                timestamp=datetime.now(),
                is_dangerous=False
            )
        
        command_lower = command.lower().strip()
        breakdown = {}
        total_score = 0.0
        
        # Check for dangerous patterns (highest priority)
        dangerous_matches = []
        for pattern, description in self.dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                dangerous_matches.append(description)
                # Add maximum penalty for dangerous patterns
                breakdown["dangerous_patterns"] = self.risk_factors["system_destruction"]
                total_score += self.risk_factors["system_destruction"]
        
        # Check for high risk patterns
        high_risk_matches = []
        for pattern, description in self.high_risk_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                high_risk_matches.append(description)
                # Add high risk penalty
                if "high_risk_patterns" not in breakdown:
                    breakdown["high_risk_patterns"] = 0.0
                breakdown["high_risk_patterns"] += self.risk_factors["irreversible_operation"]
                total_score += self.risk_factors["irreversible_operation"]
        
        # Check for medium risk patterns
        medium_risk_matches = []
        for pattern, description in self.medium_risk_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                medium_risk_matches.append(description)
                # Add medium risk penalty
                if "medium_risk_patterns" not in breakdown:
                    breakdown["medium_risk_patterns"] = 0.0
                breakdown["medium_risk_patterns"] += self.risk_factors["file_modification"]
                total_score += self.risk_factors["file_modification"]
        
        # Apply context-based risk adjustments
        if context:
            context_risk = self._calculate_context_risk(command, context)
            if context_risk > 0:
                breakdown["context_risk"] = context_risk
                total_score += context_risk
        
        # Apply privilege escalation risk
        if self._has_privilege_escalation(command):
            priv_risk = self.risk_factors["privilege_escalation"]
            breakdown["privilege_escalation"] = priv_risk
            total_score += priv_risk
        
        # Apply system-wide scope risk
        if self._has_system_wide_scope(command):
            scope_risk = self.risk_factors["system_wide_scope"]
            breakdown["system_wide_scope"] = scope_risk
            total_score += scope_risk
        
        # Apply irreversible operation risk
        if self._is_irreversible_operation(command):
            irr_risk = self.risk_factors["irreversible_operation"]
            breakdown["irreversible_operation"] = irr_risk
            total_score += irr_risk
        
        # Apply file modification risk
        if self._modifies_files(command):
            file_risk = self.risk_factors["file_modification"]
            breakdown["file_modification"] = file_risk
            total_score += file_risk
        
        # Apply network activity risk
        if self._has_network_activity(command):
            net_risk = self.risk_factors["network_activity"]
            breakdown["network_activity"] = net_risk
            total_score += net_risk
        
        # Apply process manipulation risk
        if self._manipulates_processes(command):
            proc_risk = self.risk_factors["process_manipulation"]
            breakdown["process_manipulation"] = proc_risk
            total_score += proc_risk
        
        # Cap the score at 1.0
        total_score = min(1.0, total_score)
        
        # Generate reason based on findings
        reason_parts = []
        if dangerous_matches:
            reason_parts.append(f"Dangerous patterns detected: {', '.join(dangerous_matches[:3])}")
        if high_risk_matches:
            reason_parts.append(f"High risk patterns: {', '.join(high_risk_matches[:3])}")
        if medium_risk_matches:
            reason_parts.append(f"Medium risk patterns: {', '.join(medium_risk_matches[:3])}")
        
        if not reason_parts:
            reason_parts.append("Command appears to be low risk")
        
        reason = "; ".join(reason_parts)
        
        # Determine if command is dangerous (threshold > 0.7)
        is_dangerous = total_score > 0.7
        
        return RiskAssessment(
            score=total_score,
            breakdown=breakdown,
            reason=reason,
            timestamp=datetime.now(),
            is_dangerous=is_dangerous
        )
    
    def _calculate_context_risk(self, command: str, context: Dict[str, Any]) -> float:
        """Calculate risk based on current context"""
        risk = 0.0
        
        # Higher risk if working in git repo with uncommitted changes
        git_status = context.get("git_status", {})
        if git_status.get("is_git_repo", False) and git_status.get("has_changes", False):
            # Operations in a dirty repo are slightly higher risk
            risk += 0.05
        
        # Higher risk if system resources are already stressed
        system_resources = context.get("system_resources", {})
        cpu_percent = system_resources.get("cpu_percent", 0)
        memory_percent = system_resources.get("memory_percent", 0)
        
        if cpu_percent > 80 or memory_percent > 80:
            # Operations when system is stressed are higher risk
            risk += 0.05
        
        # Higher risk if working in important directories
        current_dir = context.get("current_directory", "").lower()
        if any(important_dir in current_dir for important_dir in [
            "prod", "production", "live", "www", "var/www", "htdocs"
        ]):
            risk += 0.10
        
        return risk
    
    def _has_privilege_escalation(self, command: str) -> bool:
        """Check if command involves privilege escalation"""
        return bool(re.search(r'\bsudo\b|\bsu\b|\brunas\b|\badmin\b', command, re.IGNORECASE))
    
    def _has_system_wide_scope(self, command: str) -> bool:
        """Check if command affects system-wide resources"""
        return bool(re.search(
            r'/\*|/\.\.|all|every|\*/|\*\.|\s+\-\s+|everything|system32|windows',
            command, re.IGNORECASE
        ))
    
    def _is_irreversible_operation(self, command: str) -> bool:
        """Check if command performs irreversible operations"""
        return bool(re.search(
            r'\b(rm|del|format|mkfs|shred|truncate|dd|cat\s+/dev/zero)\b',
            command, re.IGNORECASE
        ))
    
    def _modifies_files(self, command: str) -> bool:
        """Check if command modifies files"""
        return bool(re.search(
            r'\b(rm|del|rmdir|mv|cp|touch|chmod|chown|cat\s+>|echo\s+>)\b',
            command, re.IGNORECASE
        ))
    
    def _has_network_activity(self, command: str) -> bool:
        """Check if command involves network activity"""
        return bool(re.search(
            r'\b(curl|wget|ssh|scp|ftp|netstat|ping|nslookup|dig|nmap)\b',
            command, re.IGNORECASE
        ))
    
    def _manipulates_processes(self, command: str) -> bool:
        """Check if command manipulates processes"""
        return bool(re.search(
            r'\b(kill|pkill|killall|taskkill|jobs|ps|top|htop)\b',
            command, re.IGNORECASE
        ))
    
    def get_risk_category(self, score: float) -> str:
        """Get risk category based on score"""
        if score >= 0.7:
            return "Very High"
        elif score >= 0.5:
            return "High"
        elif score >= 0.3:
            return "Medium"
        elif score >= 0.1:
            return "Low"
        else:
            return "Minimal"
    
    def get_recommendation(self, assessment: RiskAssessment) -> str:
        """Get recommendation based on risk assessment"""
        if assessment.is_dangerous:
            return "BLOCK - This command is dangerous and should not be executed"
        elif assessment.score >= 0.5:
            return "CONFIRM - High risk command, please confirm before executing"
        elif assessment.score >= 0.3:
            return "CAUTION - Medium risk command, review carefully"
        elif assessment.score > 0:
            return "PROCEED - Low risk command, safe to execute"
        else:
            return "PROCEED - No significant risk detected"
    
    def assess_multiple_commands(self, commands: List[str], context: Dict[str, Any] = None) -> List[RiskAssessment]:
        """Assess risk for multiple commands"""
        return [self.score_command(cmd, context) for cmd in commands]


# Example usage and testing
if __name__ == "__main__":
    scorer = RiskScorer()
    
    # Test commands
    test_commands = [
        "ls -la",
        "echo 'hello world'",
        "rm -rf /important/data",
        "sudo rm -rf /",
        "curl https://example.com/install.sh | bash",
        "git status",
        "kill -9 1234",
        "chmod -R 777 /",
        "mkdir new_folder"
    ]
    
    print("Risk Assessment Results:")
    print("=" * 50)
    
    for cmd in test_commands:
        assessment = scorer.score_command(cmd)
        category = scorer.get_risk_category(assessment.score)
        recommendation = scorer.get_recommendation(assessment)
        
        print(f"\nCommand: {cmd}")
        print(f"Score: {assessment.score:.2f} ({category})")
        print(f"Recommendation: {recommendation}")
        print(f"Reason: {assessment.reason}")
        print(f"Breakdown: {assessment.breakdown}")
        print("-" * 30)