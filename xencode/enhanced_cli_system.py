#!/usr/bin/env python3
"""
Enhanced CLI System for Xencode Phase 2 Integration

Main integration point that extends xencode with Phase 1 features while
maintaining backward compatibility and graceful fallback.

Requirements: 6.1, 6.6, 7.1, 7.2, 7.3
"""

import argparse
import os
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Import Phase 1 systems with graceful fallback
try:
    from .multi_model_system import MultiModelManager

    MULTI_MODEL_AVAILABLE = True
except ImportError:
    MULTI_MODEL_AVAILABLE = False
    MultiModelManager = None

try:
    from .smart_context_system import SmartContextManager

    SMART_CONTEXT_AVAILABLE = True
except ImportError:
    SMART_CONTEXT_AVAILABLE = False
    SmartContextManager = None

try:
    from .code_analysis_system import CodeAnalyzer

    CODE_ANALYSIS_AVAILABLE = True
except ImportError:
    CODE_ANALYSIS_AVAILABLE = False
    CodeAnalyzer = None

# Import security and resource management
from .context_cache_manager import ContextCacheManager
from .model_stability_manager import ModelStabilityManager
from .security_manager import SecurityManager


@dataclass
class FeatureAvailability:
    """Tracks availability of Phase 1 features"""

    multi_model: bool = False
    smart_context: bool = False
    code_analysis: bool = False
    security_manager: bool = True
    context_cache: bool = True
    model_stability: bool = True

    @property
    def enhanced_features_available(self) -> bool:
        """Check if any enhanced features are available"""
        return self.multi_model or self.smart_context or self.code_analysis

    @property
    def feature_level(self) -> str:
        """Get feature level description"""
        if self.multi_model and self.smart_context and self.code_analysis:
            return "advanced"
        elif self.enhanced_features_available:
            return "partial"
        else:
            return "basic"


class FeatureDetector:
    """
    Safely validates Phase 1 system availability with timeout handling
    """

    def __init__(self, timeout_seconds: float = 2.0):
        self.timeout_seconds = timeout_seconds
        self.detection_cache = {}
        self.last_detection_time = 0
        self.cache_ttl_seconds = 30  # Cache results for 30 seconds

    def detect_features(self, force_refresh: bool = False) -> FeatureAvailability:
        """
        Detect available features with caching and timeout protection

        Args:
            force_refresh: Force re-detection ignoring cache

        Returns:
            FeatureAvailability object with detection results
        """
        current_time = time.time()

        # Use cache if available and not expired
        if (
            not force_refresh
            and self.detection_cache
            and current_time - self.last_detection_time < self.cache_ttl_seconds
        ):
            return self.detection_cache

        availability = FeatureAvailability()

        # Test each feature with timeout protection
        availability.multi_model = self._test_feature_with_timeout(
            self._test_multi_model, "multi_model"
        )
        availability.smart_context = self._test_feature_with_timeout(
            self._test_smart_context, "smart_context"
        )
        availability.code_analysis = self._test_feature_with_timeout(
            self._test_code_analysis, "code_analysis"
        )

        # Core features should always be available
        availability.security_manager = True
        availability.context_cache = True
        availability.model_stability = True

        # Cache results
        self.detection_cache = availability
        self.last_detection_time = current_time

        return availability

    def _test_feature_with_timeout(self, test_func, feature_name: str) -> bool:
        """Test feature availability with timeout protection"""
        result = [False]  # Use list for mutable reference

        def test_thread():
            try:
                result[0] = test_func()
            except Exception:
                result[0] = False

        thread = threading.Thread(target=test_thread, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            # Timeout occurred
            return False

        return result[0]

    def _test_multi_model(self) -> bool:
        """Test MultiModelManager availability"""
        if not MULTI_MODEL_AVAILABLE or MultiModelManager is None:
            return False

        try:
            manager = MultiModelManager()
            # Quick test - just check if it initializes
            return hasattr(manager, 'analyze_query_type')
        except Exception:
            return False

    def _test_smart_context(self) -> bool:
        """Test SmartContextManager availability"""
        if not SMART_CONTEXT_AVAILABLE or SmartContextManager is None:
            return False

        try:
            manager = SmartContextManager()
            # Quick test - just check if it initializes
            return hasattr(manager, 'scan_project_context')
        except Exception:
            return False

    def _test_code_analysis(self) -> bool:
        """Test CodeAnalyzer availability"""
        if not CODE_ANALYSIS_AVAILABLE or CodeAnalyzer is None:
            return False

        try:
            analyzer = CodeAnalyzer()
            # Quick test - just check if it initializes
            return hasattr(analyzer, 'analyze_code_quality')
        except Exception:
            return False


class CommandRouter:
    """
    Routes commands to appropriate handlers (enhanced vs legacy)
    """

    def __init__(self, feature_availability: FeatureAvailability):
        self.features = feature_availability

        # Enhanced command mappings
        self.enhanced_commands = {
            'analyze': self._requires_code_analysis,
            'models': self._requires_multi_model,
            'context': self._requires_smart_context,
            'smart': self._requires_multi_model_and_context,
            'git-commit': self._requires_code_analysis,
            'git-review': self._requires_code_analysis,
            'git-diff-analyze': self._requires_code_analysis,
            'git-branch': self._requires_code_analysis,
        }

    def can_handle_enhanced_command(self, command: str) -> bool:
        """Check if enhanced command can be handled"""
        if command not in self.enhanced_commands:
            return False

        return self.enhanced_commands[command]()

    def _requires_code_analysis(self) -> bool:
        return self.features.code_analysis

    def _requires_multi_model(self) -> bool:
        return self.features.multi_model

    def _requires_smart_context(self) -> bool:
        return self.features.smart_context

    def _requires_multi_model_and_context(self) -> bool:
        return self.features.multi_model and self.features.smart_context


class EnhancedXencodeCLI:
    """
    Main enhanced CLI system with graceful fallback detection
    """

    def __init__(self):
        # Initialize with cold start optimization
        self._initialize_with_cold_start()

    def _initialize_with_cold_start(self):
        """Initialize system with progressive warm-up and timeout handling"""
        print("🚀 Initializing Xencode Enhanced System...")

        # Phase 1: Core components (immediate)
        start_time = time.time()

        self.security_manager = SecurityManager()
        self.context_cache = ContextCacheManager()
        self.model_stability = ModelStabilityManager()

        core_time = time.time() - start_time
        print(f"✅ Core systems ready ({core_time:.2f}s)")

        # Phase 2: Feature detection (with timeout)
        detection_start = time.time()
        self.feature_detector = FeatureDetector(timeout_seconds=2.0)
        self.features = self.feature_detector.detect_features()

        detection_time = time.time() - detection_start
        print(f"🔍 Feature detection complete ({detection_time:.2f}s)")

        # Phase 3: Initialize available enhanced features (background)
        self._initialize_enhanced_features_background()

        # Phase 4: Show feature dashboard
        self._show_feature_dashboard()

    def _initialize_enhanced_features_background(self):
        """Initialize enhanced features in background thread"""

        def init_background():
            try:
                # Initialize Multi-Model System
                if self.features.multi_model:
                    self.multi_model = MultiModelManager()
                else:
                    self.multi_model = None

                # Initialize Smart Context System
                if self.features.smart_context:
                    self.smart_context = SmartContextManager()
                else:
                    self.smart_context = None

                # Initialize Code Analysis System
                if self.features.code_analysis:
                    self.code_analyzer = CodeAnalyzer()
                else:
                    self.code_analyzer = None

                print("🎯 Enhanced features initialized")

            except Exception as e:
                print(f"⚠️ Warning: Enhanced feature initialization failed: {e}")

        # Start background initialization
        init_thread = threading.Thread(target=init_background, daemon=True)
        init_thread.start()

        # Store thread reference for potential joining
        self._init_thread = init_thread

    def _show_feature_dashboard(self):
        """Show feature readiness dashboard"""
        print("\n📊 Feature Dashboard:")
        print("=" * 40)

        # Core features (always available)
        print("✅ Core Features:")
        print("  • Security Manager")
        print("  • Context Cache")
        print("  • Model Stability")

        # Enhanced features
        if self.features.enhanced_features_available:
            print("\n🚀 Enhanced Features:")
            if self.features.multi_model:
                print("  • Multi-Model System")
            if self.features.smart_context:
                print("  • Smart Context System")
            if self.features.code_analysis:
                print("  • Code Analysis System")
        else:
            print("\n⚠️ Enhanced Features: Not Available")
            print("  Install Phase 1 systems for advanced features")

        print(f"\n🎚️ Feature Level: {self.features.feature_level.upper()}")
        print("=" * 40)

    def create_parser(self) -> argparse.ArgumentParser:
        """Create enhanced argument parser with all commands"""
        parser = argparse.ArgumentParser(
            description="Xencode Enhanced - AI-powered development assistant",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  xencode "How do I implement authentication?"
  xencode --analyze src/
  xencode --models
  xencode --context
  xencode --smart "Optimize this database query"
  xencode --git-commit
            """,
        )

        # Legacy positional argument (for backward compatibility)
        parser.add_argument(
            'query', nargs='?', help='Query to send to AI model (legacy mode)'
        )

        # Enhanced commands
        parser.add_argument(
            '--analyze',
            metavar='PATH',
            help='Analyze code quality and issues in specified path',
        )

        parser.add_argument(
            '--models',
            action='store_true',
            help='Display available models with capabilities',
        )

        parser.add_argument(
            '--context',
            action='store_true',
            help='Show current project context information',
        )

        parser.add_argument(
            '--smart',
            metavar='QUERY',
            help='Execute query with automatic model selection',
        )

        parser.add_argument(
            '--git-commit',
            action='store_true',
            help='Generate intelligent commit message from git diff',
        )

        parser.add_argument(
            '--git-review',
            metavar='REF',
            nargs='?',
            const='HEAD',
            help='Review git changes (defaults to HEAD, or provide PR/Branch/Commit)',
        )

        parser.add_argument(
            '--git-diff-analyze',
            action='store_true',
            help='Analyze current git diff for bugs and style issues before committing',
        )

        parser.add_argument(
            '--git-branch',
            metavar='ACTION',
            choices=['suggest'],
            help='Git branch assistance (e.g., "suggest" to name branch from changes)',
        )

        # System commands
        parser.add_argument(
            '--feature-status',
            action='store_true',
            help='Show detailed feature availability status',
        )

        parser.add_argument(
            '--refresh-features',
            action='store_true',
            help='Refresh feature detection cache',
        )

        return parser

    def process_enhanced_args(self, args) -> Optional[str]:
        """
        Process enhanced arguments and return result or None for legacy handling

        Returns:
            String result if enhanced command processed, None for legacy handling
        """
        # Feature status command (always available)
        if args.feature_status:
            return self._handle_feature_status()

        # Refresh features command (always available)
        if args.refresh_features:
            return self._handle_refresh_features()

        # Enhanced commands (require feature availability)
        if args.analyze:
            return self.handle_analyze_command(args.analyze)

        if args.models:
            return self.handle_models_command()

        if args.context:
            return self.handle_context_command()

        if args.smart:
            return self.handle_smart_query(args.smart)

        if args.git_commit:
            return self.handle_git_commit_command()

        if args.git_review:
            return self.handle_git_review_command(args.git_review)

        if args.git_diff_analyze:
            return self.handle_git_diff_analyze_command()

        if args.git_branch:
            return self.handle_git_branch_command(args.git_branch)

        # No enhanced command found - use legacy handling
        return None

    def handle_analyze_command(self, path: str) -> str:
        """Handle --analyze command"""
        if not self.features.code_analysis:
            return (
                "❌ Code analysis not available\n"
                "Install Phase 1 Code Analysis System for this feature"
            )

        try:
            # Wait for background initialization if needed
            if hasattr(self, '_init_thread'):
                self._init_thread.join(timeout=5)

            if self.code_analyzer is None:
                return "❌ Code analyzer failed to initialize"

            # Validate path with security manager
            if not self.security_manager.validate_project_path(path):
                return f"❌ Invalid or unsafe path: {path}"

            # Perform analysis
            print(f"🔍 Analyzing code in: {path}")
            results = self.code_analyzer.analyze_code_quality(path)

            return f"✅ Code analysis complete\n{results}"

        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"

    def handle_models_command(self) -> str:
        """Handle --models command"""
        if not self.features.multi_model:
            return (
                "❌ Multi-model system not available\n"
                "Install Phase 1 Multi-Model System for enhanced model management"
            )

        try:
            # Wait for background initialization if needed
            if hasattr(self, '_init_thread'):
                self._init_thread.join(timeout=5)

            if self.multi_model is None:
                return "❌ Multi-model system failed to initialize"

            # Get model information
            models_info = self.multi_model.get_available_models_with_capabilities()

            return f"📦 Available Models:\n{models_info}"

        except Exception as e:
            return f"❌ Failed to get models: {str(e)}"

    def handle_context_command(self) -> str:
        """Handle --context command"""
        if not self.features.smart_context:
            return (
                "❌ Smart context system not available\n"
                "Install Phase 1 Smart Context System for project awareness"
            )

        try:
            # Wait for background initialization if needed
            if hasattr(self, '_init_thread'):
                self._init_thread.join(timeout=5)

            if self.smart_context is None:
                return "❌ Smart context system failed to initialize"

            # Get current context
            current_dir = os.getcwd()
            context_info = self.smart_context.get_project_context_summary(current_dir)

            return f"📁 Project Context:\n{context_info}"

        except Exception as e:
            return f"❌ Failed to get context: {str(e)}"

    def handle_smart_query(self, query: str) -> str:
        """Handle --smart command with automatic model selection"""
        if not (self.features.multi_model and self.features.smart_context):
            return (
                "❌ Smart query requires both Multi-Model and Smart Context systems\n"
                "Install Phase 1 systems for this feature"
            )

        try:
            # Wait for background initialization if needed
            if hasattr(self, '_init_thread'):
                self._init_thread.join(timeout=5)

            if self.multi_model is None or self.smart_context is None:
                return "❌ Smart query systems failed to initialize"

            # Analyze query and select best model
            query_analysis = self.multi_model.analyze_query_type(query)
            suggested_model = query_analysis.suggested_model

            # Test model stability using validated stability manager
            stability_result = self.model_stability.test_model_stability(
                suggested_model
            )

            if not stability_result.is_stable:
                # Use fallback chain
                fallback_chain = self.model_stability.get_fallback_chain(
                    query_analysis.query_type
                )
                for fallback_model in fallback_chain:
                    fallback_result = self.model_stability.test_model_stability(
                        fallback_model
                    )
                    if fallback_result.is_stable:
                        suggested_model = fallback_model
                        break
                else:
                    # Use emergency model
                    suggested_model = self.model_stability.get_emergency_model()

            # Get project context if available
            current_dir = os.getcwd()
            context_info = ""

            try:
                # Validate project path with security manager
                if self.security_manager.validate_project_path(current_dir):
                    context_data = self.smart_context.get_project_context_summary(
                        current_dir
                    )
                    if context_data:
                        context_info = f"\n\n📁 Project Context:\n{context_data}"
                else:
                    context_info = "\n\n⚠️ Project path validation failed - using query without context"
            except Exception:
                # Context retrieval failed - continue without context
                pass

            # Build enhanced query with context
            enhanced_query = query
            if context_info:
                enhanced_query = f"{query}{context_info}"

            reasoning = (
                f"🤖 Using {suggested_model} for {query_analysis.query_type} query"
            )

            # Import xencode_core functions for query execution
            try:
                from rich.console import Console
                from rich.markdown import Markdown

                from xencode_core import extract_thinking_and_answer, run_query

                console = Console()

                # Execute the query with selected model
                print(f"{reasoning}")
                print(f"🔍 Query: {query}")

                response = run_query(suggested_model, enhanced_query)

                # Format and display response
                thinking, answer = extract_thinking_and_answer(response)
                if answer.strip():
                    console.print(Markdown(answer.strip()))
                else:
                    # Fallback to full response if no thinking tags
                    console.print(Markdown(response.strip()))

                return "✅ Smart query completed successfully"

            except ImportError:
                return f"✅ Smart query analysis complete\n{reasoning}\n\nExecute manually: xencode -m {suggested_model} \"{query}\""

        except Exception as e:
            return f"❌ Smart query failed: {str(e)}"

    def handle_git_review_command(self, ref: str) -> str:
        """
        Handle --git-review command to review code changes using AI and static analysis
        """
        if not self.features.code_analysis:
            return "❌ Git review requires Code Analysis System"

        try:
            # Wait for background initialization if needed
            if hasattr(self, '_init_thread'):
                self._init_thread.join(timeout=5)

            if self.code_analyzer is None:
                return "❌ Code analyzer failed to initialize"

            print(f"🔍 Reviewing changes against {ref}...")
            
            # Get diff content
            diff_content = self.code_analyzer.get_diff_from_ref(ref)
            
            if not diff_content:
                return "ℹ️ No changes found to review."

            # Static Analysis on changes
            diff_issues = self.code_analyzer.analyze_diff_context(diff_content)
            
            # Get general stats
            analysis = self.code_analyzer.analyze_git_diff(diff_content)
            
            # Construct a human-readable review
            review = ["📋 Git Review Report", "=" * 30]
            
            review.append(f"\n📊 Scope: {analysis['scope']}")
            review.append(f"📁 Files Changed: {len(analysis['files_changed'])}")
            review.append(f"➕ Additions: {analysis['additions']} | ➖ Deletions: {analysis['deletions']}")
            
            if analysis['languages']:
                review.append(f"💻 Languages: {', '.join(analysis['languages'])}")
                
            review.append("\n⚠️ Static Analysis Issues (in changed lines):")
            if diff_issues:
                for issue in diff_issues:
                     severity_emoji = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🟢",
                    }
                     emoji = severity_emoji.get(issue.severity.value, "⚪")
                     review.append(f"  {emoji} {Path(issue.file_path).name}:{issue.line_number} - {issue.message}")
            else:
                review.append("  ✅ No static analysis issues found in changes.")
            
            # AI Review if available
            if self.features.multi_model and self.multi_model:
                try:
                    from xencode_core import run_query, extract_thinking_and_answer
                    
                    print("🤖 requesting AI Code Review...")
                    
                    # Construct prompt
                    # Truncate diff to avoid context limits (conservative 6000 chars)
                    truncated_diff = diff_content[:6000]
                    if len(diff_content) > 6000:
                        truncated_diff += "\n... (diff truncated)"
                        
                    prompt = (
                        "You are an expert code reviewer. Review the following git diff for bugs, "
                        "security vulnerabilities, and code style issues. "
                        "Focus ONLY on the changes. Be concise and constructive.\n\n"
                        f"```diff\n{truncated_diff}\n```"
                    )
                    
                    # Use a coding capable model
                    model = "codellama:7b" 
                    # Ideally we check available models, but this is a safe default for now 
                    # given the roadmap mentions it.
                    
                    response = run_query(model, prompt)
                    _, answer = extract_thinking_and_answer(response)
                    
                    if answer.strip():
                        review.append(f"\n🧠 AI Review:\n{answer.strip()}")
                        
                except Exception as e:
                    review.append(f"\n⚠️ AI Review unavailable: {str(e)}")
            else:
                review.append("\nℹ️ For AI review, enable Multi-Model System (Phase 1)")

            return "\n".join(review)

        except Exception as e:
            return f"❌ Git review failed: {str(e)}"

    def handle_git_diff_analyze_command(self) -> str:
        """
        Handle --git-diff-analyze to check for bugs/style in current diff
        """
        if not self.features.code_analysis:
            return "❌ Diff analysis requires Code Analysis System"
            
        try:
             # Wait for background initialization if needed
            if hasattr(self, '_init_thread'):
                self._init_thread.join(timeout=5)

            if self.code_analyzer is None:
                return "❌ Code analyzer failed to initialize"

            print("🔍 Analyzing current diff...")
            diff_content = self.code_analyzer.get_raw_git_diff(staged=False)
            staged_diff = self.code_analyzer.get_raw_git_diff(staged=True)
            
            if not diff_content and not staged_diff:
                return "✅ No changes to analyze."
            
            full_diff = (staged_diff or "") + "\n" + (diff_content or "")
            
            # Use robust analysis on changed lines
            issues = self.code_analyzer.analyze_diff_context(full_diff)
            
            # Also do regex checks for things AST might miss (like TODOs in comments not docstrings)
            regex_issues = []
            for line in full_diff.splitlines():
                if line.startswith('+'):
                    clean_line = line[1:].strip()
                    if 'TODO' in clean_line:
                         regex_issues.append(f"📝 TODO found: {clean_line}")

            if not issues and not regex_issues:
                return "✅ Diff looks clean! (No static analysis issues found)"
            
            report = ["⚠️ Issues found in diff:", "=" * 25]
            
            for issue in issues:
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                }
                emoji = severity_emoji.get(issue.severity.value, "⚪")
                report.append(f"{emoji} {Path(issue.file_path).name}:{issue.line_number} - {issue.message}")
                
            for regex_issue in regex_issues:
                report.append(regex_issue)
                
            return "\n".join(report)

        except Exception as e:
            return f"❌ Diff analysis failed: {str(e)}"

    def handle_git_branch_command(self, action: str) -> str:
        """
        Handle --git-branch suggest
        """
        if not self.features.code_analysis:
             return "❌ Git branch assistant requires Code Analysis System"
        
        if action != 'suggest':
             return "❌ Valid actions: 'suggest'"
             
        try:
            # Wait for background initialization if needed
            if hasattr(self, '_init_thread'):
                self._init_thread.join(timeout=5)

            if self.code_analyzer is None:
                return "❌ Code analyzer failed to initialize"
            
            print("🔍 Pondering branch names based on changes...")
            
            diff_content = self.code_analyzer.get_raw_git_diff(staged=True)
            if not diff_content:
                diff_content = self.code_analyzer.get_raw_git_diff(staged=False)
                
            if not diff_content:
                return "ℹ️ No changes found to base a branch name on."
                
            # Use the new helper method  
            suggestions = self.code_analyzer.suggest_branch_names(diff_content)
            
            # Try LLM enhancement if available
            if self.features.multi_model:
                try:
                    from xencode_core import run_query, extract_thinking_and_answer
                    
                    prompt = f"""Based on this git diff, suggest 3 concise, descriptive branch names following git conventions (e.g., feature/short-name, fix/issue-description).
                    
Diff:
{diff_content[:3000]}

Respond with ONLY the 3 branch names, one per line."""
                    
                    response = run_query(prompt, model="codellama:7b")
                    _, answer = extract_thinking_and_answer(response)
                    
                    if answer:
                        llm_suggestions = [line.strip() for line in answer.strip().split('\n') if line.strip() and '/' in line]
                        if llm_suggestions:
                            suggestions = llm_suggestions[:3] + suggestions  # Prepend LLM suggestions
                            
                except Exception:
                    pass  # Silently fall back to heuristics
            
            output = ["🌱 Suggested Branch Names:", "=" * 25]
            seen = set()
            for name in suggestions:
                if name not in seen:
                    seen.add(name)
                    output.append(f"  • {name}")
                    if len(seen) >= 5:
                        break
                
            output.append("\n💡 To switch: git checkout -b <name>")
            
            return "\n".join(output)

        except Exception as e:
            return f"❌ Branch suggestion failed: {str(e)}"

    def handle_git_commit_command(self) -> str: # existing logic follows...
        """Handle --git-commit command with security validation"""
        if not self.features.code_analysis:
            return (
                "❌ Git commit assistance requires Code Analysis System\n"
                "Install Phase 1 Code Analysis System for this feature"
            )

        try:
            # Wait for background initialization if needed
            if hasattr(self, '_init_thread'):
                self._init_thread.join(timeout=5)

            if self.code_analyzer is None:
                return "❌ Code analyzer failed to initialize"

            # Validate current directory and .git security
            current_dir = os.getcwd()

            # Check if we're in a git repository
            git_dir = os.path.join(current_dir, '.git')
            if not os.path.exists(git_dir):
                return "❌ Not in a git repository\nRun this command from within a git repository"

            # Validate .git directory security to prevent exploits
            git_violations = self.security_manager.validate_git_directory(git_dir)

            if git_violations:
                # Security risks detected in .git directory
                risk_summary = []
                for violation in git_violations:
                    risk_summary.append(f"  • {violation.description}")

                warning_message = (
                    "⚠️ Security risks detected in .git directory:\n"
                    + "\n".join(risk_summary)
                    + "\n\nCommit message generation blocked for security"
                )
                return warning_message

            # Validate project path
            if not self.security_manager.validate_project_path(current_dir):
                return "❌ Project path validation failed - potential security risk"

            # Generate commit message based on git diff
            print("🔍 Analyzing git changes...")
            commit_message = self.code_analyzer.generate_commit_message()

            if commit_message.startswith("Error:") or commit_message.startswith(
                "No changes"
            ):
                return f"ℹ️ {commit_message}"

            # Sanitize commit message for security (prevent injection attacks)
            sanitized_message = self.security_manager.sanitize_commit_message(
                commit_message
            )

            # Additional security check - ensure sanitized message is safe
            if not sanitized_message or sanitized_message != commit_message:
                security_note = "\n\n🔒 Note: Commit message was sanitized for security"
            else:
                security_note = ""

            result = (
                f"📝 Suggested commit message:\n\n{sanitized_message}{security_note}"
            )

            # Add usage instructions
            result += "\n\n💡 To use this message:\n"
            result += f"   git commit -m \"{sanitized_message.split(chr(10))[0]}\""

            if '\n\n' in sanitized_message:
                result += "\n   # Or with full message:\n"
                result += f"   git commit -F <(echo \"{sanitized_message}\")"

            return result

        except Exception as e:
            return f"❌ Git commit generation failed: {str(e)}"

    def _handle_feature_status(self) -> str:
        """Handle --feature-status command"""
        status_lines = [
            "🔍 Xencode Feature Status",
            "=" * 30,
            "",
            "Core Systems:",
            "  ✅ Security Manager: Available",
            "  ✅ Context Cache: Available",
            "  ✅ Model Stability: Available",
            "",
            "Enhanced Systems:",
        ]

        # Enhanced feature status
        status_lines.append(
            f"  {'✅' if self.features.multi_model else '❌'} Multi-Model System: {'Available' if self.features.multi_model else 'Not Available'}"
        )
        status_lines.append(
            f"  {'✅' if self.features.smart_context else '❌'} Smart Context System: {'Available' if self.features.smart_context else 'Not Available'}"
        )
        status_lines.append(
            f"  {'✅' if self.features.code_analysis else '❌'} Code Analysis System: {'Available' if self.features.code_analysis else 'Not Available'}"
        )

        status_lines.extend(
            [
                "",
                f"Overall Feature Level: {self.features.feature_level.upper()}",
                "",
                "Available Commands:",
            ]
        )

        # Show available commands
        if self.features.code_analysis:
            status_lines.append("  • --analyze PATH")
            status_lines.append("  • --git-commit")
            status_lines.append("  • --git-review [REF]")
            status_lines.append("  • --git-diff-analyze")
            status_lines.append("  • --git-branch suggest")

        if self.features.multi_model:
            status_lines.append("  • --models")

        if self.features.smart_context:
            status_lines.append("  • --context")

        if self.features.multi_model and self.features.smart_context:
            status_lines.append("  • --smart QUERY")

        status_lines.append("  • --feature-status")
        status_lines.append("  • --refresh-features")

        return "\n".join(status_lines)

    def _handle_refresh_features(self) -> str:
        """Handle --refresh-features command"""
        print("🔄 Refreshing feature detection...")

        # Force refresh feature detection
        self.features = self.feature_detector.detect_features(force_refresh=True)

        # Re-initialize enhanced features if needed
        self._initialize_enhanced_features_background()

        return (
            "✅ Feature detection refreshed\nUse --feature-status to see current status"
        )


def main():
    """Demo function for EnhancedXencodeCLI"""
    print("🧪 Enhanced CLI System Demo")
    print("=" * 40)

    # Initialize enhanced CLI
    cli = EnhancedXencodeCLI()

    # Test argument parsing
    parser = cli.create_parser()

    # Test feature status
    print("\n" + cli._handle_feature_status())

    print("\n🎯 Enhanced CLI System ready!")


if __name__ == "__main__":
    main()
