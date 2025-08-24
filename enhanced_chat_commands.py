#!/usr/bin/env python3
"""
Enhanced Chat Commands for Xencode Phase 2 Integration

Provides advanced commands within chat mode including:
- /analyze for code analysis
- /model for model management
- /context for project context
- /smart for intelligent mode toggle

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9
"""

import os
import time
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

# Import Phase 1 systems with graceful fallback
try:
    from multi_model_system import MultiModelManager
    MULTI_MODEL_AVAILABLE = True
except ImportError:
    MULTI_MODEL_AVAILABLE = False
    MultiModelManager = None

try:
    from smart_context_system import SmartContextManager
    SMART_CONTEXT_AVAILABLE = True
except ImportError:
    SMART_CONTEXT_AVAILABLE = False
    SmartContextManager = None

try:
    from code_analysis_system import CodeAnalyzer
    CODE_ANALYSIS_AVAILABLE = True
except ImportError:
    CODE_ANALYSIS_AVAILABLE = False
    CodeAnalyzer = None

# Import security and resource management
from security_manager import SecurityManager
from context_cache_manager import ContextCacheManager
from model_stability_manager import ModelStabilityManager


class EnhancedChatCommands:
    """
    Enhanced chat command handler with Phase 1 integration
    """
    
    def __init__(self, feature_availability, enhanced_systems=None):
        """
        Initialize enhanced chat commands
        
        Args:
            feature_availability: FeatureAvailability object
            enhanced_systems: Dict of initialized enhanced systems
        """
        self.features = feature_availability
        self.systems = enhanced_systems or {}
        
        # Initialize core systems
        self.security_manager = SecurityManager()
        self.context_cache = ContextCacheManager()
        self.model_stability = ModelStabilityManager()
        
        # Smart mode state
        self.smart_mode_enabled = False
        self.last_model_suggestion = None
        self.consecutive_query_count = 0
        self.last_query_time = 0
    
    def handle_chat_command(self, command: str, args: str, current_model: str) -> Tuple[str, Optional[str]]:
        """
        Handle enhanced chat commands
        
        Args:
            command: Command name (without /)
            args: Command arguments
            current_model: Currently selected model
            
        Returns:
            Tuple of (response_message, new_model_or_None)
        """
        if command == "analyze":
            return self.handle_analyze_command(args), None
        elif command == "model":
            return self.handle_model_command(args, current_model)
        elif command == "models":
            return self.handle_models_command(), None
        elif command == "context":
            return self.handle_context_command(args), None
        elif command == "smart":
            return self.handle_smart_command(args), None
        elif command == "help":
            return self.get_enhanced_help(), None
        else:
            return f"❌ Unknown command: /{command}\nType /help for available commands", None
    
    def handle_analyze_command(self, args: str) -> str:
        """Handle /analyze command in chat mode"""
        if not self.features.code_analysis:
            return ("❌ Code analysis not available\n"
                   "Install Phase 1 Code Analysis System for this feature")
        
        # Get path argument or use current directory
        path = args.strip() if args.strip() else "."
        
        try:
            code_analyzer = self.systems.get('code_analyzer')
            if not code_analyzer:
                return "❌ Code analyzer not initialized"
            
            # Validate path with security manager
            if not self.security_manager.validate_project_path(path):
                return f"❌ Invalid or unsafe path: {path}"
            
            # Perform analysis
            print(f"🔍 Analyzing code in: {path}")
            
            if os.path.isfile(path):
                # Analyze single file
                file_path = Path(path)
                issues = code_analyzer.analyze_file(file_path)
                
                if not issues:
                    return f"✅ No issues found in {path}"
                
                result = f"📊 Analysis Results for {path}:\n\n"
                for issue in issues:
                    result += f"• {issue.severity.value.upper()}: {issue.message}\n"
                    if issue.suggestion:
                        result += f"  💡 Suggestion: {issue.suggestion}\n"
                
                return result
            else:
                # Analyze directory
                dir_path = Path(path)
                results = code_analyzer.analyze_directory(dir_path, recursive=True)
                
                if not results:
                    return f"✅ No supported files found in {path}"
                
                # Generate summary report
                total_issues = sum(len(issues) for issues in results.values())
                
                if total_issues == 0:
                    return f"✅ No issues found in {len(results)} files"
                
                report = f"📊 Analysis Results for {path}:\n\n"
                report += f"Files analyzed: {len(results)}\n"
                report += f"Total issues: {total_issues}\n\n"
                
                # Show top issues
                all_issues = []
                for file_path, issues in results.items():
                    for issue in issues:
                        all_issues.append((file_path, issue))
                
                # Sort by severity
                severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
                all_issues.sort(key=lambda x: severity_order.get(x[1].severity.value, 4))
                
                # Show top 10 issues
                for i, (file_path, issue) in enumerate(all_issues[:10]):
                    report += f"{i+1}. {Path(file_path).name}: {issue.severity.value.upper()} - {issue.message}\n"
                
                if len(all_issues) > 10:
                    report += f"\n... and {len(all_issues) - 10} more issues"
                
                return report
                
        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"
    
    def handle_model_command(self, args: str, current_model: str) -> Tuple[str, Optional[str]]:
        """Handle /model command in chat mode"""
        if not args.strip():
            # Show current model and available options
            if not self.features.multi_model:
                return ("📦 Current Model: " + current_model + "\n" +
                       "❌ Multi-model system not available for enhanced model management"), None
            
            try:
                multi_model = self.systems.get('multi_model')
                if not multi_model:
                    return f"📦 Current Model: {current_model}\n❌ Multi-model system not initialized", None
                
                models_info = multi_model.get_available_models_with_capabilities()
                return f"📦 Current Model: {current_model}\n\n{models_info}", None
                
            except Exception as e:
                return f"📦 Current Model: {current_model}\n❌ Failed to get model info: {e}", None
        
        # Switch to specified model
        new_model = args.strip()
        
        if not self.features.multi_model:
            return f"❌ Model switching requires Multi-Model System\nStaying with {current_model}", None
        
        try:
            # Test model stability before switching
            stability_result = self.model_stability.test_model_stability(new_model)
            
            if not stability_result.is_stable:
                return (f"❌ Model {new_model} is not stable or available\n"
                       f"Error: {stability_result.error_message or 'Unknown stability issue'}\n"
                       f"Staying with {current_model}"), None
            
            # Model is stable - proceed with switch
            return f"✅ Switched to model: {new_model}", new_model
            
        except Exception as e:
            return f"❌ Model switch failed: {str(e)}\nStaying with {current_model}", None
    
    def handle_models_command(self) -> str:
        """Handle /models command in chat mode"""
        if not self.features.multi_model:
            return ("❌ Enhanced model listing requires Multi-Model System\n"
                   "Use xencode --list-models for basic model listing")
        
        try:
            multi_model = self.systems.get('multi_model')
            if not multi_model:
                return "❌ Multi-model system not initialized"
            
            models_info = multi_model.get_available_models_with_capabilities()
            return f"📦 Available Models:\n{models_info}"
            
        except Exception as e:
            return f"❌ Failed to get models: {str(e)}"
    
    def handle_context_command(self, args: str) -> str:
        """Handle /context command in chat mode"""
        if not args.strip():
            # Show current context
            if not self.features.smart_context:
                return ("❌ Context display requires Smart Context System\n"
                       "Install Phase 1 Smart Context System for project awareness")
            
            try:
                smart_context = self.systems.get('smart_context')
                if not smart_context:
                    return "❌ Smart context system not initialized"
                
                current_dir = os.getcwd()
                context_info = smart_context.get_project_context_summary(current_dir)
                
                return f"📁 Project Context:\n{context_info}"
                
            except Exception as e:
                return f"❌ Failed to get context: {str(e)}"
        
        # Handle context subcommands
        subcommand = args.strip().lower()
        
        if subcommand == "clear":
            try:
                # Clear context cache
                current_dir = os.getcwd()
                project_hash = self.context_cache.get_project_hash(current_dir)
                
                if self.context_cache.clear_context(project_hash):
                    return "✅ Context cache cleared"
                else:
                    return "⚠️ No context cache to clear"
                    
            except Exception as e:
                return f"❌ Failed to clear context: {str(e)}"
        
        elif subcommand == "refresh":
            try:
                if not self.features.smart_context:
                    return "❌ Context refresh requires Smart Context System"
                
                smart_context = self.systems.get('smart_context')
                if not smart_context:
                    return "❌ Smart context system not initialized"
                
                current_dir = os.getcwd()
                
                # Force refresh context
                print("🔄 Refreshing project context...")
                context_info = smart_context.scan_project_context(current_dir, force_refresh=True)
                
                return f"✅ Context refreshed\n📁 Updated Context:\n{context_info}"
                
            except Exception as e:
                return f"❌ Failed to refresh context: {str(e)}"
        
        else:
            return f"❌ Unknown context command: {subcommand}\nAvailable: clear, refresh"
    
    def handle_smart_command(self, args: str) -> str:
        """Handle /smart command in chat mode"""
        if not (self.features.multi_model and self.features.smart_context):
            return ("❌ Smart mode requires both Multi-Model and Smart Context systems\n"
                   "Install Phase 1 systems for this feature")
        
        if not args.strip():
            # Show current smart mode status
            status = "enabled" if self.smart_mode_enabled else "disabled"
            return f"🤖 Smart mode: {status}\nUse '/smart on' or '/smart off' to toggle"
        
        command = args.strip().lower()
        
        if command == "on":
            self.smart_mode_enabled = True
            return ("✅ Smart mode enabled\n"
                   "🤖 I'll automatically select the best model for each query")
        
        elif command == "off":
            self.smart_mode_enabled = False
            return ("✅ Smart mode disabled\n"
                   "🤖 I'll use the currently selected model for all queries")
        
        else:
            return f"❌ Unknown smart command: {command}\nUse 'on' or 'off'"
    
    def suggest_model_for_query(self, query: str, current_model: str) -> Tuple[str, str]:
        """
        Suggest best model for query when smart mode is enabled
        
        Args:
            query: User query
            current_model: Currently selected model
            
        Returns:
            Tuple of (suggested_model, reasoning)
        """
        if not self.smart_mode_enabled:
            return current_model, "Smart mode disabled"
        
        if not (self.features.multi_model and self.features.smart_context):
            return current_model, "Smart mode requires Phase 1 systems"
        
        try:
            multi_model = self.systems.get('multi_model')
            if not multi_model:
                return current_model, "Multi-model system not available"
            
            # Check for consecutive queries within 3 seconds (maintain conversational flow)
            current_time = time.time()
            if (current_time - self.last_query_time < 3.0 and 
                self.last_model_suggestion and
                self.consecutive_query_count > 0):
                
                self.consecutive_query_count += 1
                self.last_query_time = current_time
                return self.last_model_suggestion, "Maintaining conversational flow"
            
            # Analyze query and select best model
            query_analysis = multi_model.analyze_query_type(query)
            suggested_model = query_analysis.suggested_model
            
            # Test model stability
            stability_result = self.model_stability.test_model_stability(suggested_model)
            
            if not stability_result.is_stable:
                # Use fallback chain
                fallback_chain = self.model_stability.get_fallback_chain(query_analysis.query_type)
                for fallback_model in fallback_chain:
                    fallback_result = self.model_stability.test_model_stability(fallback_model)
                    if fallback_result.is_stable:
                        suggested_model = fallback_model
                        break
                else:
                    # Use emergency model
                    suggested_model = self.model_stability.get_emergency_model()
            
            # Update state
            self.last_model_suggestion = suggested_model
            self.consecutive_query_count = 1
            self.last_query_time = current_time
            
            reasoning = f"Using {suggested_model} for {query_analysis.query_type} query"
            
            return suggested_model, reasoning
            
        except Exception as e:
            return current_model, f"Smart selection failed: {e}"
    
    def get_enhanced_help(self) -> str:
        """Get help text for enhanced chat commands"""
        help_text = "🚀 Enhanced Chat Commands:\n\n"
        
        # Core commands (always available)
        help_text += "📋 Core Commands:\n"
        help_text += "  /help - Show this help message\n\n"
        
        # Enhanced commands (based on availability)
        if self.features.code_analysis:
            help_text += "🔍 Code Analysis:\n"
            help_text += "  /analyze [path] - Analyze code quality (default: current directory)\n\n"
        
        if self.features.multi_model:
            help_text += "🤖 Model Management:\n"
            help_text += "  /model - Show current model and capabilities\n"
            help_text += "  /model <name> - Switch to specified model\n"
            help_text += "  /models - List all available models\n\n"
        
        if self.features.smart_context:
            help_text += "📁 Context Management:\n"
            help_text += "  /context - Show current project context\n"
            help_text += "  /context clear - Clear context cache\n"
            help_text += "  /context refresh - Refresh project context\n\n"
        
        if self.features.multi_model and self.features.smart_context:
            help_text += "🧠 Smart Mode:\n"
            help_text += "  /smart - Show smart mode status\n"
            help_text += "  /smart on - Enable automatic model selection\n"
            help_text += "  /smart off - Disable automatic model selection\n\n"
        
        # Feature status
        help_text += f"🎚️ Feature Level: {self.features.feature_level.upper()}\n"
        
        if not self.features.enhanced_features_available:
            help_text += "\n💡 Install Phase 1 systems for enhanced features:\n"
            help_text += "  • Multi-Model System\n"
            help_text += "  • Smart Context System\n"
            help_text += "  • Code Analysis System"
        
        return help_text


def main():
    """Demo function for EnhancedChatCommands"""
    print("🧪 Enhanced Chat Commands Demo")
    print("=" * 40)
    
    # Mock feature availability for demo
    class MockFeatures:
        multi_model = True
        smart_context = True
        code_analysis = True
        enhanced_features_available = True
        feature_level = "advanced"
    
    # Initialize enhanced chat commands
    chat_commands = EnhancedChatCommands(MockFeatures())
    
    # Test help command
    print(chat_commands.get_enhanced_help())
    
    print("\n🎯 Enhanced Chat Commands ready!")


if __name__ == "__main__":
    main()