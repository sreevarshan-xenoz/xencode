#!/usr/bin/env python3
"""
Xencode Enhanced - Phase 2 Integration
Merges all Phase 1 features into the main system with new CLI commands
"""

import sys
import os
import argparse
from pathlib import Path

# Import our Phase 1 systems
from multi_model_system import MultiModelManager
from smart_context_system import SmartContextManager
from code_analysis_system import CodeAnalyzer, analyze_code_command

# Import original xencode functionality
sys.path.append('.')
import xencode_core

class XencodeEnhanced:
    """Enhanced Xencode with all Phase 1 features integrated"""
    
    def __init__(self):
        self.multi_model = MultiModelManager()
        self.context_manager = SmartContextManager()
        self.code_analyzer = CodeAnalyzer()
        self.current_model = xencode_core.DEFAULT_MODEL
        
    def handle_analyze_command(self, path: str = ".", recursive: bool = True):
        """Handle --analyze command"""
        print("🔍 Xencode Code Analysis")
        print("=" * 40)
        
        result = analyze_code_command(path, recursive)
        print(result)
        
        # Offer AI-powered suggestions
        if "issues found" in result and "0" not in result:
            print("\n🤖 AI-Powered Suggestions:")
            print("Run: xencode \"How can I fix these code issues?\" --context")
            print("     xencode \"Explain the security issues found\" --smart")
    
    def handle_models_command(self):
        """Handle --models command"""
        print("🤖 Xencode Multi-Model System")
        print("=" * 40)
        
        comparison = self.multi_model.get_model_comparison()
        
        if not comparison:
            print("❌ No models available. Run: xencode --update")
            return
        
        print(f"\n📋 Available Models ({len(comparison)}):")
        for model_name, info in comparison.items():
            current = " (CURRENT)" if model_name == self.current_model else ""
            print(f"\n🤖 {model_name}{current}")
            print(f"   Size: {info['size']}")
            print(f"   Speed: {info['speed']}")
            print(f"   Quality: {info['quality']}")
            print(f"   Strengths: {info['strengths']}")
            print(f"   Specialties: {', '.join(info['specialties'])}")
            print(f"   Context: {info['context_window']}")
        
        print(f"\n💡 Usage:")
        print(f"   xencode --smart \"your query\"     # Auto-select best model")
        print(f"   xencode -m mistral \"query\"       # Use specific model")
    
    def handle_context_command(self, query: str = ""):
        """Handle --context command"""
        print("🧠 Xencode Smart Context")
        print("=" * 40)
        
        if query:
            context = self.context_manager.get_context_for_query(query)
            print(f"\n🎯 Context for: '{query}'")
            print(f"Length: {len(context)} characters")
            print("\nRelevant Context:")
            print("-" * 20)
            print(context[:1000] + "..." if len(context) > 1000 else context)
        else:
            summary = self.context_manager.get_context_summary()
            print(f"\n📊 Context Summary:")
            print(f"   Project: {summary['project_root']}")
            print(f"   Total items: {summary['total_items']}")
            print(f"   By type: {summary['by_type']}")
            print(f"   Estimated tokens: {summary['estimated_tokens']}")
            
            if summary['total_items'] > 0:
                print(f"\n💡 Usage:")
                print(f"   xencode --context \"your query\"   # Get relevant context")
                print(f"   xencode \"query\" --with-context   # Include context in response")
    
    def handle_smart_command(self, query: str):
        """Handle --smart command with automatic model selection"""
        print("🎯 Xencode Smart Mode")
        print("=" * 40)
        
        # Get model recommendation
        suggested_model, reason = self.multi_model.suggest_best_model(query)
        query_type = self.multi_model.detect_query_type(query)
        
        print(f"\n🧠 Query Analysis:")
        print(f"   Type: {query_type.value}")
        print(f"   Suggested Model: {suggested_model}")
        print(f"   Reason: {reason}")
        
        # Get relevant context
        context = self.context_manager.get_context_for_query(query)
        context_summary = self.context_manager.get_context_summary()
        
        print(f"\n📋 Context Added:")
        print(f"   Files: {context_summary['by_type'].get('file', 0)}")
        print(f"   Tokens: {context_summary['estimated_tokens']}")
        
        # Build enhanced prompt with context
        if context:
            enhanced_query = f"Context:\n{context}\n\nQuery: {query}"
        else:
            enhanced_query = query
        
        print(f"\n🚀 Processing with {suggested_model}...")
        print("=" * 40)
        
        # Use the suggested model for the query
        try:
            if suggested_model != self.current_model:
                print(f"🔄 Switching to {suggested_model}...")
            
            # Process the query (this would call the actual xencode system)
            response = xencode_core.run_query(suggested_model, enhanced_query)
            
            # Extract and display only the answer (no thinking in smart mode)
            thinking, answer = xencode_core.extract_thinking_and_answer(response)
            if answer.strip():
                print(answer.strip())
            else:
                print(response.strip())
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Falling back to default model...")
            response = xencode_core.run_query(self.current_model, query)
            thinking, answer = xencode_core.extract_thinking_and_answer(response)
            print(answer.strip() if answer.strip() else response.strip())
    
    def handle_git_commit_command(self):
        """Handle --git-commit command"""
        print("📝 Xencode Git Integration")
        print("=" * 40)
        
        try:
            # Get git diff
            result = subprocess.run(
                ["git", "diff", "--cached"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            if not result.stdout.strip():
                print("❌ No staged changes found")
                print("💡 Stage changes first: git add <files>")
                return
            
            diff_content = result.stdout
            
            # Analyze the diff with AI
            commit_query = f"""
            Analyze this git diff and generate a concise, professional commit message:

            {diff_content}

            Requirements:
            - Use conventional commit format (feat:, fix:, docs:, etc.)
            - Be specific about what changed
            - Keep under 72 characters for the title
            - Add body if needed for complex changes
            """
            
            print("🧠 Analyzing changes...")
            
            # Use smart model selection for commit message generation
            suggested_model, reason = self.multi_model.suggest_best_model(commit_query)
            print(f"Using {suggested_model} ({reason})")
            
            response = xencode_core.run_query(suggested_model, commit_query)
            thinking, answer = xencode_core.extract_thinking_and_answer(response)
            
            commit_message = answer.strip() if answer.strip() else response.strip()
            
            print(f"\n📝 Suggested Commit Message:")
            print("-" * 40)
            print(commit_message)
            print("-" * 40)
            
            # Ask for confirmation
            confirm = input("\n✅ Use this commit message? (y/N): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                # Create the commit
                subprocess.run(["git", "commit", "-m", commit_message], check=True)
                print("✅ Commit created successfully!")
            else:
                print("❌ Commit cancelled")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Git error: {e}")
            print("💡 Make sure you're in a git repository with staged changes")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Enhanced main function with new CLI commands"""
    parser = argparse.ArgumentParser(
        description="Xencode Enhanced - AI Development Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  xencode "hello"                          # Basic query
  xencode --smart "write a Python function" # Auto-select best model
  xencode --analyze ./src/                 # Analyze code
  xencode --models                         # Show available models
  xencode --context "fix this bug"         # Show relevant context
  xencode --git-commit                     # Generate commit message
        """
    )
    
    # Enhanced command arguments
    parser.add_argument("query", nargs="*", help="Query to process")
    parser.add_argument("--analyze", metavar="PATH", help="Analyze code in directory/file")
    parser.add_argument("--models", action="store_true", help="Show multi-model information")
    parser.add_argument("--context", metavar="QUERY", help="Show context for query")
    parser.add_argument("--smart", metavar="QUERY", help="Smart mode with auto model selection")
    parser.add_argument("--git-commit", action="store_true", help="Generate git commit message")
    parser.add_argument("-m", "--model", help="Specify model to use")
    parser.add_argument("--with-context", action="store_true", help="Include project context")
    
    # Original xencode arguments
    parser.add_argument("--chat-mode", action="store_true", help="Launch chat mode")
    parser.add_argument("--inline", action="store_true", help="Force inline mode")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--update", action="store_true", help="Update models")
    parser.add_argument("--online", choices=["true", "false"], default="true", help="Online mode")
    
    args = parser.parse_args()
    
    # Initialize enhanced system
    enhanced = XencodeEnhanced()
    
    # Handle new enhanced commands
    if args.analyze:
        enhanced.handle_analyze_command(args.analyze)
        return
    
    if args.models:
        enhanced.handle_models_command()
        return
    
    if args.context:
        enhanced.handle_context_command(args.context)
        return
    
    if args.smart:
        enhanced.handle_smart_command(args.smart)
        return
    
    if args.git_commit:
        enhanced.handle_git_commit_command()
        return
    
    # Handle original xencode commands by delegating to xencode_core
    if args.list_models:
        xencode_core.list_models()
        return
    
    if args.update:
        model = args.model or xencode_core.DEFAULT_MODEL
        xencode_core.update_model(model)
        return
    
    if args.chat_mode:
        xencode_core.chat_mode(args.model or xencode_core.DEFAULT_MODEL, args.online)
        return
    
    # Handle query with optional enhancements
    if args.query:
        query = " ".join(args.query)
        
        if args.with_context:
            # Add project context to the query
            context = enhanced.context_manager.get_context_for_query(query)
            if context:
                enhanced_query = f"Context:\n{context}\n\nQuery: {query}"
                print("🧠 Added project context to query")
            else:
                enhanced_query = query
        else:
            enhanced_query = query
        
        # Use specified model or current model
        model = args.model or enhanced.current_model
        
        try:
            response = xencode_core.run_query(model, enhanced_query)
            # Show only answer in inline mode
            thinking, answer = xencode_core.extract_thinking_and_answer(response)
            if answer.strip():
                print(answer.strip())
            else:
                print(response.strip())
        except Exception as e:
            print(f"❌ Error: {e}")
        return
    
    # No arguments - launch chat mode (default behavior)
    xencode_core.chat_mode(args.model or xencode_core.DEFAULT_MODEL, args.online)

if __name__ == "__main__":
    main()