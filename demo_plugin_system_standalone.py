#!/usr/bin/env python3
"""
Standalone Plugin System Demo

Interactive demonstration of the plugin system without importing the main xencode package.
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress
import tempfile
import shutil

# Add the plugin system directly to path
sys.path.insert(0, str(Path(__file__).parent / "xencode"))

from plugin_system import (
    PluginManager, PluginInterface, PluginMetadata, PluginMarketplace
)

console = Console()


# Example Plugin Implementations
class CodeFormatterPlugin(PluginInterface):
    """Example plugin for code formatting"""
    
    async def initialize(self, context):
        self.context = context
        context.register_service("code_formatter", self)
        console.print("✅ Code Formatter Plugin initialized")
        return True
    
    async def shutdown(self):
        console.print("⚪ Code Formatter Plugin shutdown")
    
    def get_metadata(self):
        return PluginMetadata(
            name="code-formatter",
            version="1.0.0",
            description="Formats code according to style guidelines",
            author="Xencode Team",
            tags=["productivity", "formatting"],
            permissions=["file_access"]
        )
    
    async def format_code(self, code, language="python"):
        """Format code (simulated)"""
        await asyncio.sleep(0.1)  # Simulate processing
        return f"# Formatted {language} code\n{code.strip()}"


class AITranslatorPlugin(PluginInterface):
    """Example plugin for AI translation"""
    
    async def initialize(self, context):
        self.context = context
        context.register_service("ai_translator", self)
        context.subscribe_event("translation_request", self.handle_translation)
        console.print("✅ AI Translator Plugin initialized")
        return True
    
    async def shutdown(self):
        console.print("⚪ AI Translator Plugin shutdown")
    
    def get_metadata(self):
        return PluginMetadata(
            name="ai-translator",
            version="2.1.0",
            description="AI-powered language translation",
            author="AI Team",
            tags=["ai", "translation", "language"],
            dependencies=["transformers", "torch"]
        )
    
    async def handle_translation(self, data):
        """Handle translation events"""
        text = data.get("text")
        target_lang = data.get("target_language", "en")
        console.print(f"🔄 Translating to {target_lang}: {text}")
    
    async def translate(self, text, source_lang="auto", target_lang="en"):
        """Translate text (simulated)"""
        await asyncio.sleep(0.2)  # Simulate AI processing
        return f"[{target_lang}] Translated: {text}"


class SystemMonitorPlugin(PluginInterface):
    """Example plugin for system monitoring"""
    
    async def initialize(self, context):
        self.context = context
        context.register_service("system_monitor", self)
        console.print("✅ System Monitor Plugin initialized")
        return True
    
    async def shutdown(self):
        console.print("⚪ System Monitor Plugin shutdown")
    
    def get_metadata(self):
        return PluginMetadata(
            name="system-monitor",
            version="1.5.0",
            description="Monitor system resources and performance",
            author="DevOps Team",
            tags=["monitoring", "system", "performance"]
        )
    
    async def get_system_info(self):
        """Get system information (simulated)"""
        await asyncio.sleep(0.1)
        return {
            "cpu_usage": "45%",
            "memory_usage": "62%",
            "disk_usage": "78%",
            "uptime": "2 days, 14 hours"
        }


class DemoManager:
    """Manages the plugin system demonstration"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.plugin_manager = PluginManager(self.temp_dir)
        self.marketplace = PluginMarketplace(self.plugin_manager)
        self.sample_plugins = {
            "code_formatter": CodeFormatterPlugin(),
            "ai_translator": AITranslatorPlugin(),
            "system_monitor": SystemMonitorPlugin()
        }
    
    async def setup_sample_plugins(self):
        """Create sample plugin files"""
        console.print("\n🔧 Setting up sample plugins...")
        
        with Progress() as progress:
            task = progress.add_task("Creating plugins...", total=len(self.sample_plugins))
            
            for name, plugin in self.sample_plugins.items():
                # Register plugin directly (simulating file-based loading)
                self.plugin_manager.plugins[plugin.get_metadata().name] = plugin
                await plugin.initialize(self.plugin_manager.context)
                progress.advance(task)
        
        console.print("✅ Sample plugins created successfully!")
    
    def show_welcome(self):
        """Show welcome message"""
        welcome_text = Text()
        welcome_text.append("Welcome to the ", style="bold white")
        welcome_text.append("Xencode Plugin System", style="bold cyan")
        welcome_text.append(" Demo!", style="bold white")
        
        console.print(Panel(
            welcome_text,
            title="🚀 Plugin System Demo",
            border_style="cyan",
            padding=(1, 2)
        ))
        
        console.print("\n📝 This demo showcases the plugin architecture capabilities:")
        console.print("• Dynamic plugin loading and management")
        console.print("• Service registration and discovery")
        console.print("• Event-driven plugin communication")
        console.print("• Plugin marketplace simulation")
        console.print("• Interactive plugin testing\n")
    
    def show_menu(self):
        """Show interactive menu"""
        table = Table(title="🎯 Demo Options")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        
        table.add_row("1", "List loaded plugins")
        table.add_row("2", "Show plugin details")
        table.add_row("3", "Test plugin services")
        table.add_row("4", "Plugin marketplace search")
        table.add_row("5", "System status")
        table.add_row("6", "Event system demo")
        table.add_row("0", "Exit demo")
        
        console.print(table)
    
    async def list_plugins(self):
        """List all loaded plugins"""
        console.print("\n📋 Loaded Plugins:")
        
        table = Table()
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Services", style="blue")
        
        for name, plugin in self.plugin_manager.plugins.items():
            metadata = plugin.get_metadata()
            services = len([s for s in self.plugin_manager.context.list_services() 
                          if s.startswith(metadata.name.replace('-', '_'))])
            table.add_row(
                metadata.name,
                metadata.version,
                "🟢 Active",
                str(services)
            )
        
        console.print(table)
    
    async def show_plugin_details(self):
        """Show detailed plugin information"""
        plugin_names = list(self.plugin_manager.plugins.keys())
        
        if not plugin_names:
            console.print("❌ No plugins loaded")
            return
        
        console.print("\n📋 Available plugins:")
        for i, name in enumerate(plugin_names, 1):
            console.print(f"{i}. {name}")
        
        try:
            choice = int(Prompt.ask("Select plugin number")) - 1
            if 0 <= choice < len(plugin_names):
                plugin_name = plugin_names[choice]
                plugin = self.plugin_manager.plugins[plugin_name]
                metadata = plugin.get_metadata()
                
                details_table = Table(title=f"📦 {metadata.name} Details")
                details_table.add_column("Property", style="cyan")
                details_table.add_column("Value", style="white")
                
                details_table.add_row("Name", metadata.name)
                details_table.add_row("Version", metadata.version)
                details_table.add_row("Description", metadata.description)
                details_table.add_row("Author", metadata.author)
                details_table.add_row("Tags", ", ".join(metadata.tags))
                details_table.add_row("Dependencies", ", ".join(metadata.dependencies) or "None")
                details_table.add_row("Permissions", ", ".join(metadata.permissions) or "None")
                
                console.print(details_table)
            else:
                console.print("❌ Invalid selection")
        except (ValueError, KeyboardInterrupt):
            console.print("❌ Invalid input")
    
    async def test_plugin_services(self):
        """Test plugin services"""
        console.print("\n🧪 Testing Plugin Services:")
        
        # Test Code Formatter
        formatter = self.plugin_manager.context.get_service("code_formatter")
        if formatter:
            code = "def hello():print('world')"
            formatted = await formatter.format_code(code)
            console.print(f"📝 Code Formatter: {formatted}")
        
        # Test AI Translator
        translator = self.plugin_manager.context.get_service("ai_translator")
        if translator:
            translated = await translator.translate("Hello, world!", target_lang="es")
            console.print(f"🌐 AI Translator: {translated}")
        
        # Test System Monitor
        monitor = self.plugin_manager.context.get_service("system_monitor")
        if monitor:
            info = await monitor.get_system_info()
            console.print(f"📊 System Monitor: {info}")
    
    async def marketplace_search(self):
        """Demonstrate marketplace search"""
        console.print("\n🏪 Plugin Marketplace Search:")
        
        search_term = Prompt.ask("Enter search term (or press Enter for all)", default="")
        
        plugins = await self.marketplace.search_plugins(search_term if search_term else None)
        
        table = Table(title=f"🔍 Search Results: '{search_term or 'all'}'")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Description", style="white")
        table.add_column("Tags", style="blue")
        
        for plugin in plugins[:10]:  # Show first 10 results
            table.add_row(
                plugin["name"],
                plugin["version"],
                plugin["description"][:50] + "..." if len(plugin["description"]) > 50 else plugin["description"],
                ", ".join(plugin.get("tags", [])[:3])
            )
        
        console.print(table)
        console.print(f"📊 Found {len(plugins)} plugins")
    
    async def show_system_status(self):
        """Show plugin system status"""
        status = await self.plugin_manager.get_plugin_status()
        
        status_table = Table(title="⚡ System Status")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="green")
        
        for key, value in status.items():
            status_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(status_table)
        
        # Show services
        services = self.plugin_manager.context.list_services()
        console.print(f"\n🔧 Registered Services: {', '.join(services)}")
    
    async def event_system_demo(self):
        """Demonstrate event system"""
        console.print("\n📢 Event System Demo:")
        
        # Emit a translation event
        await self.plugin_manager.context.emit_event("translation_request", {
            "text": "Hello, plugin system!",
            "target_language": "fr"
        })
        
        console.print("✅ Translation event emitted")
        await asyncio.sleep(0.5)  # Let event propagate
    
    async def run_demo(self):
        """Run the interactive demo"""
        try:
            self.show_welcome()
            await self.setup_sample_plugins()
            
            while True:
                self.show_menu()
                choice = Prompt.ask("\n🎯 Select an option", default="0")
                
                try:
                    if choice == "1":
                        await self.list_plugins()
                    elif choice == "2":
                        await self.show_plugin_details()
                    elif choice == "3":
                        await self.test_plugin_services()
                    elif choice == "4":
                        await self.marketplace_search()
                    elif choice == "5":
                        await self.show_system_status()
                    elif choice == "6":
                        await self.event_system_demo()
                    elif choice == "0":
                        break
                    else:
                        console.print("❌ Invalid option")
                
                except Exception as e:
                    console.print(f"❌ Error: {e}")
                
                if choice != "0":
                    Prompt.ask("\n⏸️  Press Enter to continue")
            
            # Cleanup
            await self.plugin_manager.shutdown_all_plugins()
            console.print("\n👋 Thanks for trying the Plugin System Demo!")
            
        finally:
            # Cleanup temporary directory
            shutil.rmtree(self.temp_dir, ignore_errors=True)


async def main():
    """Main entry point"""
    demo = DemoManager()
    await demo.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        console.print(f"\n❌ Demo error: {e}")