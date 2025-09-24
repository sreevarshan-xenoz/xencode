#!/usr/bin/env python3
"""
Demo Plugin System for Xencode Phase 3

Interactive demonstration of the plugin architecture with
real-world examples and comprehensive testing.
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add xencode to path
sys.path.append(str(Path(__file__).parent))

from xencode.plugin_system import (
    PluginManager, PluginInterface, PluginMetadata, PluginMarketplace
)

console = Console()


class CodeFormatterPlugin(PluginInterface):
    """Example code formatting plugin"""
    
    def __init__(self):
        self.name = "code-formatter"
        self.formats_processed = 0
    
    async def initialize(self, context) -> bool:
        self.context = context
        
        # Register formatting service
        context.register_service("code_formatter", self)
        
        # Subscribe to code events
        context.subscribe_event("code_submitted", self.handle_code_submission)
        
        console.print("🎨 Code Formatter Plugin initialized")
        return True
    
    async def shutdown(self) -> None:
        console.print("🎨 Code Formatter Plugin shutting down")
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="code-formatter",
            version="1.2.0",
            description="Advanced code formatting and style checking",
            author="Xencode Community",
            license="MIT",
            tags=["formatting", "code", "productivity"],
            dependencies=["black", "isort"],
            permissions=["file_access"]
        )
    
    async def format_code(self, code: str, language: str = "python") -> str:
        """Format code according to best practices"""
        self.formats_processed += 1
        
        # Simulate formatting
        formatted_code = code.strip()
        if language == "python":
            # Simple formatting simulation
            lines = formatted_code.split('\n')
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                if stripped.endswith(':'):
                    formatted_lines.append('    ' * indent_level + stripped)
                    indent_level += 1
                elif stripped in ['else:', 'elif', 'except:', 'finally:']:
                    indent_level = max(0, indent_level - 1)
                    formatted_lines.append('    ' * indent_level + stripped)
                    indent_level += 1
                else:
                    formatted_lines.append('    ' * indent_level + stripped)
            
            formatted_code = '\n'.join(formatted_lines)
        
        return formatted_code
    
    async def handle_code_submission(self, data):
        """Handle code submission events"""
        if "code" in data:
            formatted = await self.format_code(data["code"], data.get("language", "python"))
            console.print(f"🎨 Formatted code: {len(formatted)} characters")
    
    async def health_check(self):
        return {
            "status": "healthy",
            "details": {
                "formats_processed": self.formats_processed,
                "services_registered": ["code_formatter"],
                "events_subscribed": ["code_submitted"]
            }
        }


class AITranslatorPlugin(PluginInterface):
    """Example AI translation plugin"""
    
    def __init__(self):
        self.name = "ai-translator"
        self.translations_count = 0
        self.supported_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja"]
    
    async def initialize(self, context) -> bool:
        self.context = context
        
        # Register translation service
        context.register_service("translator", self)
        
        # Subscribe to translation requests
        context.subscribe_event("translate_request", self.handle_translation_request)
        
        console.print("🌍 AI Translator Plugin initialized")
        return True
    
    async def shutdown(self) -> None:
        console.print("🌍 AI Translator Plugin shutting down")
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="ai-translator",
            version="2.1.0",
            description="Multi-language translation using advanced AI",
            author="AI Community",
            license="Apache-2.0",
            tags=["translation", "ai", "language", "international"],
            dependencies=["transformers", "torch"],
            permissions=["network_access", "cache_access"]
        )
    
    async def translate_text(self, text: str, from_lang: str, to_lang: str) -> str:
        """Translate text between languages"""
        self.translations_count += 1
        
        # Simulate AI translation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Mock translation (in real implementation, use actual AI model)
        if to_lang == "es":
            return f"[ES] {text} (traducido)"
        elif to_lang == "fr":
            return f"[FR] {text} (traduit)"
        elif to_lang == "de":
            return f"[DE] {text} (übersetzt)"
        else:
            return f"[{to_lang.upper()}] {text} (translated)"
    
    async def handle_translation_request(self, data):
        """Handle translation request events"""
        if all(key in data for key in ["text", "from_lang", "to_lang"]):
            result = await self.translate_text(
                data["text"], data["from_lang"], data["to_lang"]
            )
            console.print(f"🌍 Translated: {data['text']} → {result}")
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return self.supported_languages
    
    async def health_check(self):
        return {
            "status": "healthy",
            "details": {
                "translations_processed": self.translations_count,
                "supported_languages": len(self.supported_languages),
                "services_registered": ["translator"]
            }
        }


class SystemMonitorPlugin(PluginInterface):
    """Example system monitoring plugin"""
    
    def __init__(self):
        self.name = "system-monitor"
        self.metrics_collected = 0
        self.alerts_sent = 0
    
    async def initialize(self, context) -> bool:
        self.context = context
        
        # Register monitoring service
        context.register_service("system_monitor", self)
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitor_system())
        
        console.print("📊 System Monitor Plugin initialized")
        return True
    
    async def shutdown(self) -> None:
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
        console.print("📊 System Monitor Plugin shutting down")
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="system-monitor",
            version="1.0.5",
            description="Real-time system performance monitoring and alerting",
            author="DevOps Team",
            license="MIT",
            tags=["monitoring", "performance", "system", "alerts"],
            dependencies=["psutil", "async-timeout"],
            permissions=["system_access", "network_access"]
        )
    
    async def _monitor_system(self):
        """Background monitoring task"""
        while True:
            try:
                # Simulate metric collection
                await asyncio.sleep(5)
                self.metrics_collected += 1
                
                # Simulate alert condition (every 20 metrics)
                if self.metrics_collected % 20 == 0:
                    await self._send_alert("High memory usage detected")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                console.print(f"📊 Monitoring error: {e}")
    
    async def _send_alert(self, message: str):
        """Send system alert"""
        self.alerts_sent += 1
        await self.context.emit_event("system_alert", {
            "severity": "warning",
            "message": message,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def get_metrics(self) -> dict:
        """Get current system metrics"""
        return {
            "cpu_usage": 45.2,
            "memory_usage": 68.7,
            "disk_usage": 23.1,
            "network_io": 1024.5
        }
    
    async def health_check(self):
        return {
            "status": "monitoring",
            "details": {
                "metrics_collected": self.metrics_collected,
                "alerts_sent": self.alerts_sent,
                "monitoring_active": hasattr(self, 'monitoring_task') and not self.monitoring_task.done()
            }
        }


async def create_demo_plugins(plugin_dir: Path):
    """Create demo plugin files"""
    enabled_dir = plugin_dir / "enabled"
    
    # Code Formatter Plugin
    formatter_file = enabled_dir / "code_formatter.py"
    if not formatter_file.exists():
        formatter_content = f"""
import sys
sys.path.append('{Path(__file__).parent.parent}')
from demo_plugin_system import CodeFormatterPlugin
"""
        formatter_file.write_text(formatter_content)
    
    # AI Translator Plugin
    translator_file = enabled_dir / "ai_translator.py"
    if not translator_file.exists():
        translator_content = f"""
import sys
sys.path.append('{Path(__file__).parent.parent}')
from demo_plugin_system import AITranslatorPlugin
"""
        translator_file.write_text(translator_content)
    
    # System Monitor Plugin
    monitor_file = enabled_dir / "system_monitor.py"
    if not monitor_file.exists():
        monitor_content = f"""
import sys
sys.path.append('{Path(__file__).parent.parent}')
from demo_plugin_system import SystemMonitorPlugin
"""
        monitor_file.write_text(monitor_content)


async def demonstrate_plugin_lifecycle(manager: PluginManager):
    """Demonstrate plugin lifecycle management"""
    console.print(Panel.fit("🔄 Plugin Lifecycle Demo", style="bold blue"))
    
    # Show current plugins
    console.print("\n📋 Current Plugin Status:")
    manager.display_plugins()
    
    if not manager.plugins:
        console.print("❌ No plugins loaded. Load some plugins first!")
        return
    
    # Demonstrate plugin operations
    plugin_names = list(manager.plugins.keys())
    
    if len(plugin_names) > 0:
        test_plugin = plugin_names[0]
        
        console.print(f"\n🔄 Testing lifecycle operations with '{test_plugin}':")
        
        # Get initial status
        status = await manager.get_plugin_status()
        console.print(f"✅ Plugin active: {status['plugins'][test_plugin]['active']}")
        
        # Test reload
        console.print(f"🔄 Reloading plugin '{test_plugin}'...")
        success = await manager.reload_plugin(test_plugin)
        console.print(f"{'✅' if success else '❌'} Reload {'successful' if success else 'failed'}")
        
        # Show final status
        manager.display_plugins()


async def demonstrate_event_system(manager: PluginManager):
    """Demonstrate the event system"""
    console.print(Panel.fit("📡 Event System Demo", style="bold green"))
    
    # Test code formatting event
    console.print("\n🎨 Testing code formatting event:")
    test_code = """
def hello_world():
print("Hello, World!")
if True:
print("This is a test")
"""
    
    await manager.emit_event("code_submitted", {
        "code": test_code,
        "language": "python"
    })
    
    # Test translation event
    console.print("\n🌍 Testing translation event:")
    await manager.emit_event("translate_request", {
        "text": "Hello, how are you?",
        "from_lang": "en",
        "to_lang": "es"
    })
    
    # Test system alert
    console.print("\n📊 Testing system alert:")
    await manager.emit_event("system_alert", {
        "severity": "info",
        "message": "Demo alert from event system",
        "timestamp": asyncio.get_event_loop().time()
    })


async def demonstrate_service_interaction(manager: PluginManager):
    """Demonstrate service interaction between plugins"""
    console.print(Panel.fit("🔗 Service Interaction Demo", style="bold yellow"))
    
    # Get services
    services = manager.context.list_services()
    console.print(f"\n📋 Available services: {', '.join(services)}")
    
    # Test code formatter service
    formatter = manager.context.get_service("code_formatter")
    if formatter:
        console.print("\n🎨 Testing code formatter service:")
        test_code = "def test():print('hello')"
        formatted = await formatter.format_code(test_code, "python")
        console.print(f"Original: {test_code}")
        console.print(f"Formatted:\n{formatted}")
    
    # Test translator service
    translator = manager.context.get_service("translator")
    if translator:
        console.print("\n🌍 Testing translator service:")
        languages = translator.get_supported_languages()
        console.print(f"Supported languages: {', '.join(languages[:5])}...")
        
        translated = await translator.translate_text("Good morning!", "en", "fr")
        console.print(f"Translation: 'Good morning!' → '{translated}'")
    
    # Test system monitor
    monitor = manager.context.get_service("system_monitor")
    if monitor:
        console.print("\n📊 Testing system monitor service:")
        metrics = await monitor.get_metrics()
        
        table = Table(title="System Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for metric, value in metrics.items():
            table.add_row(metric.replace('_', ' ').title(), f"{value}%")
        
        console.print(table)


async def demonstrate_marketplace(manager: PluginManager):
    """Demonstrate marketplace functionality"""
    console.print(Panel.fit("🏪 Plugin Marketplace Demo", style="bold magenta"))
    
    marketplace = PluginMarketplace(manager)
    
    # Search all plugins
    console.print("\n🔍 All available plugins:")
    all_plugins = await marketplace.search_plugins()
    marketplace.display_marketplace(all_plugins)
    
    # Search by query
    console.print("\n🔍 Searching for 'ai' plugins:")
    ai_plugins = await marketplace.search_plugins("ai")
    marketplace.display_marketplace(ai_plugins)
    
    # Search by tags
    console.print("\n🔍 Searching for productivity plugins:")
    productivity_plugins = await marketplace.search_plugins(tags=["productivity"])
    marketplace.display_marketplace(productivity_plugins)


async def interactive_demo():
    """Interactive plugin system demonstration"""
    console.print(Panel.fit(
        "🔌 Xencode Plugin System - Interactive Demo\n"
        "Explore the power of extensible architecture!",
        style="bold cyan"
    ))
    
    # Setup plugin directory
    plugin_dir = Path.home() / ".xencode" / "plugins"
    manager = PluginManager(plugin_dir)
    
    # Create demo plugins
    await create_demo_plugins(plugin_dir)
    
    try:
        # Load all plugins
        console.print("\n🚀 Loading plugins...")
        await manager.load_all_plugins()
        
        while True:
            console.print("\n" + "="*60)
            console.print("🎮 Plugin System Demo Menu:")
            console.print("1. 📋 Show Plugin Status")
            console.print("2. 🔄 Lifecycle Management Demo")
            console.print("3. 📡 Event System Demo")
            console.print("4. 🔗 Service Interaction Demo")
            console.print("5. 🏪 Marketplace Demo")
            console.print("6. ⚡ Performance Test")
            console.print("7. 🔧 Plugin Health Check")
            console.print("8. 🎨 Custom Plugin Demo")
            console.print("0. 🚪 Exit")
            
            choice = Prompt.ask("\nSelect an option", choices=[
                "1", "2", "3", "4", "5", "6", "7", "8", "0"
            ])
            
            if choice == "1":
                console.print("\n📋 Current Plugin Status:")
                manager.display_plugins()
                status = await manager.get_plugin_status()
                console.print(Panel(
                    f"Total: {status['total_plugins']} | "
                    f"Active: {status['active_plugins']} | "
                    f"Errors: {status['error_count']}",
                    title="📊 System Summary"
                ))
            
            elif choice == "2":
                await demonstrate_plugin_lifecycle(manager)
            
            elif choice == "3":
                await demonstrate_event_system(manager)
            
            elif choice == "4":
                await demonstrate_service_interaction(manager)
            
            elif choice == "5":
                await demonstrate_marketplace(manager)
            
            elif choice == "6":
                console.print("\n⚡ Running performance test...")
                with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
                    task = progress.add_task("Testing event performance...", total=None)
                    
                    # Event performance test
                    start_time = asyncio.get_event_loop().time()
                    for i in range(1000):
                        await manager.emit_event("performance_test", {"iteration": i})
                    end_time = asyncio.get_event_loop().time()
                    
                    progress.update(task, completed=True)
                
                console.print(f"✅ Emitted 1000 events in {end_time - start_time:.3f} seconds")
                console.print(f"⚡ Average: {(end_time - start_time) * 1000:.2f} ms per event")
            
            elif choice == "7":
                console.print("\n🔧 Plugin Health Check:")
                for name, plugin in manager.plugins.items():
                    health = await plugin.instance.health_check()
                    status_icon = "✅" if health["status"] == "healthy" else "⚠️"
                    console.print(f"{status_icon} {name}: {health['status']}")
                    if health.get("details"):
                        for key, value in health["details"].items():
                            console.print(f"   • {key}: {value}")
            
            elif choice == "8":
                console.print("\n🎨 Custom Plugin Interaction:")
                
                # Interactive code formatting
                if "code-formatter" in manager.plugins:
                    code = Prompt.ask("Enter Python code to format", 
                                    default="def test():print('hello world')")
                    formatter = manager.context.get_service("code_formatter")
                    if formatter:
                        formatted = await formatter.format_code(code, "python")
                        console.print("🎨 Formatted code:")
                        console.print(Panel(formatted, title="Formatted Code"))
                
                # Interactive translation
                if "ai-translator" in manager.plugins:
                    text = Prompt.ask("Enter text to translate", default="Hello world!")
                    to_lang = Prompt.ask("Target language", choices=["es", "fr", "de"], default="es")
                    translator = manager.context.get_service("translator")
                    if translator:
                        translated = await translator.translate_text(text, "en", to_lang)
                        console.print(f"🌍 Translation: '{text}' → '{translated}'")
            
            elif choice == "0":
                break
            
            input("\nPress Enter to continue...")
    
    finally:
        console.print("\n🔌 Shutting down plugin system...")
        await manager.shutdown_all_plugins()
        console.print("✅ Plugin system demo completed!")


async def main():
    """Main demo function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_demo()
    else:
        # Quick demo
        console.print(Panel.fit("🔌 Xencode Plugin System - Quick Demo", style="bold cyan"))
        
        plugin_dir = Path.home() / ".xencode" / "plugins"
        manager = PluginManager(plugin_dir)
        
        await create_demo_plugins(plugin_dir)
        await manager.load_all_plugins()
        
        manager.display_plugins()
        await demonstrate_event_system(manager)
        await demonstrate_service_interaction(manager)
        
        await manager.shutdown_all_plugins()


if __name__ == "__main__":
    asyncio.run(main())