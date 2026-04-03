#!/usr/bin/env python3
"""
Custom Models CLI and TUI Demo

Demonstrates the usage of Custom AI Models feature through:
1. Feature usage programmatically
2. CLI commands demonstration
3. Feature manager integration
4. Model training and monitoring
5. Interactive TUI demo

Run this file to see all examples in action.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


async def demo_1_feature_usage():
    """Demo 1: Using the Custom Models feature programmatically"""
    print("\n" + "="*80)
    print("DEMO 1: Custom Models Feature Usage")
    print("="*80 + "\n")
    
    from xencode.features.custom_models import CustomModelManager, FeatureConfig
    
    # Create and initialize feature
    config = FeatureConfig(
        name="custom_models",
        enabled=True,
        config={
            'max_models': 10,
            'training': {
                'max_epochs': 10,
                'batch_size': 32
            },
            'performance': {
                'min_accuracy': 0.85
            }
        }
    )
    
    feature = CustomModelManager(config)
    await feature.initialize()
    
    print("✅ Custom Models feature initialized")
    print(f"   Name: {feature.name}")
    print(f"   Description: {feature.description}")
    print(f"   Status: {feature.get_status().value}")
    
    # Analyze a codebase
    print("\n📊 Analyzing codebase...")
    try:
        # Use current directory as example
        codebase_path = str(Path.cwd())
        analysis = await feature.analyze(codebase_path)
        
        print(f"✅ Analysis complete!")
        print(f"   Total patterns: {analysis['total_patterns']}")
        print(f"   Codebase path: {analysis['codebase_path']}")
        
        # Show some patterns
        if analysis['patterns']:
            print("\n   Top patterns:")
            for pattern in analysis['patterns'][:5]:
                print(f"   • {pattern['pattern_type']}: {pattern['frequency']} occurrences")
    except Exception as e:
        print(f"⚠️  Analysis skipped: {e}")
    
    # Train a model
    print("\n🎯 Training a custom model...")
    try:
        result = await feature.train(
            model_name="demo-model",
            codebase_path=codebase_path,
            task_type="code_completion"
        )
        
        print(f"✅ Training complete!")
        print(f"   Model: {result['model_name']}")
        print(f"   Version: {result['version']}")
        print(f"   Status: {result['status']}")
        print(f"   Accuracy: {result['accuracy']:.1%}")
        print(f"   Training samples: {result['training_samples']}")
    except Exception as e:
        print(f"⚠️  Training skipped: {e}")
    
    # List models
    print("\n📋 Listing custom models...")
    models = await feature.list_models()
    
    if models:
        print(f"✅ Found {len(models)} models:")
        for model in models:
            print(f"   • {model['name']} (version: {model['current_version']})")
    else:
        print("   No models found")
    
    # Shutdown
    await feature.shutdown()
    print("\n✅ Feature shutdown complete")


async def demo_2_cli_commands():
    """Demo 2: CLI commands demonstration"""
    print("\n" + "="*80)
    print("DEMO 2: CLI Commands Demonstration")
    print("="*80 + "\n")
    
    print("Custom Models CLI commands are available through:")
    print("  xencode models custom <command>")
    print()
    
    commands = {
        "analyze": {
            "description": "Analyze codebase to identify patterns",
            "example": "xencode models custom analyze ./my-project"
        },
        "train": {
            "description": "Train a custom model on your codebase",
            "example": "xencode models custom train my-model ./my-project"
        },
        "list": {
            "description": "List all custom models",
            "example": "xencode models custom list"
        },
        "performance": {
            "description": "Check model performance metrics",
            "example": "xencode models custom performance my-model"
        },
        "delete": {
            "description": "Delete a model or specific version",
            "example": "xencode models custom delete my-model"
        },
        "rollback": {
            "description": "Rollback to a previous model version",
            "example": "xencode models custom rollback my-model v20240115120000"
        },
        "compare": {
            "description": "Compare two model versions",
            "example": "xencode models custom compare my-model v1 v2"
        }
    }
    
    for cmd, info in commands.items():
        print(f"📌 {cmd}")
        print(f"   Description: {info['description']}")
        print(f"   Example: {info['example']}")
        print()
    
    print("To see all options for a command:")
    print("  xencode models custom <command> --help")


async def demo_3_feature_manager():
    """Demo 3: Feature manager integration"""
    print("\n" + "="*80)
    print("DEMO 3: Feature Manager Integration")
    print("="*80 + "\n")
    
    from xencode.features import FeatureManager
    
    # Create feature manager
    manager = FeatureManager()
    
    print("✅ Feature manager created")
    
    # Initialize custom models feature
    print("\n📦 Initializing custom_models feature...")
    success = await manager.initialize_feature("custom_models")
    
    if success:
        print("✅ Feature initialized successfully")
        
        # Get feature
        feature = manager.get_feature("custom_models")
        print(f"   Feature name: {feature.name}")
        print(f"   Feature status: {feature.get_status().value}")
        
        # Get CLI commands
        cli_commands = feature.get_cli_commands()
        print(f"\n📋 CLI commands available: {len(cli_commands)}")
        
        # Get TUI components
        tui_components = feature.get_tui_components()
        print(f"🖥️  TUI components available: {len(tui_components)}")
        
        # Shutdown feature
        print("\n🛑 Shutting down feature...")
        await manager.shutdown_feature("custom_models")
        print("✅ Feature shutdown complete")
    else:
        print("❌ Failed to initialize feature")


async def demo_4_model_workflow():
    """Demo 4: Complete model workflow"""
    print("\n" + "="*80)
    print("DEMO 4: Complete Model Workflow")
    print("="*80 + "\n")
    
    from xencode.features.custom_models import CustomModelManager, FeatureConfig
    
    # Initialize feature
    config = FeatureConfig(name="custom_models", enabled=True, config={})
    feature = CustomModelManager(config)
    await feature.initialize()
    
    print("✅ Feature initialized")
    
    # Step 1: Analyze codebase
    print("\n📊 Step 1: Analyze codebase")
    codebase_path = str(Path.cwd())
    
    try:
        analysis = await feature.analyze(codebase_path)
        print(f"✅ Found {analysis['total_patterns']} patterns")
    except Exception as e:
        print(f"⚠️  Analysis skipped: {e}")
        analysis = {'patterns': [], 'total_patterns': 0}
    
    # Step 2: Train model
    print("\n🎯 Step 2: Train custom model")
    
    try:
        training_result = await feature.train(
            model_name="workflow-demo",
            codebase_path=codebase_path,
            task_type="code_completion"
        )
        
        print(f"✅ Model trained: {training_result['model_name']}")
        print(f"   Version: {training_result['version']}")
        print(f"   Accuracy: {training_result['accuracy']:.1%}")
        
        model_name = training_result['model_name']
        version = training_result['version']
    except Exception as e:
        print(f"⚠️  Training skipped: {e}")
        model_name = None
        version = None
    
    # Step 3: Monitor performance
    if model_name and version:
        print("\n📈 Step 3: Monitor performance")
        
        try:
            # Record some metrics
            await feature.performance_monitor.record_metrics(
                model_name=model_name,
                version=version,
                speed_ms=150.5,
                accuracy=0.92,
                memory_mb=800.0,
                samples_tested=100
            )
            
            # Get metrics
            performance = await feature.monitor(model_name, version)
            
            if performance['metrics']:
                metrics = performance['metrics']
                print(f"✅ Performance metrics:")
                print(f"   Speed: {metrics['speed_ms']:.1f}ms")
                print(f"   Accuracy: {metrics['accuracy']:.1%}")
                print(f"   Memory: {metrics['memory_mb']:.1f}MB")
        except Exception as e:
            print(f"⚠️  Performance monitoring skipped: {e}")
    
    # Step 4: List all models
    print("\n📋 Step 4: List all models")
    models = await feature.list_models()
    
    print(f"✅ Total models: {len(models)}")
    for model in models:
        print(f"   • {model['name']}: {len(model['versions'])} versions")
    
    # Cleanup
    await feature.shutdown()
    print("\n✅ Workflow complete")


def demo_5_tui_components():
    """Demo 5: TUI components demonstration"""
    print("\n" + "="*80)
    print("DEMO 5: TUI Components Demonstration")
    print("="*80 + "\n")
    
    print("Custom Models TUI components:")
    print()
    
    components = {
        "CustomModelsPanel": {
            "description": "Main panel integrating all custom models features",
            "features": [
                "Model list browser",
                "Training dashboard",
                "Performance viewer",
                "Version history",
                "Codebase analysis results"
            ]
        },
        "TrainingDashboard": {
            "description": "Dashboard showing model training progress",
            "features": [
                "Training progress bar",
                "Epoch counter",
                "Accuracy metrics",
                "Training status"
            ]
        },
        "PerformanceViewer": {
            "description": "Viewer for model performance metrics",
            "features": [
                "Speed metrics",
                "Accuracy metrics",
                "Memory usage",
                "Samples tested"
            ]
        },
        "VersionHistory": {
            "description": "Panel showing version history",
            "features": [
                "Chronological version list",
                "Status indicators",
                "Accuracy tracking",
                "Current version highlighting"
            ]
        },
        "CodebaseAnalysisResults": {
            "description": "Panel showing codebase analysis results",
            "features": [
                "Pattern table",
                "Frequency counts",
                "Confidence scores",
                "Example display"
            ]
        }
    }
    
    for component, info in components.items():
        print(f"🖥️  {component}")
        print(f"   {info['description']}")
        print("   Features:")
        for feature in info['features']:
            print(f"   • {feature}")
        print()
    
    print("Keybindings:")
    print("  Ctrl+A: Analyze codebase")
    print("  Ctrl+T: Train model")
    print("  1: Show models list")
    print("  2: Show training dashboard")
    print("  3: Show performance metrics")
    print("  4: Show version history")
    print("  5: Show analysis results")
    print()
    
    print("Usage in Textual app:")
    print("""
from textual.app import App, ComposeResult
from xencode.tui.widgets.custom_models_panel import CustomModelsPanel

class MyApp(App):
    def compose(self) -> ComposeResult:
        yield CustomModelsPanel()
    
    def on_custom_models_panel_model_action(self, message):
        action = message.action
        model_name = message.model_name
        print(f"Action: {action}, Model: {model_name}")

app = MyApp()
app.run()
    """)


async def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("CUSTOM MODELS CLI AND TUI DEMO")
    print("="*80)
    
    demos = [
        ("Feature Usage", demo_1_feature_usage),
        ("CLI Commands", demo_2_cli_commands),
        ("Feature Manager", demo_3_feature_manager),
        ("Model Workflow", demo_4_model_workflow),
        ("TUI Components", demo_5_tui_components),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{'='*80}")
        print(f"Running Demo {i}/{len(demos)}: {name}")
        print(f"{'='*80}")
        
        try:
            if asyncio.iscoroutinefunction(demo_func):
                await demo_func()
            else:
                demo_func()
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(demos):
            print("\n" + "-"*80)
            input("Press Enter to continue to next demo...")
    
    print("\n" + "="*80)
    print("ALL DEMOS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Try the CLI commands: xencode models custom --help")
    print("2. Analyze your codebase: xencode models custom analyze ./your-project")
    print("3. Train a model: xencode models custom train my-model ./your-project")
    print("4. Use the TUI components in your Textual app")
    print("\nFor more information, see:")
    print("• xencode/features/custom_models/README.md")
    print("• examples/custom_models_demo.py")


if __name__ == "__main__":
    asyncio.run(main())
