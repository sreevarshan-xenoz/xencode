"""
Custom AI Models Feature Demo

Demonstrates how to use the Custom AI Models feature to analyze codebases,
train custom models, and monitor their performance.
"""

import asyncio

from xencode.features.base import FeatureConfig
from xencode.features.custom_models import CustomModelManager


async def main():
    """Demo of Custom AI Models feature"""

    # Create feature configuration
    config = FeatureConfig(
        name="custom_models",
        enabled=True,
        config={
            'enabled': True,
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

    # Initialize the feature
    manager = CustomModelManager(config)
    await manager.initialize()

    print("=" * 60)
    print("Custom AI Models Feature Demo")
    print("=" * 60)

    # Example 1: Analyze a codebase
    print("\n1. Analyzing codebase...")
    codebase_path = "xencode/features"  # Analyze the features directory

    try:
        analysis = await manager.analyze(codebase_path)
        print(f"   ✓ Found {analysis['total_patterns']} patterns")
        print(f"   ✓ Analyzed: {analysis['codebase_path']}")

        # Show some patterns
        if analysis['patterns']:
            print("\n   Sample patterns:")
            for pattern in analysis['patterns'][:3]:
                print(f"   - {pattern['pattern_type']}: {pattern['frequency']} occurrences")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 2: Train a custom model
    print("\n2. Training custom model...")

    try:
        training_result = await manager.train(
            model_name="my_code_assistant",
            codebase_path=codebase_path,
            task_type="code_completion"
        )

        print(f"   ✓ Model: {training_result['model_name']}")
        print(f"   ✓ Version: {training_result['version']}")
        print(f"   ✓ Status: {training_result['status']}")
        print(f"   ✓ Accuracy: {training_result['accuracy']:.2%}")
        print(f"   ✓ Training samples: {training_result['training_samples']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 3: List all models
    print("\n3. Listing all models...")

    try:
        models = await manager.list_models()
        print(f"   ✓ Total models: {len(models)}")

        for model in models:
            print(f"\n   Model: {model['name']}")
            print(f"   Current version: {model['current_version']}")
            print(f"   Total versions: {len(model['versions'])}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 4: Monitor model performance
    print("\n4. Monitoring model performance...")

    try:
        if models:
            model_name = models[0]['name']
            version = models[0]['current_version']

            # Record some sample metrics
            await manager.performance_monitor.record_metrics(
                model_name=model_name,
                version=version,
                speed_ms=150.0,
                accuracy=0.92,
                memory_mb=800.0,
                samples_tested=100
            )

            # Get performance report
            report = await manager.performance_monitor.generate_report(
                model_name=model_name,
                version=version
            )

            print(f"   ✓ Model: {report['model_name']}")
            print(f"   ✓ Version: {report['version']}")
            print(f"   ✓ Performance grade: {report['grade']}")
            print(f"   ✓ Speed: {report['metrics']['speed_ms']:.2f}ms")
            print(f"   ✓ Accuracy: {report['metrics']['accuracy']:.2%}")
            print(f"   ✓ Memory: {report['metrics']['memory_mb']:.2f}MB")

            if report['recommendations']:
                print("\n   Recommendations:")
                for rec in report['recommendations']:
                    print(f"   - {rec}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 5: Train another version
    print("\n5. Training a new version...")

    try:
        if models:
            model_name = models[0]['name']

            training_result = await manager.train(
                model_name=model_name,
                codebase_path=codebase_path,
                task_type="code_completion"
            )

            print(f"   ✓ New version: {training_result['version']}")
            print(f"   ✓ Accuracy: {training_result['accuracy']:.2%}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 6: Compare versions
    print("\n6. Comparing model versions...")

    try:
        if models and len(models[0]['versions']) >= 2:
            model_name = models[0]['name']
            versions = models[0]['versions']
            version1 = versions[0]['version']
            version2 = versions[1]['version']

            # Record metrics for second version
            await manager.performance_monitor.record_metrics(
                model_name=model_name,
                version=version2,
                speed_ms=120.0,
                accuracy=0.94,
                memory_mb=750.0,
                samples_tested=100
            )

            comparison = await manager.compare_versions(
                model_name=model_name,
                version1=version1,
                version2=version2
            )

            print(f"   ✓ Comparing {version1} vs {version2}")
            print(f"   ✓ Speed difference: {comparison['comparison']['speed_diff_ms']:.2f}ms")
            print(f"   ✓ Accuracy difference: {comparison['comparison']['accuracy_diff']:.2%}")
            print(f"   ✓ Memory difference: {comparison['comparison']['memory_diff_mb']:.2f}MB")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Cleanup
    await manager.shutdown()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
