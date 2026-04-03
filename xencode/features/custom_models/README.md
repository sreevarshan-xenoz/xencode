# Custom AI Models Feature

Fine-tune AI models on your codebase for personalized assistance.

## Overview

The Custom AI Models feature allows you to:
- Analyze your codebase to identify coding patterns and style
- Train custom AI models based on your coding preferences
- Monitor model performance with detailed metrics
- Manage multiple model versions with rollback support
- Compare different model versions

## Features

### 1. Codebase Analysis (Requirement 7.1)

Analyze your codebase to extract patterns:
- File structure patterns
- Coding style (indentation, formatting)
- Naming conventions (snake_case, camelCase, PascalCase)
- Common imports and dependencies

### 2. Model Training (Requirements 7.2, 7.3)

Train custom models on your coding style:
- Fine-tune models for specific tasks (code completion, review, refactoring)
- Create task-specific model variants
- Configurable training parameters (epochs, batch size)
- Automatic accuracy validation

### 3. Version Management (Requirement 7.4)

Manage model versions:
- Create multiple versions of the same model
- Rollback to previous versions
- Compare versions side-by-side
- Archive old versions

### 4. Performance Monitoring (Requirement 7.5)

Track model performance:
- Speed metrics (inference time in ms)
- Accuracy metrics (0-1 scale)
- Memory usage (MB)
- Performance grades (A-F)
- Optimization recommendations

## Usage

### Basic Example

```python
from xencode.features.custom_models import CustomModelManager
from xencode.features.base import FeatureConfig

# Create configuration
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

# Initialize manager
manager = CustomModelManager(config)
await manager.initialize()

# Analyze codebase
analysis = await manager.analyze("path/to/codebase")
print(f"Found {analysis['total_patterns']} patterns")

# Train a model
result = await manager.train(
    model_name="my_assistant",
    codebase_path="path/to/codebase",
    task_type="code_completion"
)
print(f"Model trained with {result['accuracy']:.2%} accuracy")

# Monitor performance
metrics = await manager.monitor("my_assistant")
print(f"Speed: {metrics['metrics']['speed_ms']}ms")

# Cleanup
await manager.shutdown()
```

## Configuration

```yaml
custom_models:
  enabled: true
  max_models: 10
  training:
    max_epochs: 10
    batch_size: 32
  performance:
    min_accuracy: 0.85
  auto_backup: true
```

## API Endpoints

- `POST /api/models/custom/analyze` - Analyze codebase
- `POST /api/models/custom/train` - Train custom model
- `GET /api/models/custom/monitor` - Get performance metrics
- `GET /api/models/custom/list` - List all models
- `DELETE /api/models/custom/delete` - Delete model
- `POST /api/models/custom/rollback` - Rollback to version
- `GET /api/models/custom/compare` - Compare versions

## Task Types

- `code_completion` - Code completion and suggestions
- `code_review` - Code review and quality checks
- `refactoring` - Code refactoring suggestions
- `documentation` - Documentation generation
- `bug_detection` - Bug detection and fixes

## Performance Grades

Models are graded based on speed, accuracy, and memory usage:

- **A**: Excellent performance (90-100 points)
- **B**: Good performance (80-89 points)
- **C**: Acceptable performance (70-79 points)
- **D**: Below target performance (60-69 points)
- **F**: Poor performance (<60 points)

## Storage

Models and metrics are stored in:
- `~/.xencode/custom_models.json` - Model metadata
- `~/.xencode/model_metrics.json` - Performance metrics

## Examples

See `examples/custom_models_demo.py` for a complete demonstration.

## Testing

Run tests with:
```bash
pytest tests/features/test_custom_models.py -v
```

## Requirements Mapping

- **7.1**: Codebase analysis with pattern extraction
- **7.2**: Model fine-tuning on coding style
- **7.3**: Custom model variants for specific tasks
- **7.4**: Model versioning and rollback support
- **7.5**: Performance metrics and monitoring
