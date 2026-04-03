# Xencode TUI Feature Panels

This directory contains feature-specific TUI panels for the Xencode platform. Each panel provides a consistent, high-performance interface for interacting with Xencode features.

## Architecture

### Base Feature Panel

All feature panels inherit from `BaseFeaturePanel`, which provides:

- **Status Indicator**: Shows feature status (enabled/disabled/loading)
- **Title Bar**: Displays feature name with icon
- **Content Area**: Scrollable area for feature-specific content
- **Consistent Styling**: Unified look and feel across all features
- **60 FPS Rendering**: Optimized for smooth performance

### Feature Panels

#### 1. Project Analyzer Panel (F2)
**Location**: `project_analyzer_panel.py`

Provides project analysis and documentation generation:
- Analyze project structure
- Generate documentation
- Health checks
- Metrics tracking

**Key Components**:
- `ProjectMetricsCard`: Displays project metrics
- Project structure tree viewer
- Analysis controls

#### 2. Learning Mode Panel (F3)
**Location**: `learning_mode_panel.py`

Interactive learning and tutorials:
- Browse learning topics
- Track progress
- Interactive tutorials
- Adaptive difficulty

**Key Components**:
- `TopicCard`: Learning topic with progress
- Tutorial viewer
- Progress tracking

#### 3. Multi-language Panel (F4)
**Location**: `multi_language_panel.py`

Multi-language support and translation:
- Language selection
- Auto-detection
- Translation tools
- RTL support

**Key Components**:
- `LanguageCard`: Language option with native name
- Current language indicator
- Language list viewer

#### 4. Custom Models Panel (F5)
**Location**: `custom_models_panel.py`

Custom AI model management:
- Analyze codebase
- Train custom models
- Model versioning
- Performance metrics

**Key Components**:
- `ModelCard`: Custom model with accuracy
- Training progress indicator
- Model list viewer

#### 5. Security Auditor Panel (F6)
**Location**: `security_auditor_panel.py`

Security auditing and vulnerability scanning:
- Code vulnerability scanning
- Dependency checks
- Security reports
- Risk assessment

**Key Components**:
- `VulnerabilityCard`: Security issue with severity
- Security summary
- Vulnerability list

#### 6. Performance Profiler Panel (F7)
**Location**: `performance_profiler_panel.py`

Performance profiling and optimization:
- Run performance profiles
- Identify bottlenecks
- Optimization suggestions
- Before/after comparisons

**Key Components**:
- `BottleneckCard`: Performance bottleneck
- Profile summary
- Bottleneck list

## Integration with Main TUI

### Keybindings

Feature panels are accessible via function keys:

```
F2 - Project Analyzer
F3 - Learning Mode
F4 - Multi-language Support
F5 - Custom Models
F6 - Security Auditor
F7 - Performance Profiler
```

Existing feature panels:
```
Ctrl+R - Code Review
Ctrl+Y - Terminal Assistant
Ctrl+P - Performance Dashboard
```

### Panel Toggle System

The main TUI app uses a helper method `_toggle_feature_panel()` to manage panel visibility:

1. Hides all other panels
2. Toggles target panel visibility
3. Adjusts chat panel size

This ensures only one feature panel is visible at a time, maintaining a clean interface.

## UI Patterns

### Consistent Layout

All feature panels follow this structure:

```
┌─────────────────────────────────────┐
│ ✓ Feature Name: ENABLED             │ ← Status Indicator
├─────────────────────────────────────┤
│ 🎯 Feature Title                    │ ← Title Bar
├─────────────────────────────────────┤
│                                     │
│  [Button] [Button] [Button]         │ ← Controls
│                                     │
│  Content Area                       │ ← Scrollable Content
│  - Cards                            │
│  - Lists                            │
│  - Tables                           │
│                                     │
└─────────────────────────────────────┘
```

### Status States

- **Enabled** (✓): Feature is active and ready
- **Disabled** (✗): Feature is inactive
- **Loading** (⟳): Feature is processing

### Color Coding

- **Primary**: Main UI elements
- **Success**: Positive states, enabled features
- **Warning**: Caution states, in-progress operations
- **Error**: Critical issues, disabled features
- **Accent**: Highlights, important information

## Performance Considerations

### 60 FPS Rendering

All panels are optimized for 60 FPS rendering:

1. **Efficient Updates**: Only update changed components
2. **Lazy Loading**: Load content on demand
3. **Virtualization**: Use scrollable containers for large lists
4. **Reactive Properties**: Use Textual's reactive system for state management

### Memory Management

- Panels are created once during app initialization
- Content is cleared when panels are hidden
- Large datasets use pagination or virtualization

## Adding New Feature Panels

To add a new feature panel:

1. **Create Panel Class**:
   ```python
   from .base_feature_panel import BaseFeaturePanel
   
   class MyFeaturePanel(BaseFeaturePanel):
       def __init__(self, *args, **kwargs):
           super().__init__(
               feature_name="my_feature",
               title="🎯 My Feature",
               *args,
               **kwargs
           )
   ```

2. **Add to `__init__.py`**:
   ```python
   from .my_feature_panel import MyFeaturePanel
   __all__ = [..., "MyFeaturePanel"]
   ```

3. **Import in `app.py`**:
   ```python
   from xencode.tui.features.my_feature_panel import MyFeaturePanel
   ```

4. **Add to Compose Method**:
   ```python
   with Vertical(id="my-feature-panel-container", classes="hidden"):
       self.my_feature_panel = MyFeaturePanel()
       yield self.my_feature_panel
   ```

5. **Add Keybinding**:
   ```python
   Binding("f8", "toggle_my_feature", "My Feature"),
   ```

6. **Add Toggle Action**:
   ```python
   def action_toggle_my_feature(self) -> None:
       """Toggle my feature panel visibility."""
       self._toggle_feature_panel("my-feature-panel-container")
   ```

## Testing

Test feature panels with:

```bash
# Run TUI
python -m xencode.tui

# Test specific panel
# Press F2-F7 to open feature panels
# Verify:
# - Panel opens/closes correctly
# - Status indicator updates
# - Controls work as expected
# - Content renders properly
# - No performance issues
```

## Future Enhancements

1. **Feature Navigator**: Sidebar for quick feature access
2. **Panel Layouts**: Support for split-screen feature panels
3. **Customization**: User-configurable panel layouts
4. **Themes**: Feature-specific color themes
5. **Shortcuts**: Customizable keybindings per feature
6. **State Persistence**: Remember panel states across sessions

## Requirements

- Technical Requirements - Integration Points
- TUI: 60 FPS rendering
- Consistent UI patterns across all features
- Feature navigation and status indicators
