# Terminal Assistant TUI Components

## Overview

This document describes the TUI (Text User Interface) components implemented for the Terminal Assistant feature. These components provide an interactive interface for command suggestions, explanations, error fixes, learning progress tracking, and command history browsing.

## Components

### 1. CommandSuggestionPanel

**Purpose**: Display context-aware command suggestions with scoring and explanations.

**Features**:
- Real-time command suggestions based on partial input
- Score-based ranking with visual indicators (⭐)
- Source attribution (history, context, pattern, temporal)
- Detailed explanations and reasons for suggestions
- Keyboard shortcuts for quick access

**Key Methods**:
- `update_suggestions(suggestions)`: Update the panel with new suggestions
- `on_input_changed()`: Handle real-time input changes

**Bindings**:
- `Ctrl+R`: Refresh suggestions
- `Enter`: Select current suggestion

### 2. CommandExplanationViewer

**Purpose**: Show detailed explanations for commands with arguments, examples, and warnings.

**Features**:
- Command description and purpose
- Argument explanations with descriptions
- Usage examples
- Safety warnings for dangerous commands
- Scrollable content for long explanations

**Key Methods**:
- `show_explanation(explanation)`: Display command explanation

**Data Structure**:
```python
{
    'command': str,
    'description': str,
    'arguments': [{'value': str, 'description': str}],
    'examples': [str],
    'warnings': [str]
}
```

### 3. ErrorFixPanel

**Purpose**: Display intelligent error fix suggestions with confidence scores.

**Features**:
- Error context display (command + error message)
- Multiple fix suggestions ranked by confidence
- Confidence indicators (🟢 high, 🟡 medium, 🔴 low)
- Category-based organization
- Sudo and installation requirement warnings
- Alternative commands and documentation links

**Key Methods**:
- `show_fixes(command, error, fixes)`: Display fix suggestions

**Data Structure**:
```python
{
    'fix': str,
    'explanation': str,
    'confidence': float,
    'category': str,
    'requires_sudo': bool,
    'requires_install': bool,
    'install_command': str,
    'documentation_url': str,
    'alternative_commands': [str]
}
```

### 4. LearningProgressTracker

**Purpose**: Track and display user's learning progress and skill levels.

**Features**:
- Overall statistics (commands learned, executions, mastered)
- Per-command skill levels (0-100%)
- Mastery tracking with progress indicators
- Usage frequency display
- Skill progression visualization

**Key Methods**:
- `update_progress(stats)`: Update learning statistics

**Data Structure**:
```python
{
    'total_commands_learned': int,
    'total_executions': int,
    'mastered_commands': [str],
    'skill_levels': {str: float},
    'learning_progress': {
        str: {
            'total_uses': int,
            'successful_uses': int,
            'mastery_level': float
        }
    }
}
```

### 5. CommandHistoryBrowser

**Purpose**: Browse and search command history with filtering capabilities.

**Features**:
- Chronological command history display
- Real-time search/filter functionality
- Success/failure indicators (✅/❌)
- Timestamp display
- Project context information
- Last 50 commands display

**Key Methods**:
- `update_history(history)`: Update history display
- `_filter_history(pattern)`: Filter history by pattern

**Bindings**:
- `Ctrl+F`: Focus search input

**Data Structure**:
```python
{
    'command': str,
    'timestamp': str,  # ISO format
    'success': bool,
    'context': {
        'project_type': str,
        'directory': str
    }
}
```

### 6. TerminalAssistantPanel (Main Panel)

**Purpose**: Main container panel with tabbed interface for all components.

**Features**:
- Tabbed navigation between all components
- Keyboard shortcuts for quick tab switching
- Unified interface for all Terminal Assistant features
- Automatic tab switching on certain actions

**Tabs**:
1. Suggestions - Command suggestions panel
2. Explanation - Command explanation viewer
3. Fixes - Error fix suggestions panel
4. Progress - Learning progress tracker
5. History - Command history browser

**Key Methods**:
- `update_suggestions(suggestions)`: Update suggestions tab
- `show_explanation(explanation)`: Show explanation and switch to tab
- `show_error_fixes(command, error, fixes)`: Show fixes and switch to tab
- `update_learning_progress(stats)`: Update progress tab
- `update_command_history(history)`: Update history tab

**Bindings**:
- `1`: Show Suggestions tab
- `2`: Show Explanation tab
- `3`: Show Fixes tab
- `4`: Show Progress tab
- `5`: Show History tab

## Usage Example

```python
from xencode.tui.widgets.terminal_assistant_panel import TerminalAssistantPanel

# Create main panel
panel = TerminalAssistantPanel()

# Update suggestions
suggestions = [
    {
        'command': 'git status',
        'score': 15.5,
        'source': 'history',
        'explanation': 'Check repository status',
        'reason': 'Frequently used after commits'
    }
]
panel.update_suggestions(suggestions)

# Show command explanation
explanation = {
    'command': 'git commit -m "message"',
    'description': 'Commit changes to repository',
    'arguments': [
        {'value': '-m', 'description': 'Commit message'}
    ],
    'examples': [
        'git commit -m "Initial commit"',
        'git commit -m "Fix bug #123"'
    ],
    'warnings': [
        '⚠️  Make sure to stage files first with git add'
    ]
}
panel.show_explanation(explanation)

# Show error fixes
fixes = [
    {
        'fix': 'npm install',
        'explanation': 'Install missing dependencies',
        'confidence': 0.95,
        'category': 'dependency',
        'requires_sudo': False,
        'requires_install': False
    }
]
panel.show_error_fixes("npm start", "Module not found", fixes)

# Update learning progress
stats = {
    'total_commands_learned': 25,
    'total_executions': 150,
    'mastered_commands': ['git', 'npm'],
    'skill_levels': {
        'git': 0.85,
        'npm': 0.75
    },
    'learning_progress': {
        'git': {
            'total_uses': 50,
            'successful_uses': 48,
            'mastery_level': 0.85
        }
    }
}
panel.update_learning_progress(stats)

# Update command history
history = [
    {
        'command': 'git status',
        'timestamp': '2024-01-01T10:00:00',
        'success': True,
        'context': {'project_type': 'python'}
    }
]
panel.update_command_history(history)
```

## Integration with Terminal Assistant Feature

The TUI components integrate with the Terminal Assistant feature through the following flow:

1. **Command Input** → CommandSuggestionPanel displays suggestions
2. **Suggestion Selection** → CommandExplanationViewer shows details
3. **Command Execution** → If error occurs, ErrorFixPanel shows fixes
4. **Learning** → LearningProgressTracker updates skill levels
5. **History** → CommandHistoryBrowser records all commands

## Styling

All components use Textual's CSS-like styling system with:
- Consistent color scheme (primary, accent, success, warning, error)
- Responsive layouts
- Hover effects for interactive elements
- Severity-based color coding
- Clear visual hierarchy

## Testing

Comprehensive test suite in `tests/tui/test_terminal_assistant_panel.py`:
- Unit tests for each component
- Data handling tests
- Integration tests
- Keyboard shortcut tests
- Filter logic tests

All 26 tests passing ✅

## Files Created

1. `xencode/tui/widgets/terminal_assistant_panel.py` - Main TUI components (730 lines)
2. `tests/tui/test_terminal_assistant_panel.py` - Comprehensive test suite (430 lines)
3. `xencode/tui/widgets/TERMINAL_ASSISTANT_TUI.md` - This documentation

## Next Steps

1. Integrate TUI components with Terminal Assistant feature backend
2. Add real-time updates from command execution
3. Implement persistent state across sessions
4. Add export functionality for history and progress
5. Create example applications demonstrating usage
