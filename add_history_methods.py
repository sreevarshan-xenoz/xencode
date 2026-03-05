#!/usr/bin/env python3
"""Add get_history and rollback methods to ConflictResolver"""

# Read the file
with open('xencode/features/collaborative_coding.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where to insert (after _record_resolution method, before _last_write_wins)
insert_index = None
for i, line in enumerate(lines):
    if 'async def _last_write_wins' in line and i > 750:
        insert_index = i
        break

if insert_index is None:
    print("Could not find insertion point")
    exit(1)

# Methods to insert
indent = ' ' * 4
methods = [
    f'\n',
    f'{indent}async def get_history(self, session_id: str, limit: int = 50) -> Dict[str, Any]:\n',
    f'{indent}    """Get edit history for a session"""\n',
    f'{indent}    if session_id not in self.edit_history:\n',
    f'{indent}        return {{\n',
    f'{indent}            \'success\': True,\n',
    f'{indent}            \'history\': [],\n',
    f'{indent}            \'count\': 0\n',
    f'{indent}        }}\n',
    f'{indent}    \n',
    f'{indent}    history = self.edit_history[session_id][-limit:]\n',
    f'{indent}    \n',
    f'{indent}    return {{\n',
    f'{indent}        \'success\': True,\n',
    f'{indent}        \'history\': history,\n',
    f'{indent}        \'count\': len(history)\n',
    f'{indent}    }}\n',
    f'\n',
    f'{indent}async def rollback(self, session_id: str, steps: int = 1) -> Dict[str, Any]:\n',
    f'{indent}    """\n',
    f'{indent}    Rollback edit history by specified steps\n',
    f'{indent}    \n',
    f'{indent}    Args:\n',
    f'{indent}        session_id: Session to rollback\n',
    f'{indent}        steps: Number of resolutions to rollback\n',
    f'{indent}    \n',
    f'{indent}    Returns:\n',
    f'{indent}        Dict with rollback information\n',
    f'{indent}    """\n',
    f'{indent}    if session_id not in self.edit_history:\n',
    f'{indent}        return {{\n',
    f'{indent}            \'success\': False,\n',
    f'{indent}            \'error\': \'No history found for session\'\n',
    f'{indent}        }}\n',
    f'{indent}    \n',
    f'{indent}    history = self.edit_history[session_id]\n',
    f'{indent}    \n',
    f'{indent}    if steps > len(history):\n',
    f'{indent}        return {{\n',
    f'{indent}            \'success\': False,\n',
    f'{indent}            \'error\': f\'Cannot rollback {{steps}} steps, only {{len(history)}} available\'\n',
    f'{indent}        }}\n',
    f'{indent}    \n',
    f'{indent}    # Get entries to rollback\n',
    f'{indent}    rollback_entries = history[-steps:]\n',
    f'{indent}    \n',
    f'{indent}    # Remove from history\n',
    f'{indent}    self.edit_history[session_id] = history[:-steps]\n',
    f'{indent}    \n',
    f'{indent}    return {{\n',
    f'{indent}        \'success\': True,\n',
    f'{indent}        \'rolled_back\': rollback_entries,\n',
    f'{indent}        \'steps\': steps,\n',
    f'{indent}        \'remaining_history\': len(self.edit_history[session_id])\n',
    f'{indent}    }}\n',
    f'\n',
]

# Insert the methods
lines[insert_index:insert_index] = methods

# Write back
with open('xencode/features/collaborative_coding.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Added get_history and rollback methods before line {insert_index+1}")
