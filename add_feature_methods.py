#!/usr/bin/env python3
"""Add get_edit_history and rollback_edits methods to CollaborativeCodingFeature"""

# Read the file
with open('xencode/features/collaborative_coding.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where to insert (after resolve_conflicts method in CollaborativeCodingFeature)
insert_index = None
for i, line in enumerate(lines):
    if 'async def resolve_conflicts' in line and i > 1200:
        # Find the end of this method
        for j in range(i+1, len(lines)):
            if lines[j].strip() and not lines[j].startswith(' ' * 8) and not lines[j].strip().startswith('#'):
                insert_index = j
                break
        break

if insert_index is None:
    print("Could not find insertion point")
    # Try finding get_cli_commands instead
    for i, line in enumerate(lines):
        if 'def get_cli_commands' in line and i > 1200:
            insert_index = i
            break

if insert_index is None:
    print("Still could not find insertion point")
    exit(1)

# Methods to insert
indent = ' ' * 4
methods = [
    f'\n',
    f'{indent}async def get_edit_history(self, session_id: str, limit: int = 50) -> Dict[str, Any]:\n',
    f'{indent}    """Get edit history for a session"""\n',
    f'{indent}    if not self.resolver:\n',
    f'{indent}        return {{\n',
    f'{indent}            \'success\': False,\n',
    f'{indent}            \'error\': \'Conflict resolver not initialized\'\n',
    f'{indent}        }}\n',
    f'{indent}    \n',
    f'{indent}    return await self.resolver.get_history(session_id, limit)\n',
    f'\n',
    f'{indent}async def rollback_edits(self, session_id: str, steps: int = 1) -> Dict[str, Any]:\n',
    f'{indent}    """Rollback edit history by specified steps"""\n',
    f'{indent}    if not self.resolver:\n',
    f'{indent}        return {{\n',
    f'{indent}            \'success\': False,\n',
    f'{indent}            \'error\': \'Conflict resolver not initialized\'\n',
    f'{indent}        }}\n',
    f'{indent}    \n',
    f'{indent}    return await self.resolver.rollback(session_id, steps)\n',
    f'\n',
]

# Insert the methods
lines[insert_index:insert_index] = methods

# Write back
with open('xencode/features/collaborative_coding.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Added get_edit_history and rollback_edits methods before line {insert_index+1}")
