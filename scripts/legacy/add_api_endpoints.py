#!/usr/bin/env python3
"""Add missing API endpoints to get_api_endpoints"""

# Read the file
with open('xencode/features/collaborative_coding.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the end of get_api_endpoints (look for the closing bracket and return statement)
insert_index = None
for i, line in enumerate(lines):
    if i > 1560 and i < 1700 and line.strip() == ']':
        # This might be the end of the endpoints list
        # Check if next few lines have 'def get_cli_commands' or similar
        for j in range(i+1, min(i+10, len(lines))):
            if 'def get_cli_commands' in lines[j] or 'def get_tui_components' in lines[j]:
                insert_index = i
                break
        if insert_index:
            break

if insert_index is None:
    print("Could not find insertion point")
    exit(1)

# Endpoints to insert (before the closing bracket)
indent = ' ' * 12
endpoints = [
    f'{indent}{{\n',
    f'{indent}    \'path\': \'/api/collab/history/{{session_id}}\',\n',
    f'{indent}    \'method\': \'GET\',\n',
    f'{indent}    \'handler\': self.get_edit_history\n',
    f'{indent}}},\n',
    f'{indent}{{\n',
    f'{indent}    \'path\': \'/api/collab/rollback\',\n',
    f'{indent}    \'method\': \'POST\',\n',
    f'{indent}    \'handler\': self.rollback_edits\n',
    f'{indent}}},\n',
]

# Insert the endpoints before the closing bracket
lines[insert_index:insert_index] = endpoints

# Write back
with open('xencode/features/collaborative_coding.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Added history and rollback API endpoints before line {insert_index+1}")
