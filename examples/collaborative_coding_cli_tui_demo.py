#!/usr/bin/env python3
"""
Collaborative Coding CLI and TUI Demo

This demo shows how to use the collaborative coding feature through CLI commands
and TUI interface.

CLI Commands:
    xencode collab start <room_name> [--username USER] [--password PASS]
    xencode collab join <session_id> [--username USER] [--password PASS]
    xencode collab leave <session_id> --user-id USER_ID
    xencode collab list [--user-id USER_ID]
    xencode collab users <session_id>

TUI Components:
    - CollaborativeCodingPanel: Main panel with all features
    - PresenceIndicator: Shows active users
    - CursorTracker: Tracks user cursor positions
    - ChatInterface: Real-time chat
    - ConflictResolutionPanel: Handles merge conflicts
    - RealTimeEditor: Collaborative editor

Usage Examples:
    1. Start a new session:
       $ xencode collab start "My Project" --username "Alice"

    2. Join an existing session:
       $ xencode collab join abc123-def456 --username "Bob"

    3. List active sessions:
       $ xencode collab list

    4. View users in a session:
       $ xencode collab users abc123-def456
"""

import asyncio

from xencode.features.base import FeatureConfig
from xencode.features.collaborative_coding import CollaborativeCodingFeature


async def demo_cli_commands():
    """Demonstrate CLI command functionality"""
    print("=" * 60)
    print("Collaborative Coding CLI Demo")
    print("=" * 60)

    # Create feature instance
    config = FeatureConfig(
        name="collaborative_coding",
        enabled=True,
        config={}
    )
    feature = CollaborativeCodingFeature(config)

    # Initialize feature
    await feature.initialize()

    print("\n1. Starting a collaboration session...")
    result = await feature.start("Demo Room", "user-123", "Alice")
    if result['success']:
        session_id = result['session']['session_id']
        print(f"   ✅ Session started: {session_id}")
        print(f"   Room: {result['session']['name']}")
        print("   Owner: Alice")

    print("\n2. Joining the session...")
    result = await feature.join(session_id, "user-456", "Bob")
    if result['success']:
        print("   ✅ Bob joined the session")
        print(f"   Participants: {result['session']['participant_count']}")

    print("\n3. Listing active sessions...")
    result = await feature.list_sessions()
    if result['success']:
        print(f"   ✅ Found {result['count']} active session(s)")
        for session in result['sessions']:
            print(f"      - {session['name']} ({session['participant_count']} participants)")

    print("\n4. Getting session info...")
    result = await feature.get_session_info(session_id)
    if result['success']:
        print("   ✅ Session info retrieved")
        print("      Participants:")
        for _uid, user in result['session']['participants'].items():
            status = "🟢" if user['is_active'] else "⚫"
            print(f"        {status} {user['username']}")

    print("\n5. Leaving the session...")
    result = await feature.leave(session_id, "user-456")
    if result['success']:
        print("   ✅ Bob left the session")

    # Shutdown feature
    await feature.shutdown()

    print("\n" + "=" * 60)
    print("CLI Demo Complete!")
    print("=" * 60)


def demo_tui_components():
    """Demonstrate TUI component functionality"""
    print("\n" + "=" * 60)
    print("Collaborative Coding TUI Demo")
    print("=" * 60)

    from xencode.tui.widgets.collaborative_coding_panel import (
        ChatInterface,
        CollaborativeCodingPanel,
        ConflictResolutionPanel,
        CursorTracker,
        PresenceIndicator,
        RealTimeEditor,
    )

    print("\n1. TUI Components Available:")
    print("   - CollaborativeCodingPanel: Main panel")
    print("   - PresenceIndicator: User presence tracking")
    print("   - CursorTracker: Cursor position tracking")
    print("   - ChatInterface: Real-time chat")
    print("   - ConflictResolutionPanel: Conflict resolution")
    print("   - RealTimeEditor: Collaborative editor")

    print("\n2. Creating TUI components...")

    # Create presence indicator
    PresenceIndicator()
    print("   ✅ PresenceIndicator created")

    # Create cursor tracker
    CursorTracker()
    print("   ✅ CursorTracker created")

    # Create chat interface
    ChatInterface()
    print("   ✅ ChatInterface created")

    # Create conflict resolution panel
    ConflictResolutionPanel()
    print("   ✅ ConflictResolutionPanel created")

    # Create real-time editor
    RealTimeEditor()
    print("   ✅ RealTimeEditor created")

    # Create main panel
    CollaborativeCodingPanel()
    print("   ✅ CollaborativeCodingPanel created")

    print("\n3. Component Features:")
    print("   PresenceIndicator:")
    print("      - add_user(user_id, username, color, status)")
    print("      - update_user_status(user_id, status)")
    print("      - remove_user(user_id)")

    print("\n   CursorTracker:")
    print("      - update_cursor(user_id, username, position, file_path)")

    print("\n   ChatInterface:")
    print("      - add_message(username, text, is_self, is_system)")
    print("      - SendMessage event for sending messages")

    print("\n   ConflictResolutionPanel:")
    print("      - show_conflicts(conflicts)")
    print("      - hide_conflicts()")
    print("      - ResolveConflict event for resolution")

    print("\n   RealTimeEditor:")
    print("      - update_content(content, version)")
    print("      - ContentChanged event for edits")

    print("\n   CollaborativeCodingPanel:")
    print("      - Integrates all components")
    print("      - Keyboard shortcuts: Ctrl+S (sync), Ctrl+U (users), Ctrl+M (chat)")
    print("      - Session management: start, join, leave")

    print("\n" + "=" * 60)
    print("TUI Demo Complete!")
    print("=" * 60)


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("COLLABORATIVE CODING CLI & TUI DEMO")
    print("=" * 60)

    # Run CLI demo
    asyncio.run(demo_cli_commands())

    # Run TUI demo
    demo_tui_components()

    print("\n" + "=" * 60)
    print("All Demos Complete!")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  - xencode/features/collaborative_coding.py")
    print("  - xencode/tui/widgets/collaborative_coding_panel.py")
    print("  - tests/features/test_collaborative_coding_cli_tui.py")
    print()


if __name__ == '__main__':
    main()
