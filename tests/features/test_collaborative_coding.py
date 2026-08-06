"""
Unit tests for Collaborative Coding feature
"""


import pytest

from xencode.features.base import FeatureConfig
from xencode.features.collaborative_coding import (
    Change,
    ChangeType,
    CollaborationManager,
    CollaborativeCodingFeature,
    CollaborativeConfig,
    ConflictResolver,
    ConnectionState,
    Session,
    User,
    WebSocketEditor,
)


@pytest.fixture
def collab_config():
    """Create test configuration"""
    return CollaborativeConfig(
        enabled=True,
        websocket_port=8765,
        websocket_host='localhost',
        max_connections=5,
        sync_debounce_ms=50,
        batch_size=10
    )


@pytest.fixture
def manager(collab_config):
    """Create collaboration manager"""
    return CollaborationManager(collab_config)


@pytest.fixture
def editor(collab_config, manager):
    """Create WebSocket editor"""
    return WebSocketEditor(collab_config, manager)


@pytest.fixture
def resolver():
    """Create conflict resolver"""
    return ConflictResolver()


@pytest.fixture
def feature_config():
    """Create feature config"""
    return FeatureConfig(
        name='collaborative_coding',
        enabled=True,
        config={
            'websocket_port': 8765,
            'max_connections': 5
        }
    )


class TestCollaborationManager:
    """Test CollaborationManager class"""

    @pytest.mark.asyncio
    async def test_start_session(self, manager):
        """Test starting a new session"""
        result = await manager.start_session(
            name='Test Session',
            owner_id='user1',
            username='Alice'
        )

        assert result['success'] is True
        assert 'session_id' in result
        assert result['session']['name'] == 'Test Session'
        assert result['session']['owner_id'] == 'user1'
        assert len(result['session']['participants']) == 1
        assert result['user']['username'] == 'Alice'

    @pytest.mark.asyncio
    async def test_join_session(self, manager):
        """Test joining an existing session"""
        # Start a session
        start_result = await manager.start_session(
            name='Test Session',
            owner_id='user1',
            username='Alice'
        )
        session_id = start_result['session_id']

        # Join the session
        join_result = await manager.join_session(
            session_id=session_id,
            user_id='user2',
            username='Bob'
        )

        assert join_result['success'] is True
        assert join_result['session_id'] == session_id
        assert len(join_result['session']['participants']) == 2
        assert join_result['user']['username'] == 'Bob'

    @pytest.mark.asyncio
    async def test_join_nonexistent_session(self, manager):
        """Test joining a session that doesn't exist"""
        result = await manager.join_session(
            session_id='nonexistent',
            user_id='user1',
            username='Alice'
        )

        assert result['success'] is False
        assert 'not found' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_join_full_session(self, manager):
        """Test joining a session that is full"""
        # Start a session
        start_result = await manager.start_session(
            name='Test Session',
            owner_id='user1',
            username='Alice'
        )
        session_id = start_result['session_id']

        # Fill the session
        for i in range(2, manager.config.max_connections + 1):
            await manager.join_session(
                session_id=session_id,
                user_id=f'user{i}',
                username=f'User{i}'
            )

        # Try to join when full
        result = await manager.join_session(
            session_id=session_id,
            user_id='user_extra',
            username='Extra'
        )

        assert result['success'] is False
        assert 'full' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_leave_session(self, manager):
        """Test leaving a session"""
        # Start and join
        start_result = await manager.start_session(
            name='Test Session',
            owner_id='user1',
            username='Alice'
        )
        session_id = start_result['session_id']

        await manager.join_session(
            session_id=session_id,
            user_id='user2',
            username='Bob'
        )

        # Leave session
        result = await manager.leave_session(session_id, 'user2')

        assert result['success'] is True
        assert result['session_closed'] is False

    @pytest.mark.asyncio
    async def test_owner_leaves_closes_session(self, manager):
        """Test that session closes when owner leaves"""
        start_result = await manager.start_session(
            name='Test Session',
            owner_id='user1',
            username='Alice'
        )
        session_id = start_result['session_id']

        # Owner leaves
        result = await manager.leave_session(session_id, 'user1')

        assert result['success'] is True
        assert result['session_closed'] is True

    @pytest.mark.asyncio
    async def test_sync_change_insert(self, manager):
        """Test synchronizing an insert change"""
        # Start session
        start_result = await manager.start_session(
            name='Test Session',
            owner_id='user1',
            username='Alice'
        )
        session_id = start_result['session_id']

        # Sync a change
        change_data = {
            'change_type': 'insert',
            'position': 0,
            'content': 'Hello'
        }

        result = await manager.sync_change(session_id, 'user1', change_data)

        assert result['success'] is True
        assert 'change' in result
        assert result['change']['content'] == 'Hello'
        assert result['document_version'] == 1

        # Verify document content
        session = manager.sessions[session_id]
        assert session.document_content == 'Hello'

    @pytest.mark.asyncio
    async def test_sync_change_delete(self, manager):
        """Test synchronizing a delete change"""
        # Start session with content
        start_result = await manager.start_session(
            name='Test Session',
            owner_id='user1',
            username='Alice'
        )
        session_id = start_result['session_id']

        # Add initial content
        session = manager.sessions[session_id]
        session.document_content = 'Hello World'

        # Delete 'World'
        change_data = {
            'change_type': 'delete',
            'position': 6,
            'content': 'World'
        }

        result = await manager.sync_change(session_id, 'user1', change_data)

        assert result['success'] is True
        assert session.document_content == 'Hello '

    @pytest.mark.asyncio
    async def test_update_cursor(self, manager):
        """Test updating cursor position"""
        # Start session
        start_result = await manager.start_session(
            name='Test Session',
            owner_id='user1',
            username='Alice'
        )
        session_id = start_result['session_id']

        # Update cursor
        result = await manager.update_cursor(
            session_id=session_id,
            user_id='user1',
            cursor_position=10,
            selection_start=5,
            selection_end=10
        )

        assert result['success'] is True
        assert result['user']['cursor_position'] == 10
        assert result['user']['selection_start'] == 5
        assert result['user']['selection_end'] == 10

    @pytest.mark.asyncio
    async def test_get_session_info(self, manager):
        """Test getting session information"""
        # Start session
        start_result = await manager.start_session(
            name='Test Session',
            owner_id='user1',
            username='Alice'
        )
        session_id = start_result['session_id']

        # Get info
        result = await manager.get_session_info(session_id)

        assert result['success'] is True
        assert result['session']['session_id'] == session_id
        assert result['session']['name'] == 'Test Session'

    @pytest.mark.asyncio
    async def test_list_sessions(self, manager):
        """Test listing all sessions"""
        # Start multiple sessions
        await manager.start_session('Session 1', 'user1', 'Alice')
        await manager.start_session('Session 2', 'user2', 'Bob')

        # List all sessions
        result = await manager.list_sessions()

        assert result['success'] is True
        assert result['count'] == 2
        assert len(result['sessions']) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_filtered_by_user(self, manager):
        """Test listing sessions filtered by user"""
        # Start sessions
        await manager.start_session('Session 1', 'user1', 'Alice')
        session2 = await manager.start_session('Session 2', 'user2', 'Bob')

        # User1 joins session2
        await manager.join_session(session2['session_id'], 'user1', 'Alice')

        # List sessions for user1
        result = await manager.list_sessions(user_id='user1')

        assert result['success'] is True
        assert result['count'] == 2  # user1 is in both sessions

    @pytest.mark.asyncio
    async def test_close_session(self, manager):
        """Test closing a session"""
        # Start session
        start_result = await manager.start_session(
            name='Test Session',
            owner_id='user1',
            username='Alice'
        )
        session_id = start_result['session_id']

        # Close session
        result = await manager.close_session(session_id, 'user1')

        assert result['success'] is True
        assert manager.sessions[session_id].is_active is False

    @pytest.mark.asyncio
    async def test_close_session_non_owner(self, manager):
        """Test that non-owner cannot close session"""
        # Start session
        start_result = await manager.start_session(
            name='Test Session',
            owner_id='user1',
            username='Alice'
        )
        session_id = start_result['session_id']

        # Join as another user
        await manager.join_session(session_id, 'user2', 'Bob')

        # Try to close as non-owner
        result = await manager.close_session(session_id, 'user2')

        assert result['success'] is False
        assert 'owner' in result['error'].lower()


class TestWebSocketEditor:
    """Test WebSocketEditor class"""

    @pytest.mark.asyncio
    async def test_start_server(self, editor):
        """Test starting WebSocket server"""
        result = await editor.start_server()

        assert result['success'] is True
        assert 'host' in result
        assert 'port' in result

    @pytest.mark.asyncio
    async def test_stop_server(self, editor):
        """Test stopping WebSocket server"""
        await editor.start_server()
        result = await editor.stop_server()

        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_connect(self, editor):
        """Test WebSocket connection"""
        result = await editor.connect('user1', None)  # Mock websocket

        assert result['success'] is True
        assert result['user_id'] == 'user1'
        assert result['state'] == ConnectionState.CONNECTED.value

    @pytest.mark.asyncio
    async def test_disconnect(self, editor):
        """Test WebSocket disconnection"""
        await editor.connect('user1', None)
        result = await editor.disconnect('user1')

        assert result['success'] is True
        assert result['state'] == ConnectionState.DISCONNECTED.value

    @pytest.mark.asyncio
    async def test_broadcast_change(self, editor, manager):
        """Test broadcasting changes"""
        # Start session with multiple users
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']
        await manager.join_session(session_id, 'user2', 'Bob')

        # Connect users
        await editor.connect('user1', None)
        await editor.connect('user2', None)

        # Broadcast change
        change = {'type': 'insert', 'content': 'test'}
        result = await editor.broadcast_change(session_id, change, exclude_user='user1')

        assert result['success'] is True
        assert result['broadcast_count'] == 1  # Only user2 receives

    @pytest.mark.asyncio
    async def test_batch_changes(self, editor, manager):
        """Test batching changes"""
        # Start session
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Add changes to batch
        changes = [
            {'type': 'insert', 'content': 'a'},
            {'type': 'insert', 'content': 'b'}
        ]

        result = await editor.batch_changes(session_id, changes)

        assert result['success'] is True
        assert result['pending_count'] == 2

    @pytest.mark.asyncio
    async def test_flush_changes(self, editor, manager):
        """Test flushing batched changes"""
        # Start session
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Add changes
        changes = [{'type': 'insert', 'content': 'test'}]
        await editor.batch_changes(session_id, changes)

        # Flush
        result = await editor.flush_changes(session_id)

        assert result['success'] is True
        assert result['flushed_count'] == 1


class TestConflictResolver:
    """Test ConflictResolver class"""

    @pytest.mark.asyncio
    async def test_last_write_wins(self, resolver):
        """Test last-write-wins strategy"""
        changes = [
            Change(
                change_id='1',
                user_id='user1',
                change_type=ChangeType.INSERT,
                position=0,
                content='Hello',
                timestamp='2024-01-01T10:00:00',
                version=1
            ),
            Change(
                change_id='2',
                user_id='user2',
                change_type=ChangeType.INSERT,
                position=0,
                content='World',
                timestamp='2024-01-01T10:00:01',
                version=1
            )
        ]

        result = await resolver.resolve_conflict(changes, 'last_write_wins')

        assert result['success'] is True
        assert len(result['resolved_changes']) == 1
        assert result['resolved_changes'][0]['content'] == 'World'

    @pytest.mark.asyncio
    async def test_operational_transform(self, resolver):
        """Test operational transformation"""
        changes = [
            Change(
                change_id='1',
                user_id='user1',
                change_type=ChangeType.INSERT,
                position=0,
                content='Hello',
                timestamp='2024-01-01T10:00:00',
                version=1
            ),
            Change(
                change_id='2',
                user_id='user2',
                change_type=ChangeType.INSERT,
                position=0,
                content='World',
                timestamp='2024-01-01T10:00:01',
                version=1
            )
        ]

        result = await resolver.resolve_conflict(changes, 'operational_transform')

        assert result['success'] is True
        assert len(result['resolved_changes']) == 2
        assert result['strategy'] == 'operational_transform'

    @pytest.mark.asyncio
    async def test_crdt_merge(self, resolver):
        """Test CRDT-based merge"""
        changes = [
            Change(
                change_id='1',
                user_id='user1',
                change_type=ChangeType.INSERT,
                position=0,
                content='Hello',
                timestamp='2024-01-01T10:00:00',
                version=1
            ),
            Change(
                change_id='2',
                user_id='user2',
                change_type=ChangeType.INSERT,
                position=5,
                content=' World',
                timestamp='2024-01-01T10:00:01',
                version=1
            )
        ]

        result = await resolver.resolve_conflict(changes, 'crdt')

        assert result['success'] is True
        assert len(result['resolved_changes']) == 2
        assert result['strategy'] == 'crdt'

    @pytest.mark.asyncio
    async def test_unknown_strategy(self, resolver):
        """Test unknown resolution strategy"""
        result = await resolver.resolve_conflict([], 'unknown')

        assert result['success'] is False
        assert 'unknown' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_smart_merge_strategy(self, resolver):
        """Test smart merge strategy"""
        changes = [
            Change(
                change_id='1',
                user_id='user1',
                change_type=ChangeType.INSERT,
                position=0,
                content='Hello',
                timestamp='2024-01-01T10:00:00',
                version=1
            ),
            Change(
                change_id='2',
                user_id='user2',
                change_type=ChangeType.INSERT,
                position=0,
                content='Hi',
                timestamp='2024-01-01T10:00:01',
                version=1
            )
        ]

        result = await resolver.resolve_conflict(changes, 'smart_merge')

        assert result['success'] is True
        assert result['strategy'] == 'smart_merge'
        assert 'conflict_type' in result

    @pytest.mark.asyncio
    async def test_edit_history_recording(self, resolver):
        """Test that resolutions are recorded in history"""
        changes = [
            Change(
                change_id='1',
                user_id='user1',
                change_type=ChangeType.INSERT,
                position=0,
                content='Test',
                timestamp='2024-01-01T10:00:00',
                version=1
            )
        ]

        session_id = 'test_session'
        result = await resolver.resolve_conflict(changes, 'operational_transform', session_id)

        assert result['success'] is True

        # Check history was recorded
        history = await resolver.get_history(session_id)
        assert history['success'] is True
        assert history['count'] == 1
        assert len(history['history']) == 1

    @pytest.mark.asyncio
    async def test_get_history_empty(self, resolver):
        """Test getting history for session with no history"""
        result = await resolver.get_history('nonexistent_session')

        assert result['success'] is True
        assert result['count'] == 0
        assert len(result['history']) == 0

    @pytest.mark.asyncio
    async def test_get_history_with_limit(self, resolver):
        """Test getting history with limit"""
        session_id = 'test_session'

        # Create multiple resolutions
        for i in range(10):
            changes = [
                Change(
                    change_id=f'change_{i}',
                    user_id='user1',
                    change_type=ChangeType.INSERT,
                    position=i,
                    content=f'Test{i}',
                    timestamp=f'2024-01-01T10:00:{i:02d}',
                    version=1
                )
            ]
            await resolver.resolve_conflict(changes, 'operational_transform', session_id)

        # Get limited history
        result = await resolver.get_history(session_id, limit=5)

        assert result['success'] is True
        assert result['count'] == 5
        assert len(result['history']) == 5

    @pytest.mark.asyncio
    async def test_rollback_success(self, resolver):
        """Test successful rollback"""
        session_id = 'test_session'

        # Create some resolutions
        for i in range(5):
            changes = [
                Change(
                    change_id=f'change_{i}',
                    user_id='user1',
                    change_type=ChangeType.INSERT,
                    position=i,
                    content=f'Test{i}',
                    timestamp=f'2024-01-01T10:00:{i:02d}',
                    version=1
                )
            ]
            await resolver.resolve_conflict(changes, 'operational_transform', session_id)

        # Rollback 2 steps
        result = await resolver.rollback(session_id, steps=2)

        assert result['success'] is True
        assert result['steps'] == 2
        assert len(result['rolled_back']) == 2
        assert result['remaining_history'] == 3

    @pytest.mark.asyncio
    async def test_rollback_no_history(self, resolver):
        """Test rollback with no history"""
        result = await resolver.rollback('nonexistent_session', steps=1)

        assert result['success'] is False
        assert 'no history' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_rollback_too_many_steps(self, resolver):
        """Test rollback with too many steps"""
        session_id = 'test_session'

        # Create 2 resolutions
        for i in range(2):
            changes = [
                Change(
                    change_id=f'change_{i}',
                    user_id='user1',
                    change_type=ChangeType.INSERT,
                    position=i,
                    content=f'Test{i}',
                    timestamp=f'2024-01-01T10:00:{i:02d}',
                    version=1
                )
            ]
            await resolver.resolve_conflict(changes, 'operational_transform', session_id)

        # Try to rollback 5 steps
        result = await resolver.rollback(session_id, steps=5)

        assert result['success'] is False
        assert 'cannot rollback' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_conflict_type_detection_no_conflict(self, resolver):
        """Test conflict type detection for no conflict"""
        changes = [
            Change(
                change_id='1',
                user_id='user1',
                change_type=ChangeType.INSERT,
                position=0,
                content='Hello',
                timestamp='2024-01-01T10:00:00',
                version=1
            )
        ]

        conflict_type = resolver._detect_conflict_type(changes)
        assert conflict_type == 'no_conflict'

    @pytest.mark.asyncio
    async def test_conflict_type_detection_same_position(self, resolver):
        """Test conflict type detection for same position"""
        changes = [
            Change(
                change_id='1',
                user_id='user1',
                change_type=ChangeType.INSERT,
                position=0,
                content='Hello',
                timestamp='2024-01-01T10:00:00',
                version=1
            ),
            Change(
                change_id='2',
                user_id='user2',
                change_type=ChangeType.INSERT,
                position=0,
                content='World',
                timestamp='2024-01-01T10:00:01',
                version=1
            )
        ]

        conflict_type = resolver._detect_conflict_type(changes)
        assert conflict_type == 'same_position'

    @pytest.mark.asyncio
    async def test_conflict_type_detection_overlapping(self, resolver):
        """Test conflict type detection for overlapping edits"""
        changes = [
            Change(
                change_id='1',
                user_id='user1',
                change_type=ChangeType.DELETE,
                position=0,
                content='Hello',
                timestamp='2024-01-01T10:00:00',
                version=1
            ),
            Change(
                change_id='2',
                user_id='user2',
                change_type=ChangeType.INSERT,
                position=2,
                content='XX',
                timestamp='2024-01-01T10:00:01',
                version=1
            )
        ]

        conflict_type = resolver._detect_conflict_type(changes)
        assert conflict_type == 'overlapping_edits'

    @pytest.mark.asyncio
    async def test_transform_insert_after_insert(self, resolver):
        """Test transforming insert after another insert"""
        change = Change(
            change_id='2',
            user_id='user2',
            change_type=ChangeType.INSERT,
            position=5,
            content='World',
            timestamp='2024-01-01T10:00:01',
            version=1
        )

        against = Change(
            change_id='1',
            user_id='user1',
            change_type=ChangeType.INSERT,
            position=0,
            content='Hello',
            timestamp='2024-01-01T10:00:00',
            version=1
        )

        transformed = resolver._transform_change(change, against)

        # Position should be adjusted by length of inserted content
        assert transformed.position == 10  # 5 + len('Hello')

    @pytest.mark.asyncio
    async def test_transform_insert_after_delete(self, resolver):
        """Test transforming insert after a delete"""
        change = Change(
            change_id='2',
            user_id='user2',
            change_type=ChangeType.INSERT,
            position=10,
            content='World',
            timestamp='2024-01-01T10:00:01',
            version=1
        )

        against = Change(
            change_id='1',
            user_id='user1',
            change_type=ChangeType.DELETE,
            position=0,
            content='Hello',
            timestamp='2024-01-01T10:00:00',
            version=1
        )

        transformed = resolver._transform_change(change, against)

        # Position should be adjusted backward by deleted content length
        assert transformed.position == 5  # 10 - len('Hello')

    @pytest.mark.asyncio
    async def test_smart_merge_sequential_edits(self, resolver):
        """Test smart merge with sequential edits"""
        changes = [
            Change(
                change_id='1',
                user_id='user1',
                change_type=ChangeType.INSERT,
                position=0,
                content='Hello',
                timestamp='2024-01-01T10:00:00',
                version=1
            ),
            Change(
                change_id='2',
                user_id='user2',
                change_type=ChangeType.INSERT,
                position=10,
                content='World',
                timestamp='2024-01-01T10:00:01',
                version=1
            )
        ]

        result = await resolver.resolve_conflict(changes, 'smart_merge')

        assert result['success'] is True
        assert result['conflict_type'] in ['sequential_edits', 'no_conflict']

    @pytest.mark.asyncio
    async def test_history_size_limit(self, resolver):
        """Test that history is limited to max size"""
        session_id = 'test_session'

        # Create more resolutions than max_history_size
        for i in range(resolver.max_history_size + 100):
            changes = [
                Change(
                    change_id=f'change_{i}',
                    user_id='user1',
                    change_type=ChangeType.INSERT,
                    position=0,
                    content=f'Test{i}',
                    timestamp='2024-01-01T10:00:00',
                    version=1
                )
            ]
            await resolver.resolve_conflict(changes, 'operational_transform', session_id)

        # History should be limited
        assert len(resolver.edit_history[session_id]) == resolver.max_history_size


class TestCollaborativeCodingFeature:
    """Test CollaborativeCodingFeature class"""

    @pytest.mark.asyncio
    async def test_feature_initialization(self, feature_config):
        """Test feature initialization"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        assert feature.name == 'collaborative_coding'
        assert feature.is_initialized is True
        assert feature.manager is not None
        assert feature.editor is not None
        assert feature.resolver is not None

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_start_session(self, feature_config):
        """Test starting a session through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        result = await feature.start('Test Session', 'user1', 'Alice')

        assert result['success'] is True
        assert 'session_id' in result

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_join_session(self, feature_config):
        """Test joining a session through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session
        start_result = await feature.start('Test Session', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Join session
        result = await feature.join(session_id, 'user2', 'Bob')

        assert result['success'] is True
        assert result['session_id'] == session_id

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_leave_session(self, feature_config):
        """Test leaving a session through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start and join
        start_result = await feature.start('Test Session', 'user1', 'Alice')
        session_id = start_result['session_id']
        await feature.join(session_id, 'user2', 'Bob')

        # Leave
        result = await feature.leave(session_id, 'user2')

        assert result['success'] is True

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_sync_change(self, feature_config):
        """Test syncing changes through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session
        start_result = await feature.start('Test Session', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Sync change
        change_data = {
            'change_type': 'insert',
            'position': 0,
            'content': 'Hello'
        }

        result = await feature.sync(session_id, 'user1', change_data)

        assert result['success'] is True
        assert 'change' in result

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_update_cursor(self, feature_config):
        """Test updating cursor through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session
        start_result = await feature.start('Test Session', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Update cursor
        result = await feature.update_cursor(session_id, 'user1', 10)

        assert result['success'] is True

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_get_session_info(self, feature_config):
        """Test getting session info through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session
        start_result = await feature.start('Test Session', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Get info
        result = await feature.get_session_info(session_id)

        assert result['success'] is True
        assert result['session']['session_id'] == session_id

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_list_sessions(self, feature_config):
        """Test listing sessions through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start sessions
        await feature.start('Session 1', 'user1', 'Alice')
        await feature.start('Session 2', 'user2', 'Bob')

        # List
        result = await feature.list_sessions()

        assert result['success'] is True
        assert result['count'] == 2

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_close_session(self, feature_config):
        """Test closing session through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session
        start_result = await feature.start('Test Session', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Close
        result = await feature.close_session(session_id, 'user1')

        assert result['success'] is True

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_resolve_conflicts(self, feature_config):
        """Test resolving conflicts through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session and make changes
        start_result = await feature.start('Test Session', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Add some changes
        await feature.sync(session_id, 'user1', {
            'change_type': 'insert',
            'position': 0,
            'content': 'Hello'
        })

        # Resolve conflicts
        result = await feature.resolve_conflicts(session_id)

        assert result['success'] is True

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_get_edit_history(self, feature_config):
        """Test getting edit history through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session and make changes
        start_result = await feature.start('Test Session', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Add some changes and resolve
        await feature.sync(session_id, 'user1', {
            'change_type': 'insert',
            'position': 0,
            'content': 'Hello'
        })
        await feature.resolve_conflicts(session_id)

        # Get history
        result = await feature.get_edit_history(session_id)

        assert result['success'] is True
        assert 'history' in result
        assert result['count'] >= 0

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_rollback_edits(self, feature_config):
        """Test rolling back edits through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session and make changes
        start_result = await feature.start('Test Session', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Add changes and resolve multiple times
        for i in range(3):
            await feature.sync(session_id, 'user1', {
                'change_type': 'insert',
                'position': i,
                'content': f'Test{i}'
            })
            await feature.resolve_conflicts(session_id)

        # Rollback 1 step
        result = await feature.rollback_edits(session_id, steps=1)

        assert result['success'] is True
        assert result['steps'] == 1

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_api_endpoints(self, feature_config):
        """Test that API endpoints are defined"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        endpoints = feature.get_api_endpoints()

        assert len(endpoints) > 0
        assert any(e['path'] == '/api/collab/start' for e in endpoints)
        assert any(e['path'] == '/api/collab/join' for e in endpoints)
        assert any(e['path'] == '/api/collab/sync' for e in endpoints)
        assert any(e['path'] == '/api/collab/resolve' for e in endpoints)
        assert any(e['path'] == '/api/collab/history/{session_id}' for e in endpoints)
        assert any(e['path'] == '/api/collab/rollback' for e in endpoints)

        await feature.shutdown()


class TestDataClasses:
    """Test data classes"""

    def test_user_to_dict(self):
        """Test User.to_dict()"""
        user = User(
            user_id='user1',
            username='Alice',
            color='#FF6B6B',
            cursor_position=10
        )

        data = user.to_dict()

        assert data['user_id'] == 'user1'
        assert data['username'] == 'Alice'
        assert data['color'] == '#FF6B6B'
        assert data['cursor_position'] == 10

    def test_change_to_dict(self):
        """Test Change.to_dict()"""
        change = Change(
            change_id='change1',
            user_id='user1',
            change_type=ChangeType.INSERT,
            position=0,
            content='Hello',
            timestamp='2024-01-01T10:00:00',
            version=1
        )

        data = change.to_dict()

        assert data['change_id'] == 'change1'
        assert data['user_id'] == 'user1'
        assert data['change_type'] == 'insert'
        assert data['content'] == 'Hello'

    def test_session_to_dict(self):
        """Test Session.to_dict()"""
        user = User(
            user_id='user1',
            username='Alice',
            color='#FF6B6B'
        )

        session = Session(
            session_id='session1',
            name='Test Session',
            owner_id='user1',
            created_at='2024-01-01T10:00:00',
            participants={'user1': user}
        )

        data = session.to_dict()

        assert data['session_id'] == 'session1'
        assert data['name'] == 'Test Session'
        assert data['owner_id'] == 'user1'
        assert data['participant_count'] == 1

    def test_config_from_dict(self):
        """Test CollaborativeConfig.from_dict()"""
        data = {
            'enabled': True,
            'websocket_port': 9000,
            'max_connections': 20
        }

        config = CollaborativeConfig.from_dict(data)

        assert config.enabled is True
        assert config.websocket_port == 9000
        assert config.max_connections == 20


class TestConcurrentEditing:
    """Test concurrent editing scenarios"""

    @pytest.mark.asyncio
    async def test_concurrent_inserts(self, manager):
        """Test concurrent insert operations"""
        # Start session
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']
        await manager.join_session(session_id, 'user2', 'Bob')

        # Both users insert at same position
        await manager.sync_change(session_id, 'user1', {
            'change_type': 'insert',
            'position': 0,
            'content': 'Hello'
        })

        await manager.sync_change(session_id, 'user2', {
            'change_type': 'insert',
            'position': 5,
            'content': ' World'
        })

        session = manager.sessions[session_id]
        assert 'Hello' in session.document_content
        assert 'World' in session.document_content

    @pytest.mark.asyncio
    async def test_concurrent_delete_and_insert(self, manager):
        """Test concurrent delete and insert"""
        # Start session with content
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']
        await manager.join_session(session_id, 'user2', 'Bob')

        session = manager.sessions[session_id]
        session.document_content = 'Hello World'

        # User1 deletes, User2 inserts
        await manager.sync_change(session_id, 'user1', {
            'change_type': 'delete',
            'position': 6,
            'content': 'World'
        })

        await manager.sync_change(session_id, 'user2', {
            'change_type': 'insert',
            'position': 11,
            'content': '!'
        })

        # Document should reflect both changes
        assert session.document_version == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestPresenceManager:
    """Test PresenceManager class"""

    @pytest.fixture
    def presence_manager(self):
        """Create presence manager"""
        from xencode.features.collaborative_coding import PresenceManager
        return PresenceManager(idle_timeout_seconds=60)

    @pytest.fixture
    def test_user(self):
        """Create test user"""
        return User(
            user_id='user1',
            username='Alice',
            color='#FF6B6B'
        )

    @pytest.mark.asyncio
    async def test_update_presence_online(self, presence_manager, test_user):
        """Test updating presence to online"""
        result = await presence_manager.update_presence(test_user, 'online')

        assert result['success'] is True
        assert result['user']['status'] == 'online'
        assert result['user']['idle_since'] is None

    @pytest.mark.asyncio
    async def test_update_presence_idle(self, presence_manager, test_user):
        """Test updating presence to idle"""
        result = await presence_manager.update_presence(test_user, 'idle')

        assert result['success'] is True
        assert result['user']['status'] == 'idle'
        assert result['user']['idle_since'] is not None

    @pytest.mark.asyncio
    async def test_update_presence_invalid_status(self, presence_manager, test_user):
        """Test updating presence with invalid status"""
        result = await presence_manager.update_presence(test_user, 'invalid')

        assert result['success'] is False
        assert 'invalid' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_update_typing_status(self, presence_manager, test_user):
        """Test updating typing status"""
        result = await presence_manager.update_typing_status(test_user, True)

        assert result['success'] is True
        assert result['user']['is_typing'] is True

    @pytest.mark.asyncio
    async def test_typing_resets_idle(self, presence_manager, test_user):
        """Test that typing resets idle status"""
        # Set user to idle
        await presence_manager.update_presence(test_user, 'idle')
        assert test_user.status == 'idle'

        # Start typing
        result = await presence_manager.update_typing_status(test_user, True)

        assert result['success'] is True
        assert result['user']['status'] == 'online'
        assert result['user']['idle_since'] is None

    @pytest.mark.asyncio
    async def test_update_file_location(self, presence_manager, test_user):
        """Test updating file location"""
        result = await presence_manager.update_file_location(
            test_user,
            '/path/to/file.py',
            'line 42'
        )

        assert result['success'] is True
        assert result['user']['current_file'] == '/path/to/file.py'
        assert result['user']['current_location'] == 'line 42'

    @pytest.mark.asyncio
    async def test_file_change_resets_idle(self, presence_manager, test_user):
        """Test that file change resets idle status"""
        # Set user to idle
        await presence_manager.update_presence(test_user, 'idle')
        assert test_user.status == 'idle'

        # Change file
        result = await presence_manager.update_file_location(test_user, '/new/file.py')

        assert result['success'] is True
        assert result['user']['status'] == 'online'

    @pytest.mark.asyncio
    async def test_check_idle_status_not_idle(self, presence_manager, test_user):
        """Test checking idle status when user is active"""
        result = await presence_manager.check_idle_status(test_user)

        assert result['is_idle'] is False
        assert result['seconds_since_activity'] < 60

    @pytest.mark.asyncio
    async def test_check_idle_status_offline(self, presence_manager, test_user):
        """Test checking idle status for offline user"""
        test_user.status = 'offline'
        result = await presence_manager.check_idle_status(test_user)

        assert result['is_idle'] is False
        assert 'offline' in result['reason'].lower()

    @pytest.mark.asyncio
    async def test_record_activity(self, presence_manager, test_user):
        """Test recording user activity"""
        result = await presence_manager.record_activity(
            test_user,
            'editing',
            {'file': 'test.py', 'line': 10}
        )

        assert result['success'] is True
        assert test_user.user_id in presence_manager.activity_log

    @pytest.mark.asyncio
    async def test_get_activity_log(self, presence_manager, test_user):
        """Test getting activity log"""
        # Record some activities
        await presence_manager.record_activity(test_user, 'typing', {})
        await presence_manager.record_activity(test_user, 'editing', {})

        result = await presence_manager.get_activity_log(test_user.user_id)

        assert result['success'] is True
        assert result['count'] == 2
        assert len(result['activities']) == 2

    @pytest.mark.asyncio
    async def test_get_activity_log_empty(self, presence_manager):
        """Test getting activity log for user with no activities"""
        result = await presence_manager.get_activity_log('nonexistent')

        assert result['success'] is True
        assert result['count'] == 0
        assert len(result['activities']) == 0

    @pytest.mark.asyncio
    async def test_get_all_presence(self, presence_manager):
        """Test getting presence for all users"""
        users = {
            'user1': User('user1', 'Alice', '#FF6B6B'),
            'user2': User('user2', 'Bob', '#4ECDC4')
        }

        result = await presence_manager.get_all_presence(users)

        assert result['success'] is True
        assert result['user_count'] == 2
        assert 'user1' in result['presence']
        assert 'user2' in result['presence']

    @pytest.mark.asyncio
    async def test_activity_log_limit(self, presence_manager, test_user):
        """Test that activity log is limited to 100 entries"""
        # Record 150 activities
        for i in range(150):
            await presence_manager.record_activity(test_user, f'activity_{i}', {})

        # Should only keep last 100
        assert len(presence_manager.activity_log[test_user.user_id]) == 100


class TestCollaborationManagerPresence:
    """Test presence features in CollaborationManager"""

    @pytest.mark.asyncio
    async def test_update_user_presence(self, manager):
        """Test updating user presence through manager"""
        # Start session
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Update presence
        result = await manager.update_user_presence(session_id, 'user1', 'away')

        assert result['success'] is True
        assert result['user']['status'] == 'away'

    @pytest.mark.asyncio
    async def test_update_typing_status(self, manager):
        """Test updating typing status through manager"""
        # Start session
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Update typing
        result = await manager.update_typing_status(session_id, 'user1', True)

        assert result['success'] is True
        assert result['user']['is_typing'] is True
        assert 'broadcast_to' in result

    @pytest.mark.asyncio
    async def test_update_file_location(self, manager):
        """Test updating file location through manager"""
        # Start session
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Update file location
        result = await manager.update_file_location(
            session_id, 'user1', '/test/file.py', 'line 10'
        )

        assert result['success'] is True
        assert result['user']['current_file'] == '/test/file.py'
        assert result['user']['current_location'] == 'line 10'

    @pytest.mark.asyncio
    async def test_get_session_presence(self, manager):
        """Test getting session presence"""
        # Start session with multiple users
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']
        await manager.join_session(session_id, 'user2', 'Bob')

        # Get presence
        result = await manager.get_session_presence(session_id)

        assert result['success'] is True
        assert result['user_count'] == 2
        assert 'user1' in result['presence']
        assert 'user2' in result['presence']

    @pytest.mark.asyncio
    async def test_check_idle_users(self, manager):
        """Test checking idle users"""
        # Start session
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Check idle users
        result = await manager.check_idle_users(session_id)

        assert result['success'] is True
        assert 'idle_users' in result
        assert 'idle_count' in result


class TestWebSocketEditorPresence:
    """Test presence broadcasting in WebSocketEditor"""

    @pytest.mark.asyncio
    async def test_broadcast_presence(self, editor, manager):
        """Test broadcasting presence updates"""
        # Start session with multiple users
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']
        await manager.join_session(session_id, 'user2', 'Bob')

        # Connect users
        await editor.connect('user1', None)
        await editor.connect('user2', None)

        # Broadcast presence
        user_data = {'user_id': 'user1', 'status': 'away'}
        result = await editor.broadcast_presence(session_id, user_data, exclude_user='user1')

        assert result['success'] is True
        assert result['broadcast_count'] == 1

    @pytest.mark.asyncio
    async def test_broadcast_typing(self, editor, manager):
        """Test broadcasting typing status"""
        # Start session with multiple users
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']
        await manager.join_session(session_id, 'user2', 'Bob')

        # Connect users
        await editor.connect('user1', None)
        await editor.connect('user2', None)

        # Broadcast typing
        user_data = {'user_id': 'user1', 'is_typing': True}
        result = await editor.broadcast_typing(session_id, user_data, exclude_user='user1')

        assert result['success'] is True
        assert result['broadcast_count'] == 1

    @pytest.mark.asyncio
    async def test_broadcast_file_location(self, editor, manager):
        """Test broadcasting file location"""
        # Start session with multiple users
        start_result = await manager.start_session('Test', 'user1', 'Alice')
        session_id = start_result['session_id']
        await manager.join_session(session_id, 'user2', 'Bob')

        # Connect users
        await editor.connect('user1', None)
        await editor.connect('user2', None)

        # Broadcast file location
        user_data = {'user_id': 'user1', 'current_file': '/test.py'}
        result = await editor.broadcast_file_location(session_id, user_data, exclude_user='user1')

        assert result['success'] is True
        assert result['broadcast_count'] == 1


class TestCollaborativeCodingFeaturePresence:
    """Test presence features in CollaborativeCodingFeature"""

    @pytest.mark.asyncio
    async def test_update_presence(self, feature_config):
        """Test updating presence through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session
        start_result = await feature.start('Test', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Update presence
        result = await feature.update_presence(session_id, 'user1', 'away')

        assert result['success'] is True
        assert result['user']['status'] == 'away'

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_update_typing(self, feature_config):
        """Test updating typing through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session
        start_result = await feature.start('Test', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Update typing
        result = await feature.update_typing(session_id, 'user1', True)

        assert result['success'] is True
        assert result['user']['is_typing'] is True

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_update_file_location(self, feature_config):
        """Test updating file location through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session
        start_result = await feature.start('Test', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Update file location
        result = await feature.update_file_location(session_id, 'user1', '/test.py', 'line 10')

        assert result['success'] is True
        assert result['user']['current_file'] == '/test.py'

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_get_presence(self, feature_config):
        """Test getting presence through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session
        start_result = await feature.start('Test', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Get presence
        result = await feature.get_presence(session_id)

        assert result['success'] is True
        assert result['user_count'] == 1

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_check_idle_users(self, feature_config):
        """Test checking idle users through feature"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        # Start session
        start_result = await feature.start('Test', 'user1', 'Alice')
        session_id = start_result['session_id']

        # Check idle users
        result = await feature.check_idle_users(session_id)

        assert result['success'] is True
        assert 'idle_users' in result

        await feature.shutdown()

    @pytest.mark.asyncio
    async def test_presence_api_endpoints(self, feature_config):
        """Test that presence API endpoints are defined"""
        feature = CollaborativeCodingFeature(feature_config)
        await feature.initialize()

        endpoints = feature.get_api_endpoints()

        # Check presence endpoints exist
        assert any(e['path'] == '/api/collab/presence' for e in endpoints)
        assert any(e['path'] == '/api/collab/typing' for e in endpoints)
        assert any(e['path'] == '/api/collab/file-location' for e in endpoints)
        assert any(e['path'] == '/api/collab/presence/{session_id}' for e in endpoints)
        assert any(e['path'] == '/api/collab/idle/{session_id}' for e in endpoints)

        await feature.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
