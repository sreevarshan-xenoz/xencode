"""
Collaborative Coding Feature Extensions

Additional features for collaborative coding:
- Access Control (room-based permissions)
- Code Review in sessions
- Voice/Video Chat integration (optional)
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


class AccessControl:
    """Manages room-based access control for collaboration sessions"""

    def __init__(self):
        """Initialize access control manager"""
        self.room_permissions: Dict[str, Dict[str, Set[str]]] = {}  # session_id -> {role -> set(user_ids)}
        self.user_roles: Dict[str, Dict[str, str]] = {}  # session_id -> {user_id -> role}
        self.room_passwords: Dict[str, Optional[str]] = {}  # session_id -> password
        self.banned_users: Dict[str, Set[str]] = {}  # session_id -> set(user_ids)

        # Default permissions for each role
        self.default_permissions = {
            'owner': {'read', 'write', 'invite', 'kick', 'manage', 'review'},
            'admin': {'read', 'write', 'invite', 'kick', 'review'},
            'editor': {'read', 'write', 'review'},
            'viewer': {'read'},
            'reviewer': {'read', 'review'}
        }

    async def create_room_access(self, session_id: str, owner_id: str,
                                 password: Optional[str] = None) -> Dict[str, Any]:
        """
        Create access control for a new room

        Args:
            session_id: Session ID
            owner_id: Owner user ID
            password: Optional room password

        Returns:
            Dict with success status
        """
        self.room_permissions[session_id] = {
            'owner': {owner_id},
            'admin': set(),
            'editor': set(),
            'viewer': set(),
            'reviewer': set()
        }

        self.user_roles[session_id] = {owner_id: 'owner'}
        self.room_passwords[session_id] = password
        self.banned_users[session_id] = set()

        return {
            'success': True,
            'session_id': session_id,
            'owner_id': owner_id,
            'password_protected': password is not None
        }

    async def check_access(self, session_id: str, user_id: str,
                          password: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if user has access to a room

        Args:
            session_id: Session ID
            user_id: User ID
            password: Optional password for password-protected rooms

        Returns:
            Dict with access status and reason
        """
        if session_id not in self.room_permissions:
            return {
                'success': False,
                'has_access': False,
                'reason': 'Room not found'
            }

        # Check if user is banned
        if user_id in self.banned_users.get(session_id, set()):
            return {
                'success': True,
                'has_access': False,
                'reason': 'User is banned from this room'
            }

        # Check password if room is password-protected
        room_password = self.room_passwords.get(session_id)
        if room_password and password != room_password:
            return {
                'success': True,
                'has_access': False,
                'reason': 'Invalid password'
            }

        return {
            'success': True,
            'has_access': True,
            'reason': 'Access granted'
        }

    async def grant_role(self, session_id: str, user_id: str, role: str,
                        granted_by: str) -> Dict[str, Any]:
        """
        Grant a role to a user

        Args:
            session_id: Session ID
            user_id: User ID to grant role to
            role: Role to grant (owner, admin, editor, viewer, reviewer)
            granted_by: User ID granting the role

        Returns:
            Dict with success status
        """
        if session_id not in self.room_permissions:
            return {
                'success': False,
                'error': 'Room not found'
            }

        # Check if granter has permission to grant roles
        granter_role = self.user_roles.get(session_id, {}).get(granted_by)
        if granter_role not in ['owner', 'admin']:
            return {
                'success': False,
                'error': 'Insufficient permissions to grant roles'
            }

        # Validate role
        if role not in self.default_permissions:
            return {
                'success': False,
                'error': f'Invalid role: {role}'
            }

        # Cannot grant owner role
        if role == 'owner':
            return {
                'success': False,
                'error': 'Cannot grant owner role'
            }

        # Remove user from previous role
        if user_id in self.user_roles.get(session_id, {}):
            old_role = self.user_roles[session_id][user_id]
            if old_role in self.room_permissions[session_id]:
                self.room_permissions[session_id][old_role].discard(user_id)

        # Add user to new role
        self.room_permissions[session_id][role].add(user_id)
        if session_id not in self.user_roles:
            self.user_roles[session_id] = {}
        self.user_roles[session_id][user_id] = role

        return {
            'success': True,
            'user_id': user_id,
            'role': role,
            'permissions': list(self.default_permissions[role])
        }

    async def check_permission(self, session_id: str, user_id: str,
                              permission: str) -> Dict[str, Any]:
        """
        Check if user has a specific permission

        Args:
            session_id: Session ID
            user_id: User ID
            permission: Permission to check (read, write, invite, kick, manage, review)

        Returns:
            Dict with permission status
        """
        if session_id not in self.user_roles:
            return {
                'success': True,
                'has_permission': False,
                'reason': 'Room not found'
            }

        user_role = self.user_roles.get(session_id, {}).get(user_id)
        if not user_role:
            return {
                'success': True,
                'has_permission': False,
                'reason': 'User not in room'
            }

        role_permissions = self.default_permissions.get(user_role, set())
        has_permission = permission in role_permissions

        return {
            'success': True,
            'has_permission': has_permission,
            'user_role': user_role,
            'permission': permission
        }

    async def ban_user(self, session_id: str, user_id: str, banned_by: str) -> Dict[str, Any]:
        """
        Ban a user from a room

        Args:
            session_id: Session ID
            user_id: User ID to ban
            banned_by: User ID performing the ban

        Returns:
            Dict with success status
        """
        # Check if banner has permission
        perm_check = await self.check_permission(session_id, banned_by, 'kick')
        if not perm_check['has_permission']:
            return {
                'success': False,
                'error': 'Insufficient permissions to ban users'
            }

        # Cannot ban owner
        if self.user_roles.get(session_id, {}).get(user_id) == 'owner':
            return {
                'success': False,
                'error': 'Cannot ban room owner'
            }

        if session_id not in self.banned_users:
            self.banned_users[session_id] = set()

        self.banned_users[session_id].add(user_id)

        # Remove from room permissions
        if session_id in self.user_roles and user_id in self.user_roles[session_id]:
            role = self.user_roles[session_id][user_id]
            if role in self.room_permissions[session_id]:
                self.room_permissions[session_id][role].discard(user_id)
            del self.user_roles[session_id][user_id]

        return {
            'success': True,
            'user_id': user_id,
            'banned': True
        }

    async def unban_user(self, session_id: str, user_id: str, unbanned_by: str) -> Dict[str, Any]:
        """
        Unban a user from a room

        Args:
            session_id: Session ID
            user_id: User ID to unban
            unbanned_by: User ID performing the unban

        Returns:
            Dict with success status
        """
        # Check if unbanner has permission
        perm_check = await self.check_permission(session_id, unbanned_by, 'kick')
        if not perm_check['has_permission']:
            return {
                'success': False,
                'error': 'Insufficient permissions to unban users'
            }

        if session_id in self.banned_users:
            self.banned_users[session_id].discard(user_id)

        return {
            'success': True,
            'user_id': user_id,
            'unbanned': True
        }

    async def get_room_users(self, session_id: str) -> Dict[str, Any]:
        """
        Get all users in a room with their roles

        Args:
            session_id: Session ID

        Returns:
            Dict with user list and roles
        """
        if session_id not in self.user_roles:
            return {
                'success': False,
                'error': 'Room not found'
            }

        users = []
        for user_id, role in self.user_roles[session_id].items():
            users.append({
                'user_id': user_id,
                'role': role,
                'permissions': list(self.default_permissions[role])
            })

        return {
            'success': True,
            'users': users,
            'count': len(users)
        }


class CodeReviewManager:
    """Manages code review in collaborative sessions"""

    def __init__(self):
        """Initialize code review manager"""
        self.reviews: Dict[str, List[Dict[str, Any]]] = {}  # session_id -> list of reviews
        self.comments: Dict[str, List[Dict[str, Any]]] = {}  # review_id -> list of comments
        self.review_status: Dict[str, str] = {}  # review_id -> status

    async def start_review(self, session_id: str, reviewer_id: str,
                          file_path: str, content: str) -> Dict[str, Any]:
        """
        Start a code review in a session

        Args:
            session_id: Session ID
            reviewer_id: User ID of reviewer
            file_path: Path to file being reviewed
            content: File content to review

        Returns:
            Dict with review information
        """
        review_id = str(uuid.uuid4())

        review = {
            'review_id': review_id,
            'session_id': session_id,
            'reviewer_id': reviewer_id,
            'file_path': file_path,
            'content': content,
            'created_at': datetime.now().isoformat(),
            'status': 'in_progress',
            'comments_count': 0
        }

        if session_id not in self.reviews:
            self.reviews[session_id] = []

        self.reviews[session_id].append(review)
        self.comments[review_id] = []
        self.review_status[review_id] = 'in_progress'

        return {
            'success': True,
            'review': review
        }

    async def add_comment(self, review_id: str, user_id: str,
                         line_number: int, comment_text: str,
                         severity: str = 'info') -> Dict[str, Any]:
        """
        Add a comment to a review

        Args:
            review_id: Review ID
            user_id: User ID adding comment
            line_number: Line number in file
            comment_text: Comment text
            severity: Comment severity (info, warning, error, suggestion)

        Returns:
            Dict with comment information
        """
        if review_id not in self.comments:
            return {
                'success': False,
                'error': 'Review not found'
            }

        comment_id = str(uuid.uuid4())

        comment = {
            'comment_id': comment_id,
            'review_id': review_id,
            'user_id': user_id,
            'line_number': line_number,
            'text': comment_text,
            'severity': severity,
            'created_at': datetime.now().isoformat(),
            'resolved': False
        }

        self.comments[review_id].append(comment)

        return {
            'success': True,
            'comment': comment
        }

    async def resolve_comment(self, review_id: str, comment_id: str,
                             resolved_by: str) -> Dict[str, Any]:
        """
        Mark a comment as resolved

        Args:
            review_id: Review ID
            comment_id: Comment ID
            resolved_by: User ID resolving the comment

        Returns:
            Dict with success status
        """
        if review_id not in self.comments:
            return {
                'success': False,
                'error': 'Review not found'
            }

        for comment in self.comments[review_id]:
            if comment['comment_id'] == comment_id:
                comment['resolved'] = True
                comment['resolved_by'] = resolved_by
                comment['resolved_at'] = datetime.now().isoformat()

                return {
                    'success': True,
                    'comment': comment
                }

        return {
            'success': False,
            'error': 'Comment not found'
        }

    async def complete_review(self, review_id: str, status: str = 'approved') -> Dict[str, Any]:
        """
        Complete a review

        Args:
            review_id: Review ID
            status: Final status (approved, rejected, needs_changes)

        Returns:
            Dict with success status
        """
        if review_id not in self.review_status:
            return {
                'success': False,
                'error': 'Review not found'
            }

        if status not in ['approved', 'rejected', 'needs_changes']:
            return {
                'success': False,
                'error': f'Invalid status: {status}'
            }

        self.review_status[review_id] = status

        # Update review in session
        for session_reviews in self.reviews.values():
            for review in session_reviews:
                if review['review_id'] == review_id:
                    review['status'] = status
                    review['completed_at'] = datetime.now().isoformat()
                    break

        return {
            'success': True,
            'review_id': review_id,
            'status': status
        }

    async def get_review(self, review_id: str) -> Dict[str, Any]:
        """
        Get review details

        Args:
            review_id: Review ID

        Returns:
            Dict with review details
        """
        # Find review
        for session_reviews in self.reviews.values():
            for review in session_reviews:
                if review['review_id'] == review_id:
                    comments = self.comments.get(review_id, [])

                    return {
                        'success': True,
                        'review': review,
                        'comments': comments,
                        'comments_count': len(comments),
                        'unresolved_count': sum(1 for c in comments if not c['resolved'])
                    }

        return {
            'success': False,
            'error': 'Review not found'
        }

    async def list_session_reviews(self, session_id: str) -> Dict[str, Any]:
        """
        List all reviews in a session

        Args:
            session_id: Session ID

        Returns:
            Dict with review list
        """
        reviews = self.reviews.get(session_id, [])

        return {
            'success': True,
            'reviews': reviews,
            'count': len(reviews)
        }


class VoiceVideoChatManager:
    """Manages voice and video chat integration (optional feature)"""

    def __init__(self, enabled: bool = False):
        """
        Initialize voice/video chat manager

        Args:
            enabled: Whether voice/video chat is enabled
        """
        self.enabled = enabled
        self.active_calls: Dict[str, Dict[str, Any]] = {}  # session_id -> call info
        self.participants: Dict[str, Set[str]] = {}  # session_id -> set(user_ids)

    async def start_call(self, session_id: str, initiator_id: str,
                        call_type: str = 'voice') -> Dict[str, Any]:
        """
        Start a voice or video call

        Args:
            session_id: Session ID
            initiator_id: User ID starting the call
            call_type: Type of call (voice, video)

        Returns:
            Dict with call information
        """
        if not self.enabled:
            return {
                'success': False,
                'error': 'Voice/video chat is not enabled'
            }

        if call_type not in ['voice', 'video']:
            return {
                'success': False,
                'error': f'Invalid call type: {call_type}'
            }

        call_id = str(uuid.uuid4())

        call_info = {
            'call_id': call_id,
            'session_id': session_id,
            'initiator_id': initiator_id,
            'call_type': call_type,
            'started_at': datetime.now().isoformat(),
            'status': 'active'
        }

        self.active_calls[session_id] = call_info
        self.participants[session_id] = {initiator_id}

        return {
            'success': True,
            'call': call_info,
            'message': f'{call_type.capitalize()} call started'
        }

    async def join_call(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """
        Join an active call

        Args:
            session_id: Session ID
            user_id: User ID joining the call

        Returns:
            Dict with success status
        """
        if not self.enabled:
            return {
                'success': False,
                'error': 'Voice/video chat is not enabled'
            }

        if session_id not in self.active_calls:
            return {
                'success': False,
                'error': 'No active call in this session'
            }

        if session_id not in self.participants:
            self.participants[session_id] = set()

        self.participants[session_id].add(user_id)

        return {
            'success': True,
            'call': self.active_calls[session_id],
            'participants': list(self.participants[session_id]),
            'participant_count': len(self.participants[session_id])
        }

    async def leave_call(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """
        Leave an active call

        Args:
            session_id: Session ID
            user_id: User ID leaving the call

        Returns:
            Dict with success status
        """
        if session_id in self.participants:
            self.participants[session_id].discard(user_id)

            # End call if no participants left
            if len(self.participants[session_id]) == 0:
                return await self.end_call(session_id)

        return {
            'success': True,
            'participants': list(self.participants.get(session_id, [])),
            'participant_count': len(self.participants.get(session_id, []))
        }

    async def end_call(self, session_id: str) -> Dict[str, Any]:
        """
        End an active call

        Args:
            session_id: Session ID

        Returns:
            Dict with success status
        """
        if session_id in self.active_calls:
            call_info = self.active_calls[session_id]
            call_info['status'] = 'ended'
            call_info['ended_at'] = datetime.now().isoformat()

            del self.active_calls[session_id]

            if session_id in self.participants:
                del self.participants[session_id]

            return {
                'success': True,
                'call': call_info,
                'message': 'Call ended'
            }

        return {
            'success': False,
            'error': 'No active call in this session'
        }

    async def get_call_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get status of call in a session

        Args:
            session_id: Session ID

        Returns:
            Dict with call status
        """
        if session_id in self.active_calls:
            return {
                'success': True,
                'active': True,
                'call': self.active_calls[session_id],
                'participants': list(self.participants.get(session_id, [])),
                'participant_count': len(self.participants.get(session_id, []))
            }

        return {
            'success': True,
            'active': False,
            'message': 'No active call'
        }

    async def toggle_mute(self, session_id: str, user_id: str, muted: bool) -> Dict[str, Any]:
        """
        Toggle mute status for a user

        Args:
            session_id: Session ID
            user_id: User ID
            muted: Whether user is muted

        Returns:
            Dict with success status
        """
        if not self.enabled:
            return {
                'success': False,
                'error': 'Voice/video chat is not enabled'
            }

        if session_id not in self.active_calls:
            return {
                'success': False,
                'error': 'No active call in this session'
            }

        return {
            'success': True,
            'user_id': user_id,
            'muted': muted
        }

    async def toggle_video(self, session_id: str, user_id: str, video_enabled: bool) -> Dict[str, Any]:
        """
        Toggle video for a user

        Args:
            session_id: Session ID
            user_id: User ID
            video_enabled: Whether video is enabled

        Returns:
            Dict with success status
        """
        if not self.enabled:
            return {
                'success': False,
                'error': 'Voice/video chat is not enabled'
            }

        if session_id not in self.active_calls:
            return {
                'success': False,
                'error': 'No active call in this session'
            }

        call = self.active_calls[session_id]
        if call['call_type'] != 'video':
            return {
                'success': False,
                'error': 'This is not a video call'
            }

        return {
            'success': True,
            'user_id': user_id,
            'video_enabled': video_enabled
        }
