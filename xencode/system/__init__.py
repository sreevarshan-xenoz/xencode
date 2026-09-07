"""
System package for Xencode.

Provides system-level utilities including debugging, session management, and robustness.
"""

try:
    from .debugging_manager import DebuggingManager
except ImportError:
    DebuggingManager = None

try:
    from .session_manager import SessionManager, SessionInfo, get_session_manager
except ImportError:
    SessionManager = SessionInfo = get_session_manager = None

try:
    from .robustness_manager import RecoveryManager, RobustWarpTerminal
except ImportError:
    RecoveryManager = RobustWarpTerminal = None

__all__ = [
    "DebuggingManager",
    "SessionManager",
    "SessionInfo",
    "get_session_manager",
    "RecoveryManager",
    "RobustWarpTerminal",
]
