use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    Admin,
    Editor,
    Viewer,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::Admin => write!(f, "admin"),
            Role::Editor => write!(f, "editor"),
            Role::Viewer => write!(f, "viewer"),
        }
    }
}

impl Role {
    /// Hierarchy rank: Viewer < Editor < Admin.
    fn rank(&self) -> u8 {
        match self {
            Role::Viewer => 0,
            Role::Editor => 1,
            Role::Admin => 2,
        }
    }

    /// True when this role meets the `required` bar.
    pub fn can(&self, required: &Role) -> bool {
        self.rank() >= required.rank()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workspace {
    pub id: String,
    pub name: String,
    pub owner: String,
    pub created_at: String,
    pub members: HashMap<String, Role>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Member {
    pub user_id: String,
    pub role: Role,
    pub joined_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum WorkspaceError {
    #[error("workspace not found: {0}")]
    NotFound(String),
    #[error("{actor} may not {action}: requires {requires} (has {has})")]
    Forbidden {
        actor: String,
        action: String,
        requires: String,
        has: String,
    },
    #[error("cannot remove the last admin of workspace {0}")]
    LastAdmin(String),
}

/// Auditable workspace actions. Denied attempts are logged too — a team
/// audit trail that omits failures is a liability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditAction {
    WorkspaceCreated,
    MemberAdded,
    MemberRemoved,
    RoleChanged,
    Denied,
}

impl std::fmt::Display for AuditAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditAction::WorkspaceCreated => write!(f, "workspace_created"),
            AuditAction::MemberAdded => write!(f, "member_added"),
            AuditAction::MemberRemoved => write!(f, "member_removed"),
            AuditAction::RoleChanged => write!(f, "role_changed"),
            AuditAction::Denied => write!(f, "denied"),
        }
    }
}

/// One audit entry. `seq` is a per-manager monotonic counter so ordering
/// never depends on clock resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEvent {
    pub seq: u64,
    pub at: String,
    pub actor: String,
    pub action: AuditAction,
    pub target: String,
    pub detail: String,
}

/// In-memory workspace manager for collaboration sessions.
///
/// Every mutation is authorized (Admin-gated membership changes) and
/// appended to the audit log — including denied attempts.
#[derive(Debug, Default)]
pub struct WorkspaceManager {
    workspaces: HashMap<String, Workspace>,
    audit: Vec<AuditEvent>,
    next_seq: u64,
}

impl WorkspaceManager {
    pub fn new() -> Self {
        Self {
            workspaces: HashMap::new(),
            audit: Vec::new(),
            next_seq: 1,
        }
    }

    fn log(&mut self, actor: &str, action: AuditAction, target: &str, detail: String) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.audit.push(AuditEvent {
            seq,
            at: chrono::Utc::now().to_rfc3339(),
            actor: actor.to_string(),
            action,
            target: target.to_string(),
            detail,
        });
    }

    /// Full audit trail in append order.
    pub fn audit_log(&self) -> &[AuditEvent] {
        &self.audit
    }

    /// Audit entries touching one workspace, in append order.
    pub fn events_for(&self, workspace_id: &str) -> Vec<&AuditEvent> {
        self.audit
            .iter()
            .filter(|e| e.target == workspace_id)
            .collect()
    }

    /// Create a new workspace. The owner becomes its first admin.
    pub fn create_workspace(&mut self, name: &str, owner: &str) -> Workspace {
        let id = format!("ws-{}", &uuid::Uuid::new_v4().to_string()[..8]);
        self.create_workspace_with_id(&id, name, owner)
    }

    /// Like [`create_workspace`](Self::create_workspace) with a caller-chosen
    /// id — the collaboration server generates session ids and must keep
    /// them. Re-using an existing id overwrites that workspace (deliberate:
    /// the server picks collision-free ids itself; "fixing" this silently
    /// would break that flow).
    pub fn create_workspace_with_id(&mut self, id: &str, name: &str, owner: &str) -> Workspace {
        let mut members = HashMap::new();
        members.insert(owner.to_string(), Role::Admin);

        let ws = Workspace {
            id: id.to_string(),
            name: name.to_string(),
            owner: owner.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            members,
        };
        self.workspaces.insert(id.to_string(), ws.clone());
        self.log(
            owner,
            AuditAction::WorkspaceCreated,
            id,
            format!("workspace '{name}' created"),
        );
        ws
    }

    fn admin_count(ws: &Workspace) -> usize {
        ws.members.values().filter(|r| **r == Role::Admin).count()
    }

    /// Get workspace by ID.
    pub fn get_workspace(&self, id: &str) -> Option<&Workspace> {
        self.workspaces.get(id)
    }

    /// Actor's role in the workspace, if they belong to it.
    pub fn role_of(&self, workspace_id: &str, user_id: &str) -> Option<Role> {
        self.workspaces
            .get(workspace_id)?
            .members
            .get(user_id)
            .cloned()
    }

    fn deny(
        &mut self,
        actor: &str,
        workspace_id: &str,
        action: &str,
        requires: &Role,
        has: Option<&Role>,
    ) -> WorkspaceError {
        let has = has
            .map(|r| r.to_string())
            .unwrap_or_else(|| "non-member".to_string());
        self.log(
            actor,
            AuditAction::Denied,
            workspace_id,
            format!("{actor} attempted {action} (requires {requires}, has {has})"),
        );
        WorkspaceError::Forbidden {
            actor: actor.to_string(),
            action: action.to_string(),
            requires: requires.to_string(),
            has,
        }
    }

    /// Add a member (or change their role) — Admin only. Re-adding with the
    /// same role is an idempotent no-op and logs nothing. Demoting the sole
    /// admin is refused: an orphaned workspace has no one left to govern it.
    pub fn add_member(
        &mut self,
        workspace_id: &str,
        actor: &str,
        user_id: &str,
        role: Role,
    ) -> Result<(), WorkspaceError> {
        let actor_role = self.role_of(workspace_id, actor);
        if actor_role.as_ref().is_none_or(|r| !r.can(&Role::Admin)) {
            if self.workspaces.contains_key(workspace_id) {
                return Err(self.deny(
                    actor,
                    workspace_id,
                    "add_member",
                    &Role::Admin,
                    actor_role.as_ref(),
                ));
            }
            return Err(WorkspaceError::NotFound(workspace_id.to_string()));
        }
        let ws = self.workspaces.get_mut(workspace_id).unwrap();
        let existing = ws.members.get(user_id).cloned();
        match existing {
            Some(ref r) if *r == role => Ok(()),
            Some(old) => {
                if old == Role::Admin && role != Role::Admin && Self::admin_count(ws) == 1 {
                    let id = ws.id.clone();
                    self.log(
                        actor,
                        AuditAction::Denied,
                        &id,
                        format!("blocked demoting last admin {user_id}"),
                    );
                    return Err(WorkspaceError::LastAdmin(id));
                }
                ws.members.insert(user_id.to_string(), role.clone());
                let id = ws.id.clone();
                self.log(
                    actor,
                    AuditAction::RoleChanged,
                    &id,
                    format!("{actor} set {user_id} to {role}"),
                );
                Ok(())
            }
            None => {
                ws.members.insert(user_id.to_string(), role.clone());
                let id = ws.id.clone();
                self.log(
                    actor,
                    AuditAction::MemberAdded,
                    &id,
                    format!("{actor} added {user_id} as {role}"),
                );
                Ok(())
            }
        }
    }

    /// Self-join by presenting the workspace id — the server's WS join path
    /// (knowing the session id is the invitation). Idempotent: members keep
    /// their current role. New members land as Editor; Admin is only ever
    /// granted by an existing Admin through `add_member`.
    pub fn join(&mut self, workspace_id: &str, user_id: &str) -> Result<Role, WorkspaceError> {
        if let Some(role) = self.role_of(workspace_id, user_id) {
            return Ok(role);
        }
        let ws = self
            .workspaces
            .get_mut(workspace_id)
            .ok_or_else(|| WorkspaceError::NotFound(workspace_id.to_string()))?;
        ws.members.insert(user_id.to_string(), Role::Editor);
        let id = ws.id.clone();
        self.log(
            user_id,
            AuditAction::MemberAdded,
            &id,
            format!("{user_id} self-joined as editor"),
        );
        Ok(Role::Editor)
    }

    /// Record a denial decided outside the workspace mutators (e.g. the
    /// server's role gate on message relay). Same trail, same `Denied`
    /// action — the audit log must not depend on which layer caught it.
    pub fn log_denied(
        &mut self,
        actor: &str,
        workspace_id: &str,
        action: &str,
        has: Option<&Role>,
    ) {
        let has = has
            .map(|r| r.to_string())
            .unwrap_or_else(|| "non-member".to_string());
        self.log(
            actor,
            AuditAction::Denied,
            workspace_id,
            format!("{actor} attempted {action} (requires editor, has {has})"),
        );
    }

    /// Remove a member. Admins may remove anyone; anyone may remove
    /// themselves (leave). The last admin cannot be removed.
    pub fn remove_member(
        &mut self,
        workspace_id: &str,
        actor: &str,
        user_id: &str,
    ) -> Result<(), WorkspaceError> {
        let actor_role = self.role_of(workspace_id, actor);
        let is_self = actor == user_id;
        let permitted = actor_role.as_ref().is_some_and(|r| r.can(&Role::Admin)) || is_self;
        if !permitted {
            if self.workspaces.contains_key(workspace_id) {
                return Err(self.deny(
                    actor,
                    workspace_id,
                    "remove_member",
                    &Role::Admin,
                    actor_role.as_ref(),
                ));
            }
            return Err(WorkspaceError::NotFound(workspace_id.to_string()));
        }
        let ws = self.workspaces.get(workspace_id).unwrap();
        let target_role = ws.members.get(user_id).cloned();
        if target_role == Some(Role::Admin) && Self::admin_count(ws) == 1 {
            let id = ws.id.clone();
            self.log(
                actor,
                AuditAction::Denied,
                &id,
                format!("blocked removing last admin {user_id}"),
            );
            return Err(WorkspaceError::LastAdmin(id));
        }
        let ws = self.workspaces.get_mut(workspace_id).unwrap();
        ws.members.remove(user_id);
        let id = ws.id.clone();
        self.log(
            actor,
            AuditAction::MemberRemoved,
            &id,
            format!("{actor} removed {user_id}"),
        );
        Ok(())
    }

    /// List workspaces for a user.
    pub fn list_workspaces(&self, user_id: &str) -> Vec<&Workspace> {
        self.workspaces
            .values()
            .filter(|ws| ws.members.contains_key(user_id))
            .collect()
    }

    /// Get member count for a workspace.
    pub fn member_count(&self, workspace_id: &str) -> usize {
        self.workspaces
            .get(workspace_id)
            .map(|ws| ws.members.len())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_workspace() {
        let mut manager = WorkspaceManager::new();
        let ws = manager.create_workspace("test-project", "alice");
        assert_eq!(ws.name, "test-project");
        assert_eq!(ws.owner, "alice");
        assert!(ws.id.starts_with("ws-"));
        // Creation itself is audited.
        let events = manager.events_for(&ws.id);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].action, AuditAction::WorkspaceCreated);
        assert_eq!(events[0].actor, "alice");
    }

    #[test]
    fn test_add_member() {
        let mut manager = WorkspaceManager::new();
        let ws = manager.create_workspace("test", "alice");
        assert!(manager
            .add_member(&ws.id, "alice", "bob", Role::Editor)
            .is_ok());
        assert_eq!(manager.member_count(&ws.id), 2);
    }

    #[test]
    fn role_hierarchy_orders_viewer_editor_admin() {
        assert!(Role::Admin.can(&Role::Admin));
        assert!(Role::Admin.can(&Role::Editor));
        assert!(Role::Editor.can(&Role::Viewer));
        assert!(!Role::Editor.can(&Role::Admin));
        assert!(!Role::Viewer.can(&Role::Editor));
    }

    #[test]
    fn non_admin_add_is_forbidden_and_audited() {
        let mut manager = WorkspaceManager::new();
        let ws = manager.create_workspace("test", "alice");
        manager
            .add_member(&ws.id, "alice", "bob", Role::Viewer)
            .unwrap();
        let err = manager
            .add_member(&ws.id, "bob", "mallory", Role::Editor)
            .unwrap_err();
        assert!(matches!(err, WorkspaceError::Forbidden { .. }));
        assert_eq!(manager.member_count(&ws.id), 2);
        // The denial itself is in the trail.
        let last = manager.audit_log().last().unwrap();
        assert_eq!(last.action, AuditAction::Denied);
        assert_eq!(last.actor, "bob");
        assert!(last.detail.contains("add_member"));
    }

    #[test]
    fn role_change_logs_distinctly_and_same_role_is_quiet() {
        let mut manager = WorkspaceManager::new();
        let ws = manager.create_workspace("test", "alice");
        manager
            .add_member(&ws.id, "alice", "bob", Role::Viewer)
            .unwrap();
        let before = manager.audit_log().len();
        manager
            .add_member(&ws.id, "alice", "bob", Role::Editor)
            .unwrap();
        assert_eq!(manager.role_of(&ws.id, "bob"), Some(Role::Editor));
        assert_eq!(manager.audit_log()[before].action, AuditAction::RoleChanged);
        // Idempotent re-add: success, no new event.
        manager
            .add_member(&ws.id, "alice", "bob", Role::Editor)
            .unwrap();
        assert_eq!(manager.audit_log().len(), before + 1);
    }

    #[test]
    fn last_admin_cannot_be_removed_but_self_leave_works() {
        let mut manager = WorkspaceManager::new();
        let ws = manager.create_workspace("test", "alice");
        // Sole admin cannot remove herself.
        assert!(matches!(
            manager.remove_member(&ws.id, "alice", "alice"),
            Err(WorkspaceError::LastAdmin(_))
        ));
        // Blocked removals are audited as denials, not silently refused.
        let last = manager.audit_log().last().unwrap();
        assert_eq!(last.action, AuditAction::Denied);
        assert!(
            last.detail.contains("blocked removing last admin"),
            "{last:?}"
        );
        // With two admins, one may leave.
        manager
            .add_member(&ws.id, "alice", "bob", Role::Admin)
            .unwrap();
        manager.remove_member(&ws.id, "bob", "bob").unwrap();
        assert_eq!(manager.member_count(&ws.id), 1);
        // Non-admin cannot remove others, but may leave.
        manager
            .add_member(&ws.id, "alice", "carol", Role::Viewer)
            .unwrap();
        assert!(manager.remove_member(&ws.id, "carol", "alice").is_err());
        manager.remove_member(&ws.id, "carol", "carol").unwrap();
        assert_eq!(manager.member_count(&ws.id), 1);
    }

    #[test]
    fn last_admin_cannot_be_demoted_via_role_change() {
        let mut manager = WorkspaceManager::new();
        let ws = manager.create_workspace("test", "alice");
        let err = manager
            .add_member(&ws.id, "alice", "alice", Role::Viewer)
            .unwrap_err();
        assert!(matches!(err, WorkspaceError::LastAdmin(_)), "{err:?}");
        assert_eq!(manager.role_of(&ws.id, "alice"), Some(Role::Admin));
        let last = manager.audit_log().last().unwrap();
        assert_eq!(last.action, AuditAction::Denied);
        assert!(
            last.detail.contains("blocked demoting last admin"),
            "{last:?}"
        );
    }

    #[test]
    fn demotion_is_fine_when_a_second_admin_exists() {
        let mut manager = WorkspaceManager::new();
        let ws = manager.create_workspace("test", "alice");
        manager
            .add_member(&ws.id, "alice", "bob", Role::Admin)
            .unwrap();
        manager
            .add_member(&ws.id, "alice", "bob", Role::Viewer)
            .unwrap();
        assert_eq!(manager.role_of(&ws.id, "bob"), Some(Role::Viewer));
        let last = manager.audit_log().last().unwrap();
        assert_eq!(last.action, AuditAction::RoleChanged);
    }

    #[test]
    fn self_join_grants_editor_once_and_is_idempotent() {
        let mut manager = WorkspaceManager::new();
        let ws = manager.create_workspace("test", "alice");
        assert_eq!(manager.join(&ws.id, "bob").unwrap(), Role::Editor);
        assert_eq!(manager.role_of(&ws.id, "bob"), Some(Role::Editor));
        let before = manager.audit_log().len();
        assert_eq!(manager.join(&ws.id, "bob").unwrap(), Role::Editor);
        assert_eq!(manager.audit_log().len(), before, "re-join logs nothing");
        // Joining cannot escalate: the stored role is what an admin gave.
        manager
            .add_member(&ws.id, "alice", "carol", Role::Viewer)
            .unwrap();
        assert_eq!(manager.join(&ws.id, "carol").unwrap(), Role::Viewer);
    }

    #[test]
    fn join_unknown_workspace_is_not_found() {
        let mut manager = WorkspaceManager::new();
        assert!(matches!(
            manager.join("ws-missing", "bob"),
            Err(WorkspaceError::NotFound(_))
        ));
    }

    #[test]
    fn create_workspace_with_id_uses_the_given_id() {
        let mut manager = WorkspaceManager::new();
        let ws = manager.create_workspace_with_id("xencode-deadbeef", "s", "alice");
        assert_eq!(ws.id, "xencode-deadbeef");
        assert!(manager.get_workspace("xencode-deadbeef").is_some());
        // Caller-chosen ids are the caller's problem: same id overwrites
        // (the server loops until it draws a free one).
        let ws2 = manager.create_workspace_with_id("xencode-deadbeef", "s2", "bob");
        assert_eq!(ws2.name, "s2");
        assert_eq!(manager.member_count("xencode-deadbeef"), 1);
    }

    #[test]
    fn audit_seq_is_monotonic_and_events_filter_by_workspace() {
        let mut manager = WorkspaceManager::new();
        let a = manager.create_workspace("a", "alice");
        let b = manager.create_workspace("b", "alice");
        manager
            .add_member(&a.id, "alice", "bob", Role::Editor)
            .unwrap();
        let seqs: Vec<u64> = manager.audit_log().iter().map(|e| e.seq).collect();
        assert_eq!(seqs, vec![1, 2, 3]);
        assert_eq!(manager.events_for(&a.id).len(), 2);
        assert_eq!(manager.events_for(&b.id).len(), 1);
        assert!(manager.events_for("ws-missing").is_empty());
    }

    #[test]
    fn test_list_workspaces() {
        let mut manager = WorkspaceManager::new();
        manager.create_workspace("project-a", "alice");
        manager.create_workspace("project-b", "alice");

        let list = manager.list_workspaces("alice");
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_get_workspace_nonexistent() {
        let manager = WorkspaceManager::new();
        assert!(manager.get_workspace("nonexistent").is_none());
    }
}
