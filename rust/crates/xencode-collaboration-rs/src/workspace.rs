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

#[derive(Debug)]
pub struct WorkspaceError;

impl std::fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Workspace error")
    }
}

impl std::error::Error for WorkspaceError {}

/// In-memory workspace manager for collaboration sessions.
#[derive(Debug, Default)]
pub struct WorkspaceManager {
    workspaces: HashMap<String, Workspace>,
}

impl WorkspaceManager {
    pub fn new() -> Self {
        Self {
            workspaces: HashMap::new(),
        }
    }

    /// Create a new workspace.
    pub fn create_workspace(&mut self, name: &str, owner: &str) -> Workspace {
        let id = format!("ws-{}", &uuid::Uuid::new_v4().to_string()[..8]);
        let mut members = HashMap::new();
        members.insert(owner.to_string(), Role::Admin);

        let ws = Workspace {
            id: id.clone(),
            name: name.to_string(),
            owner: owner.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            members,
        };
        self.workspaces.insert(id, ws.clone());
        ws
    }

    /// Get workspace by ID.
    pub fn get_workspace(&self, id: &str) -> Option<&Workspace> {
        self.workspaces.get(id)
    }

    /// Add a member to a workspace.
    pub fn add_member(
        &mut self,
        workspace_id: &str,
        user_id: &str,
        role: Role,
    ) -> Result<(), WorkspaceError> {
        if let Some(ws) = self.workspaces.get_mut(workspace_id) {
            ws.members.insert(user_id.to_string(), role);
            Ok(())
        } else {
            Err(WorkspaceError)
        }
    }

    /// Remove a member from a workspace.
    pub fn remove_member(&mut self, workspace_id: &str, user_id: &str) -> Result<(), WorkspaceError> {
        if let Some(ws) = self.workspaces.get_mut(workspace_id) {
            ws.members.remove(user_id);
            Ok(())
        } else {
            Err(WorkspaceError)
        }
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
    }

    #[test]
    fn test_add_member() {
        let mut manager = WorkspaceManager::new();
        let ws = manager.create_workspace("test", "alice");
        assert!(manager.add_member(&ws.id, "bob", Role::Editor).is_ok());
        assert_eq!(manager.member_count(&ws.id), 2);
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
