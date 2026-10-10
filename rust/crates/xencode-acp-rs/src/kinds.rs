//! Tool names as ACP tool kinds, and the file texts edits carry as diffs
//! (M-7b).

use std::path::Path;

use agent_client_protocol::schema::v1::ToolKind;

/// The largest file, on either side of an edit, whose text is sent as a
/// diff. Larger files, and files that are not UTF-8 text, are reported
/// without one.
pub const DIFF_LIMIT: u64 = 512 * 1024;

/// The ACP kind of one of xencode's tools.
pub fn kind_of(tool: &str) -> ToolKind {
    match tool {
        "read_file" | "list_dir" | "read_docs" => ToolKind::Read,
        "write_file" | "edit_file" | "edit_symbol" | "ast_edit" => ToolKind::Edit,
        "search_files" | "find_refs" | "web_search" => ToolKind::Search,
        "run_command" | "background_start" => ToolKind::Execute,
        "web_fetch" => ToolKind::Fetch,
        _ => ToolKind::Other,
    }
}

/// The file a tool call works on, from its arguments.
pub fn path_of(arguments: &serde_json::Value) -> Option<String> {
    ["path", "file", "file_path"]
        .iter()
        .find_map(|key| arguments.get(key).and_then(|v| v.as_str()))
        .map(str::to_string)
}

/// The file's text, when it is UTF-8 and no larger than [`DIFF_LIMIT`].
pub fn readable(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > DIFF_LIMIT {
        return None;
    }
    String::from_utf8(std::fs::read(path).ok()?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::ToolKind;

    #[test]
    fn every_tool_gets_its_kind() {
        for (tool, kind) in [
            ("read_file", ToolKind::Read),
            ("list_dir", ToolKind::Read),
            ("read_docs", ToolKind::Read),
            ("write_file", ToolKind::Edit),
            ("edit_file", ToolKind::Edit),
            ("edit_symbol", ToolKind::Edit),
            ("ast_edit", ToolKind::Edit),
            ("search_files", ToolKind::Search),
            ("find_refs", ToolKind::Search),
            ("web_search", ToolKind::Search),
            ("run_command", ToolKind::Execute),
            ("background_start", ToolKind::Execute),
            ("web_fetch", ToolKind::Fetch),
            ("update_plan", ToolKind::Other),
            ("mcp__github__search", ToolKind::Other),
        ] {
            assert_eq!(kind_of(tool), kind, "{tool}");
        }
    }

    #[test]
    fn the_path_is_found_under_its_usual_names() {
        assert_eq!(
            path_of(&serde_json::json!({"path": "a.rs", "content": "x"})).as_deref(),
            Some("a.rs")
        );
        assert_eq!(
            path_of(&serde_json::json!({"file_path": "b.rs"})).as_deref(),
            Some("b.rs")
        );
        assert_eq!(path_of(&serde_json::json!({"command": "ls"})), None);
    }

    /// Review Focus 5.
    #[test]
    fn only_modest_utf8_files_get_a_diff() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        std::fs::write(dir.path().join("b.bin"), [0xff, 0xfe, 0x00]).unwrap();
        std::fs::write(
            dir.path().join("big.txt"),
            "x".repeat(DIFF_LIMIT as usize + 1),
        )
        .unwrap();
        assert_eq!(
            readable(&dir.path().join("a.txt")).as_deref(),
            Some("hello")
        );
        assert_eq!(readable(&dir.path().join("b.bin")), None);
        assert_eq!(readable(&dir.path().join("big.txt")), None);
        assert_eq!(readable(&dir.path().join("missing.txt")), None);
    }
}
